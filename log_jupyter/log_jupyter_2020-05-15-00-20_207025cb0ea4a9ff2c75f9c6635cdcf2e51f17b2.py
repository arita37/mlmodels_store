
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
 40%|████      | 2/5 [00:52<01:19, 26.44s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.1792328696706846, 'embedding_size_factor': 1.335536420854233, 'layers.choice': 1, 'learning_rate': 0.00011911115173385877, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.005340564725001667} and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc6\xf1\x1aH\xcdI\xffX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5^[p"\xf9"X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1f9j\x01Q\x11\xbaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?u\xdf\xfc\xed`y\x86u.' and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc6\xf1\x1aH\xcdI\xffX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5^[p"\xf9"X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1f9j\x01Q\x11\xbaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?u\xdf\xfc\xed`y\x86u.' and reward: 0.3786
 60%|██████    | 3/5 [02:52<01:48, 54.37s/it] 60%|██████    | 3/5 [02:52<01:54, 57.48s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4068525520501145, 'embedding_size_factor': 0.9495570286128341, 'layers.choice': 3, 'learning_rate': 0.0003916949129950069, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.644733947256832e-08} and reward: 0.373
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xda\t\xdfIV\\<X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeeb\xc5k\xf2\x87\xa6X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?9\xab\x8c\xd7`\xf3\xb6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>c\x91H\xc0\xefB\x8eu.' and reward: 0.373
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xda\t\xdfIV\\<X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeeb\xc5k\xf2\x87\xa6X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?9\xab\x8c\xd7`\xf3\xb6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>c\x91H\xc0\xefB\x8eu.' and reward: 0.373
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 228.91593551635742
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -111.69s of remaining time.
Ensemble size: 16
Ensemble weights: 
[0.6875 0.     0.3125]
	0.39	 = Validation accuracy score
	1.07s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 232.8s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f2bdfc02a90> 

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
 [ 0.12794836  0.12548262  0.17503378 -0.23656812  0.09337234  0.00110254]
 [ 0.09912378  0.14203623  0.13649142 -0.06535603  0.01561545 -0.00573776]
 [ 0.10147196  0.27937958  0.31619149 -0.29651836  0.04415759 -0.13316165]
 [ 0.08054693  0.21550825  0.45575896 -0.26241431  0.01329505  0.09849182]
 [ 0.64131004 -0.19729102  0.48728579  0.81941199  0.81625366  0.77306283]
 [-0.23871915  0.16466205  0.32091534  0.40048325  0.69413489 -0.0118738 ]
 [ 0.20200767 -0.44136965 -0.23155916 -0.38901734 -0.22976966 -0.12586564]
 [ 0.12601429  0.29580587  0.46921483  0.11223648  0.52518952 -0.00920465]
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
{'loss': 0.5481912419199944, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 00:25:23.394771: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4147076681256294, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 00:25:24.557640: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3833856/17464789 [=====>........................] - ETA: 0s
11223040/17464789 [==================>...........] - ETA: 0s
17432576/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 00:25:36.540316: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 00:25:36.544199: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-15 00:25:36.544354: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c2d6bde40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:25:36.544368: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:42 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 2:53 - loss: 8.1458 - accuracy: 0.4688 
   96/25000 [..............................] - ETA: 2:16 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 1:58 - loss: 8.9843 - accuracy: 0.4141
  160/25000 [..............................] - ETA: 1:48 - loss: 9.0083 - accuracy: 0.4125
  192/25000 [..............................] - ETA: 1:42 - loss: 9.1041 - accuracy: 0.4062
  224/25000 [..............................] - ETA: 1:37 - loss: 8.7619 - accuracy: 0.4286
  256/25000 [..............................] - ETA: 1:33 - loss: 8.7447 - accuracy: 0.4297
  288/25000 [..............................] - ETA: 1:30 - loss: 8.7314 - accuracy: 0.4306
  320/25000 [..............................] - ETA: 1:27 - loss: 8.7208 - accuracy: 0.4313
  352/25000 [..............................] - ETA: 1:25 - loss: 8.4507 - accuracy: 0.4489
  384/25000 [..............................] - ETA: 1:23 - loss: 8.5052 - accuracy: 0.4453
  416/25000 [..............................] - ETA: 1:21 - loss: 8.4038 - accuracy: 0.4519
  448/25000 [..............................] - ETA: 1:20 - loss: 8.4196 - accuracy: 0.4509
  480/25000 [..............................] - ETA: 1:19 - loss: 8.3694 - accuracy: 0.4542
  512/25000 [..............................] - ETA: 1:18 - loss: 8.2955 - accuracy: 0.4590
  544/25000 [..............................] - ETA: 1:17 - loss: 8.2867 - accuracy: 0.4596
  576/25000 [..............................] - ETA: 1:16 - loss: 8.2256 - accuracy: 0.4635
  608/25000 [..............................] - ETA: 1:16 - loss: 8.2214 - accuracy: 0.4638
  640/25000 [..............................] - ETA: 1:15 - loss: 8.1458 - accuracy: 0.4688
  672/25000 [..............................] - ETA: 1:14 - loss: 8.0545 - accuracy: 0.4747
  704/25000 [..............................] - ETA: 1:13 - loss: 8.1458 - accuracy: 0.4688
  736/25000 [..............................] - ETA: 1:13 - loss: 8.1874 - accuracy: 0.4660
  768/25000 [..............................] - ETA: 1:12 - loss: 8.2057 - accuracy: 0.4648
  800/25000 [..............................] - ETA: 1:12 - loss: 8.1649 - accuracy: 0.4675
  832/25000 [..............................] - ETA: 1:12 - loss: 8.1089 - accuracy: 0.4712
  864/25000 [>.............................] - ETA: 1:11 - loss: 8.0393 - accuracy: 0.4757
  896/25000 [>.............................] - ETA: 1:11 - loss: 8.0260 - accuracy: 0.4766
  928/25000 [>.............................] - ETA: 1:10 - loss: 8.0466 - accuracy: 0.4752
  960/25000 [>.............................] - ETA: 1:10 - loss: 8.0340 - accuracy: 0.4760
  992/25000 [>.............................] - ETA: 1:10 - loss: 8.0530 - accuracy: 0.4748
 1024/25000 [>.............................] - ETA: 1:09 - loss: 8.0559 - accuracy: 0.4746
 1056/25000 [>.............................] - ETA: 1:09 - loss: 8.0441 - accuracy: 0.4754
 1088/25000 [>.............................] - ETA: 1:09 - loss: 8.0189 - accuracy: 0.4770
 1120/25000 [>.............................] - ETA: 1:09 - loss: 7.9952 - accuracy: 0.4786
 1152/25000 [>.............................] - ETA: 1:08 - loss: 8.0393 - accuracy: 0.4757
 1184/25000 [>.............................] - ETA: 1:08 - loss: 7.9774 - accuracy: 0.4797
 1216/25000 [>.............................] - ETA: 1:08 - loss: 7.9945 - accuracy: 0.4786
 1248/25000 [>.............................] - ETA: 1:08 - loss: 7.9861 - accuracy: 0.4792
 1280/25000 [>.............................] - ETA: 1:08 - loss: 7.9661 - accuracy: 0.4805
 1312/25000 [>.............................] - ETA: 1:07 - loss: 7.9588 - accuracy: 0.4809
 1344/25000 [>.............................] - ETA: 1:07 - loss: 8.0089 - accuracy: 0.4777
 1376/25000 [>.............................] - ETA: 1:07 - loss: 7.9675 - accuracy: 0.4804
 1408/25000 [>.............................] - ETA: 1:07 - loss: 7.9498 - accuracy: 0.4815
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.9222 - accuracy: 0.4833
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.9583 - accuracy: 0.4810
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.9725 - accuracy: 0.4801
 1536/25000 [>.............................] - ETA: 1:06 - loss: 7.9661 - accuracy: 0.4805
 1568/25000 [>.............................] - ETA: 1:06 - loss: 7.9502 - accuracy: 0.4815
 1600/25000 [>.............................] - ETA: 1:06 - loss: 7.9158 - accuracy: 0.4837
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.9391 - accuracy: 0.4822
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.9523 - accuracy: 0.4814
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.9378 - accuracy: 0.4823
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.9594 - accuracy: 0.4809
 1760/25000 [=>............................] - ETA: 1:05 - loss: 7.9803 - accuracy: 0.4795
 1792/25000 [=>............................] - ETA: 1:05 - loss: 8.0174 - accuracy: 0.4771
 1824/25000 [=>............................] - ETA: 1:04 - loss: 8.0365 - accuracy: 0.4759
 1856/25000 [=>............................] - ETA: 1:04 - loss: 8.0301 - accuracy: 0.4763
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.9996 - accuracy: 0.4783
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.9781 - accuracy: 0.4797
 1952/25000 [=>............................] - ETA: 1:04 - loss: 7.9808 - accuracy: 0.4795
 1984/25000 [=>............................] - ETA: 1:04 - loss: 7.9680 - accuracy: 0.4803
 2016/25000 [=>............................] - ETA: 1:03 - loss: 7.9176 - accuracy: 0.4836
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.8987 - accuracy: 0.4849
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.8804 - accuracy: 0.4861
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.8989 - accuracy: 0.4848
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.8883 - accuracy: 0.4855
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.8851 - accuracy: 0.4858
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.8819 - accuracy: 0.4860
 2240/25000 [=>............................] - ETA: 1:03 - loss: 7.8925 - accuracy: 0.4853
 2272/25000 [=>............................] - ETA: 1:03 - loss: 7.8826 - accuracy: 0.4859
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.8995 - accuracy: 0.4848
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.8832 - accuracy: 0.4859
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.8609 - accuracy: 0.4873
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.8711 - accuracy: 0.4867
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.8810 - accuracy: 0.4860
 2464/25000 [=>............................] - ETA: 1:02 - loss: 7.8782 - accuracy: 0.4862
 2496/25000 [=>............................] - ETA: 1:02 - loss: 7.8509 - accuracy: 0.4880
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.8850 - accuracy: 0.4858
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.8583 - accuracy: 0.4875
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.8737 - accuracy: 0.4865
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.8945 - accuracy: 0.4851
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.8918 - accuracy: 0.4853
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.8891 - accuracy: 0.4855
 2720/25000 [==>...........................] - ETA: 1:00 - loss: 7.9203 - accuracy: 0.4835
 2752/25000 [==>...........................] - ETA: 1:00 - loss: 7.9173 - accuracy: 0.4836
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.8869 - accuracy: 0.4856
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.8953 - accuracy: 0.4851
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.8820 - accuracy: 0.4860
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.8796 - accuracy: 0.4861
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.8720 - accuracy: 0.4866
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.8906 - accuracy: 0.4854
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.8985 - accuracy: 0.4849
 3008/25000 [==>...........................] - ETA: 59s - loss: 7.9062 - accuracy: 0.4844 
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.8936 - accuracy: 0.4852
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.8962 - accuracy: 0.4850
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.8988 - accuracy: 0.4849
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.8769 - accuracy: 0.4863
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.8699 - accuracy: 0.4867
 3200/25000 [==>...........................] - ETA: 59s - loss: 7.8870 - accuracy: 0.4856
 3232/25000 [==>...........................] - ETA: 59s - loss: 7.9038 - accuracy: 0.4845
 3264/25000 [==>...........................] - ETA: 58s - loss: 7.9250 - accuracy: 0.4831
 3296/25000 [==>...........................] - ETA: 58s - loss: 7.9039 - accuracy: 0.4845
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.8878 - accuracy: 0.4856
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.8720 - accuracy: 0.4866
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.8520 - accuracy: 0.4879
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.8323 - accuracy: 0.4892
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.8219 - accuracy: 0.4899
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.8161 - accuracy: 0.4903
 3520/25000 [===>..........................] - ETA: 58s - loss: 7.8017 - accuracy: 0.4912
 3552/25000 [===>..........................] - ETA: 57s - loss: 7.8263 - accuracy: 0.4896
 3584/25000 [===>..........................] - ETA: 57s - loss: 7.8121 - accuracy: 0.4905
 3616/25000 [===>..........................] - ETA: 57s - loss: 7.8108 - accuracy: 0.4906
 3648/25000 [===>..........................] - ETA: 57s - loss: 7.8053 - accuracy: 0.4910
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.8000 - accuracy: 0.4913
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.7864 - accuracy: 0.4922
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.7977 - accuracy: 0.4915
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.8006 - accuracy: 0.4913
 3808/25000 [===>..........................] - ETA: 57s - loss: 7.7955 - accuracy: 0.4916
 3840/25000 [===>..........................] - ETA: 57s - loss: 7.8024 - accuracy: 0.4911
 3872/25000 [===>..........................] - ETA: 57s - loss: 7.8092 - accuracy: 0.4907
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.8237 - accuracy: 0.4898
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.8185 - accuracy: 0.4901
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 56s - loss: 7.8200 - accuracy: 0.4900
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.8111 - accuracy: 0.4906
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.7911 - accuracy: 0.4919
 4096/25000 [===>..........................] - ETA: 56s - loss: 7.7902 - accuracy: 0.4919
 4128/25000 [===>..........................] - ETA: 56s - loss: 7.7855 - accuracy: 0.4922
 4160/25000 [===>..........................] - ETA: 56s - loss: 7.7809 - accuracy: 0.4925
 4192/25000 [====>.........................] - ETA: 56s - loss: 7.7873 - accuracy: 0.4921
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.8009 - accuracy: 0.4912
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.8035 - accuracy: 0.4911
 4288/25000 [====>.........................] - ETA: 55s - loss: 7.8025 - accuracy: 0.4911
 4320/25000 [====>.........................] - ETA: 55s - loss: 7.7908 - accuracy: 0.4919
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.8005 - accuracy: 0.4913
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.7995 - accuracy: 0.4913
 4416/25000 [====>.........................] - ETA: 55s - loss: 7.8020 - accuracy: 0.4912
 4448/25000 [====>.........................] - ETA: 55s - loss: 7.8080 - accuracy: 0.4908
 4480/25000 [====>.........................] - ETA: 55s - loss: 7.8035 - accuracy: 0.4911
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.7924 - accuracy: 0.4918
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.8016 - accuracy: 0.4912
 4576/25000 [====>.........................] - ETA: 54s - loss: 7.7973 - accuracy: 0.4915
 4608/25000 [====>.........................] - ETA: 54s - loss: 7.7964 - accuracy: 0.4915
 4640/25000 [====>.........................] - ETA: 54s - loss: 7.7922 - accuracy: 0.4918
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.7946 - accuracy: 0.4917
 4704/25000 [====>.........................] - ETA: 54s - loss: 7.7905 - accuracy: 0.4919
 4736/25000 [====>.........................] - ETA: 54s - loss: 7.7961 - accuracy: 0.4916
 4768/25000 [====>.........................] - ETA: 54s - loss: 7.7888 - accuracy: 0.4920
 4800/25000 [====>.........................] - ETA: 54s - loss: 7.7752 - accuracy: 0.4929
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.7650 - accuracy: 0.4936
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.7643 - accuracy: 0.4936
 4896/25000 [====>.........................] - ETA: 53s - loss: 7.7700 - accuracy: 0.4933
 4928/25000 [====>.........................] - ETA: 53s - loss: 7.7631 - accuracy: 0.4937
 4960/25000 [====>.........................] - ETA: 53s - loss: 7.7810 - accuracy: 0.4925
 4992/25000 [====>.........................] - ETA: 53s - loss: 7.8018 - accuracy: 0.4912
 5024/25000 [=====>........................] - ETA: 53s - loss: 7.7979 - accuracy: 0.4914
 5056/25000 [=====>........................] - ETA: 53s - loss: 7.7879 - accuracy: 0.4921
 5088/25000 [=====>........................] - ETA: 53s - loss: 7.8052 - accuracy: 0.4910
 5120/25000 [=====>........................] - ETA: 53s - loss: 7.8044 - accuracy: 0.4910
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.8005 - accuracy: 0.4913
 5184/25000 [=====>........................] - ETA: 52s - loss: 7.7908 - accuracy: 0.4919
 5216/25000 [=====>........................] - ETA: 52s - loss: 7.8018 - accuracy: 0.4912
 5248/25000 [=====>........................] - ETA: 52s - loss: 7.7952 - accuracy: 0.4916
 5280/25000 [=====>........................] - ETA: 52s - loss: 7.8060 - accuracy: 0.4909
 5312/25000 [=====>........................] - ETA: 52s - loss: 7.8081 - accuracy: 0.4908
 5344/25000 [=====>........................] - ETA: 52s - loss: 7.7986 - accuracy: 0.4914
 5376/25000 [=====>........................] - ETA: 52s - loss: 7.8007 - accuracy: 0.4913
 5408/25000 [=====>........................] - ETA: 52s - loss: 7.8055 - accuracy: 0.4909
 5440/25000 [=====>........................] - ETA: 52s - loss: 7.8019 - accuracy: 0.4912
 5472/25000 [=====>........................] - ETA: 51s - loss: 7.8067 - accuracy: 0.4909
 5504/25000 [=====>........................] - ETA: 51s - loss: 7.8171 - accuracy: 0.4902
 5536/25000 [=====>........................] - ETA: 51s - loss: 7.8245 - accuracy: 0.4897
 5568/25000 [=====>........................] - ETA: 51s - loss: 7.8236 - accuracy: 0.4898
 5600/25000 [=====>........................] - ETA: 51s - loss: 7.8309 - accuracy: 0.4893
 5632/25000 [=====>........................] - ETA: 51s - loss: 7.8109 - accuracy: 0.4906
 5664/25000 [=====>........................] - ETA: 51s - loss: 7.8074 - accuracy: 0.4908
 5696/25000 [=====>........................] - ETA: 51s - loss: 7.8120 - accuracy: 0.4905
 5728/25000 [=====>........................] - ETA: 51s - loss: 7.8112 - accuracy: 0.4906
 5760/25000 [=====>........................] - ETA: 51s - loss: 7.8024 - accuracy: 0.4911
 5792/25000 [=====>........................] - ETA: 50s - loss: 7.8043 - accuracy: 0.4910
 5824/25000 [=====>........................] - ETA: 50s - loss: 7.8009 - accuracy: 0.4912
 5856/25000 [======>.......................] - ETA: 50s - loss: 7.8028 - accuracy: 0.4911
 5888/25000 [======>.......................] - ETA: 50s - loss: 7.8125 - accuracy: 0.4905
 5920/25000 [======>.......................] - ETA: 50s - loss: 7.8065 - accuracy: 0.4909
 5952/25000 [======>.......................] - ETA: 50s - loss: 7.8032 - accuracy: 0.4911
 5984/25000 [======>.......................] - ETA: 50s - loss: 7.8024 - accuracy: 0.4911
 6016/25000 [======>.......................] - ETA: 50s - loss: 7.8043 - accuracy: 0.4910
 6048/25000 [======>.......................] - ETA: 50s - loss: 7.8137 - accuracy: 0.4904
 6080/25000 [======>.......................] - ETA: 50s - loss: 7.8129 - accuracy: 0.4905
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.8197 - accuracy: 0.4900
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.8164 - accuracy: 0.4902
 6176/25000 [======>.......................] - ETA: 49s - loss: 7.8081 - accuracy: 0.4908
 6208/25000 [======>.......................] - ETA: 49s - loss: 7.8198 - accuracy: 0.4900
 6240/25000 [======>.......................] - ETA: 49s - loss: 7.8165 - accuracy: 0.4902
 6272/25000 [======>.......................] - ETA: 49s - loss: 7.8182 - accuracy: 0.4901
 6304/25000 [======>.......................] - ETA: 49s - loss: 7.8223 - accuracy: 0.4898
 6336/25000 [======>.......................] - ETA: 49s - loss: 7.8215 - accuracy: 0.4899
 6368/25000 [======>.......................] - ETA: 49s - loss: 7.8231 - accuracy: 0.4898
 6400/25000 [======>.......................] - ETA: 49s - loss: 7.8176 - accuracy: 0.4902
 6432/25000 [======>.......................] - ETA: 49s - loss: 7.8097 - accuracy: 0.4907
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.8089 - accuracy: 0.4907
 6496/25000 [======>.......................] - ETA: 48s - loss: 7.8082 - accuracy: 0.4908
 6528/25000 [======>.......................] - ETA: 48s - loss: 7.8122 - accuracy: 0.4905
 6560/25000 [======>.......................] - ETA: 48s - loss: 7.8092 - accuracy: 0.4907
 6592/25000 [======>.......................] - ETA: 48s - loss: 7.8085 - accuracy: 0.4907
 6624/25000 [======>.......................] - ETA: 48s - loss: 7.8032 - accuracy: 0.4911
 6656/25000 [======>.......................] - ETA: 48s - loss: 7.7910 - accuracy: 0.4919
 6688/25000 [=======>......................] - ETA: 48s - loss: 7.7996 - accuracy: 0.4913
 6720/25000 [=======>......................] - ETA: 48s - loss: 7.8035 - accuracy: 0.4911
 6752/25000 [=======>......................] - ETA: 48s - loss: 7.7983 - accuracy: 0.4914
 6784/25000 [=======>......................] - ETA: 48s - loss: 7.8022 - accuracy: 0.4912
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.7926 - accuracy: 0.4918
 6848/25000 [=======>......................] - ETA: 47s - loss: 7.7808 - accuracy: 0.4926
 6880/25000 [=======>......................] - ETA: 47s - loss: 7.7825 - accuracy: 0.4924
 6912/25000 [=======>......................] - ETA: 47s - loss: 7.7908 - accuracy: 0.4919
 6944/25000 [=======>......................] - ETA: 47s - loss: 7.7881 - accuracy: 0.4921
 6976/25000 [=======>......................] - ETA: 47s - loss: 7.7875 - accuracy: 0.4921
 7008/25000 [=======>......................] - ETA: 47s - loss: 7.7848 - accuracy: 0.4923
 7040/25000 [=======>......................] - ETA: 47s - loss: 7.7821 - accuracy: 0.4925
 7072/25000 [=======>......................] - ETA: 47s - loss: 7.7772 - accuracy: 0.4928
 7104/25000 [=======>......................] - ETA: 47s - loss: 7.7896 - accuracy: 0.4920
 7136/25000 [=======>......................] - ETA: 47s - loss: 7.7977 - accuracy: 0.4915
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.7928 - accuracy: 0.4918
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.7837 - accuracy: 0.4924
 7232/25000 [=======>......................] - ETA: 46s - loss: 7.7747 - accuracy: 0.4929
 7264/25000 [=======>......................] - ETA: 46s - loss: 7.7764 - accuracy: 0.4928
 7296/25000 [=======>......................] - ETA: 46s - loss: 7.7780 - accuracy: 0.4927
 7328/25000 [=======>......................] - ETA: 46s - loss: 7.7859 - accuracy: 0.4922
 7360/25000 [=======>......................] - ETA: 46s - loss: 7.7833 - accuracy: 0.4924
 7392/25000 [=======>......................] - ETA: 46s - loss: 7.7745 - accuracy: 0.4930
 7424/25000 [=======>......................] - ETA: 46s - loss: 7.7637 - accuracy: 0.4937
 7456/25000 [=======>......................] - ETA: 46s - loss: 7.7653 - accuracy: 0.4936
 7488/25000 [=======>......................] - ETA: 46s - loss: 7.7608 - accuracy: 0.4939
 7520/25000 [========>.....................] - ETA: 46s - loss: 7.7625 - accuracy: 0.4938
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.7641 - accuracy: 0.4936
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.7657 - accuracy: 0.4935
 7616/25000 [========>.....................] - ETA: 45s - loss: 7.7592 - accuracy: 0.4940
 7648/25000 [========>.....................] - ETA: 45s - loss: 7.7588 - accuracy: 0.4940
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.7525 - accuracy: 0.4944
 7712/25000 [========>.....................] - ETA: 45s - loss: 7.7541 - accuracy: 0.4943
 7744/25000 [========>.....................] - ETA: 45s - loss: 7.7478 - accuracy: 0.4947
 7776/25000 [========>.....................] - ETA: 45s - loss: 7.7455 - accuracy: 0.4949
 7808/25000 [========>.....................] - ETA: 45s - loss: 7.7432 - accuracy: 0.4950
 7840/25000 [========>.....................] - ETA: 45s - loss: 7.7488 - accuracy: 0.4946
 7872/25000 [========>.....................] - ETA: 45s - loss: 7.7562 - accuracy: 0.4942
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.7559 - accuracy: 0.4942
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.7613 - accuracy: 0.4938
 7968/25000 [========>.....................] - ETA: 44s - loss: 7.7590 - accuracy: 0.4940
 8000/25000 [========>.....................] - ETA: 44s - loss: 7.7548 - accuracy: 0.4942
 8032/25000 [========>.....................] - ETA: 44s - loss: 7.7602 - accuracy: 0.4939
 8064/25000 [========>.....................] - ETA: 44s - loss: 7.7579 - accuracy: 0.4940
 8096/25000 [========>.....................] - ETA: 44s - loss: 7.7518 - accuracy: 0.4944
 8128/25000 [========>.....................] - ETA: 44s - loss: 7.7553 - accuracy: 0.4942
 8160/25000 [========>.....................] - ETA: 44s - loss: 7.7474 - accuracy: 0.4947
 8192/25000 [========>.....................] - ETA: 44s - loss: 7.7508 - accuracy: 0.4945
 8224/25000 [========>.....................] - ETA: 44s - loss: 7.7561 - accuracy: 0.4942
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.7521 - accuracy: 0.4944
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.7536 - accuracy: 0.4943
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.7403 - accuracy: 0.4952
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.7309 - accuracy: 0.4958
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.7325 - accuracy: 0.4957
 8416/25000 [=========>....................] - ETA: 43s - loss: 7.7322 - accuracy: 0.4957
 8448/25000 [=========>....................] - ETA: 43s - loss: 7.7320 - accuracy: 0.4957
 8480/25000 [=========>....................] - ETA: 43s - loss: 7.7209 - accuracy: 0.4965
 8512/25000 [=========>....................] - ETA: 43s - loss: 7.7207 - accuracy: 0.4965
 8544/25000 [=========>....................] - ETA: 43s - loss: 7.7169 - accuracy: 0.4967
 8576/25000 [=========>....................] - ETA: 43s - loss: 7.7149 - accuracy: 0.4969
 8608/25000 [=========>....................] - ETA: 43s - loss: 7.7147 - accuracy: 0.4969
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.7163 - accuracy: 0.4968
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.7197 - accuracy: 0.4965
 8704/25000 [=========>....................] - ETA: 42s - loss: 7.7142 - accuracy: 0.4969
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.7123 - accuracy: 0.4970
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.7103 - accuracy: 0.4971
 8800/25000 [=========>....................] - ETA: 42s - loss: 7.7119 - accuracy: 0.4970
 8832/25000 [=========>....................] - ETA: 42s - loss: 7.7118 - accuracy: 0.4971
 8864/25000 [=========>....................] - ETA: 42s - loss: 7.7116 - accuracy: 0.4971
 8896/25000 [=========>....................] - ETA: 42s - loss: 7.7097 - accuracy: 0.4972
 8928/25000 [=========>....................] - ETA: 42s - loss: 7.7061 - accuracy: 0.4974
 8960/25000 [=========>....................] - ETA: 42s - loss: 7.7008 - accuracy: 0.4978
 8992/25000 [=========>....................] - ETA: 42s - loss: 7.6939 - accuracy: 0.4982
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.7023 - accuracy: 0.4977
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.7022 - accuracy: 0.4977
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.7004 - accuracy: 0.4978
 9120/25000 [=========>....................] - ETA: 41s - loss: 7.7036 - accuracy: 0.4976
 9152/25000 [=========>....................] - ETA: 41s - loss: 7.6985 - accuracy: 0.4979
 9184/25000 [==========>...................] - ETA: 41s - loss: 7.7033 - accuracy: 0.4976
 9216/25000 [==========>...................] - ETA: 41s - loss: 7.7032 - accuracy: 0.4976
 9248/25000 [==========>...................] - ETA: 41s - loss: 7.7097 - accuracy: 0.4972
 9280/25000 [==========>...................] - ETA: 41s - loss: 7.7096 - accuracy: 0.4972
 9312/25000 [==========>...................] - ETA: 41s - loss: 7.7111 - accuracy: 0.4971
 9344/25000 [==========>...................] - ETA: 41s - loss: 7.7044 - accuracy: 0.4975
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.7059 - accuracy: 0.4974
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.7090 - accuracy: 0.4972
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.7153 - accuracy: 0.4968
 9472/25000 [==========>...................] - ETA: 40s - loss: 7.7184 - accuracy: 0.4966
 9504/25000 [==========>...................] - ETA: 40s - loss: 7.7215 - accuracy: 0.4964
 9536/25000 [==========>...................] - ETA: 40s - loss: 7.7245 - accuracy: 0.4962
 9568/25000 [==========>...................] - ETA: 40s - loss: 7.7227 - accuracy: 0.4963
 9600/25000 [==========>...................] - ETA: 40s - loss: 7.7193 - accuracy: 0.4966
 9632/25000 [==========>...................] - ETA: 40s - loss: 7.7176 - accuracy: 0.4967
 9664/25000 [==========>...................] - ETA: 40s - loss: 7.7126 - accuracy: 0.4970
 9696/25000 [==========>...................] - ETA: 40s - loss: 7.7109 - accuracy: 0.4971
 9728/25000 [==========>...................] - ETA: 40s - loss: 7.7044 - accuracy: 0.4975
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.7059 - accuracy: 0.4974
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.7073 - accuracy: 0.4973
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.7072 - accuracy: 0.4974
 9856/25000 [==========>...................] - ETA: 39s - loss: 7.7040 - accuracy: 0.4976
 9888/25000 [==========>...................] - ETA: 39s - loss: 7.7100 - accuracy: 0.4972
 9920/25000 [==========>...................] - ETA: 39s - loss: 7.7145 - accuracy: 0.4969
 9952/25000 [==========>...................] - ETA: 39s - loss: 7.7082 - accuracy: 0.4973
 9984/25000 [==========>...................] - ETA: 39s - loss: 7.7065 - accuracy: 0.4974
10016/25000 [===========>..................] - ETA: 39s - loss: 7.7018 - accuracy: 0.4977
10048/25000 [===========>..................] - ETA: 39s - loss: 7.7078 - accuracy: 0.4973
10080/25000 [===========>..................] - ETA: 39s - loss: 7.7153 - accuracy: 0.4968
10112/25000 [===========>..................] - ETA: 39s - loss: 7.7242 - accuracy: 0.4962
10144/25000 [===========>..................] - ETA: 38s - loss: 7.7225 - accuracy: 0.4964
10176/25000 [===========>..................] - ETA: 38s - loss: 7.7329 - accuracy: 0.4957
10208/25000 [===========>..................] - ETA: 38s - loss: 7.7357 - accuracy: 0.4955
10240/25000 [===========>..................] - ETA: 38s - loss: 7.7445 - accuracy: 0.4949
10272/25000 [===========>..................] - ETA: 38s - loss: 7.7427 - accuracy: 0.4950
10304/25000 [===========>..................] - ETA: 38s - loss: 7.7440 - accuracy: 0.4950
10336/25000 [===========>..................] - ETA: 38s - loss: 7.7423 - accuracy: 0.4951
10368/25000 [===========>..................] - ETA: 38s - loss: 7.7480 - accuracy: 0.4947
10400/25000 [===========>..................] - ETA: 38s - loss: 7.7433 - accuracy: 0.4950
10432/25000 [===========>..................] - ETA: 38s - loss: 7.7460 - accuracy: 0.4948
10464/25000 [===========>..................] - ETA: 38s - loss: 7.7487 - accuracy: 0.4946
10496/25000 [===========>..................] - ETA: 37s - loss: 7.7440 - accuracy: 0.4950
10528/25000 [===========>..................] - ETA: 37s - loss: 7.7351 - accuracy: 0.4955
10560/25000 [===========>..................] - ETA: 37s - loss: 7.7305 - accuracy: 0.4958
10592/25000 [===========>..................] - ETA: 37s - loss: 7.7303 - accuracy: 0.4958
10624/25000 [===========>..................] - ETA: 37s - loss: 7.7316 - accuracy: 0.4958
10656/25000 [===========>..................] - ETA: 37s - loss: 7.7299 - accuracy: 0.4959
10688/25000 [===========>..................] - ETA: 37s - loss: 7.7355 - accuracy: 0.4955
10720/25000 [===========>..................] - ETA: 37s - loss: 7.7281 - accuracy: 0.4960
10752/25000 [===========>..................] - ETA: 37s - loss: 7.7308 - accuracy: 0.4958
10784/25000 [===========>..................] - ETA: 37s - loss: 7.7292 - accuracy: 0.4959
10816/25000 [===========>..................] - ETA: 37s - loss: 7.7290 - accuracy: 0.4959
10848/25000 [============>.................] - ETA: 37s - loss: 7.7288 - accuracy: 0.4959
10880/25000 [============>.................] - ETA: 36s - loss: 7.7357 - accuracy: 0.4955
10912/25000 [============>.................] - ETA: 36s - loss: 7.7341 - accuracy: 0.4956
10944/25000 [============>.................] - ETA: 36s - loss: 7.7367 - accuracy: 0.4954
10976/25000 [============>.................] - ETA: 36s - loss: 7.7421 - accuracy: 0.4951
11008/25000 [============>.................] - ETA: 36s - loss: 7.7474 - accuracy: 0.4947
11040/25000 [============>.................] - ETA: 36s - loss: 7.7486 - accuracy: 0.4947
11072/25000 [============>.................] - ETA: 36s - loss: 7.7497 - accuracy: 0.4946
11104/25000 [============>.................] - ETA: 36s - loss: 7.7453 - accuracy: 0.4949
11136/25000 [============>.................] - ETA: 36s - loss: 7.7451 - accuracy: 0.4949
11168/25000 [============>.................] - ETA: 36s - loss: 7.7449 - accuracy: 0.4949
11200/25000 [============>.................] - ETA: 36s - loss: 7.7447 - accuracy: 0.4949
11232/25000 [============>.................] - ETA: 36s - loss: 7.7472 - accuracy: 0.4947
11264/25000 [============>.................] - ETA: 35s - loss: 7.7497 - accuracy: 0.4946
11296/25000 [============>.................] - ETA: 35s - loss: 7.7494 - accuracy: 0.4946
11328/25000 [============>.................] - ETA: 35s - loss: 7.7451 - accuracy: 0.4949
11360/25000 [============>.................] - ETA: 35s - loss: 7.7490 - accuracy: 0.4946
11392/25000 [============>.................] - ETA: 35s - loss: 7.7528 - accuracy: 0.4944
11424/25000 [============>.................] - ETA: 35s - loss: 7.7525 - accuracy: 0.4944
11456/25000 [============>.................] - ETA: 35s - loss: 7.7496 - accuracy: 0.4946
11488/25000 [============>.................] - ETA: 35s - loss: 7.7427 - accuracy: 0.4950
11520/25000 [============>.................] - ETA: 35s - loss: 7.7398 - accuracy: 0.4952
11552/25000 [============>.................] - ETA: 35s - loss: 7.7370 - accuracy: 0.4954
11584/25000 [============>.................] - ETA: 35s - loss: 7.7315 - accuracy: 0.4958
11616/25000 [============>.................] - ETA: 35s - loss: 7.7300 - accuracy: 0.4959
11648/25000 [============>.................] - ETA: 34s - loss: 7.7298 - accuracy: 0.4959
11680/25000 [=============>................] - ETA: 34s - loss: 7.7336 - accuracy: 0.4956
11712/25000 [=============>................] - ETA: 34s - loss: 7.7347 - accuracy: 0.4956
11744/25000 [=============>................] - ETA: 34s - loss: 7.7332 - accuracy: 0.4957
11776/25000 [=============>................] - ETA: 34s - loss: 7.7304 - accuracy: 0.4958
11808/25000 [=============>................] - ETA: 34s - loss: 7.7328 - accuracy: 0.4957
11840/25000 [=============>................] - ETA: 34s - loss: 7.7353 - accuracy: 0.4955
11872/25000 [=============>................] - ETA: 34s - loss: 7.7312 - accuracy: 0.4958
11904/25000 [=============>................] - ETA: 34s - loss: 7.7284 - accuracy: 0.4960
11936/25000 [=============>................] - ETA: 34s - loss: 7.7283 - accuracy: 0.4960
11968/25000 [=============>................] - ETA: 34s - loss: 7.7268 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 34s - loss: 7.7241 - accuracy: 0.4963
12032/25000 [=============>................] - ETA: 33s - loss: 7.7303 - accuracy: 0.4958
12064/25000 [=============>................] - ETA: 33s - loss: 7.7340 - accuracy: 0.4956
12096/25000 [=============>................] - ETA: 33s - loss: 7.7363 - accuracy: 0.4955
12128/25000 [=============>................] - ETA: 33s - loss: 7.7362 - accuracy: 0.4955
12160/25000 [=============>................] - ETA: 33s - loss: 7.7385 - accuracy: 0.4953
12192/25000 [=============>................] - ETA: 33s - loss: 7.7433 - accuracy: 0.4950
12224/25000 [=============>................] - ETA: 33s - loss: 7.7431 - accuracy: 0.4950
12256/25000 [=============>................] - ETA: 33s - loss: 7.7404 - accuracy: 0.4952
12288/25000 [=============>................] - ETA: 33s - loss: 7.7402 - accuracy: 0.4952
12320/25000 [=============>................] - ETA: 33s - loss: 7.7388 - accuracy: 0.4953
12352/25000 [=============>................] - ETA: 33s - loss: 7.7436 - accuracy: 0.4950
12384/25000 [=============>................] - ETA: 33s - loss: 7.7397 - accuracy: 0.4952
12416/25000 [=============>................] - ETA: 32s - loss: 7.7444 - accuracy: 0.4949
12448/25000 [=============>................] - ETA: 32s - loss: 7.7430 - accuracy: 0.4950
12480/25000 [=============>................] - ETA: 32s - loss: 7.7416 - accuracy: 0.4951
12512/25000 [==============>...............] - ETA: 32s - loss: 7.7414 - accuracy: 0.4951
12544/25000 [==============>...............] - ETA: 32s - loss: 7.7400 - accuracy: 0.4952
12576/25000 [==============>...............] - ETA: 32s - loss: 7.7447 - accuracy: 0.4949
12608/25000 [==============>...............] - ETA: 32s - loss: 7.7432 - accuracy: 0.4950
12640/25000 [==============>...............] - ETA: 32s - loss: 7.7455 - accuracy: 0.4949
12672/25000 [==============>...............] - ETA: 32s - loss: 7.7501 - accuracy: 0.4946
12704/25000 [==============>...............] - ETA: 32s - loss: 7.7475 - accuracy: 0.4947
12736/25000 [==============>...............] - ETA: 32s - loss: 7.7473 - accuracy: 0.4947
12768/25000 [==============>...............] - ETA: 32s - loss: 7.7399 - accuracy: 0.4952
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7385 - accuracy: 0.4953
12832/25000 [==============>...............] - ETA: 31s - loss: 7.7419 - accuracy: 0.4951
12864/25000 [==============>...............] - ETA: 31s - loss: 7.7453 - accuracy: 0.4949
12896/25000 [==============>...............] - ETA: 31s - loss: 7.7498 - accuracy: 0.4946
12928/25000 [==============>...............] - ETA: 31s - loss: 7.7449 - accuracy: 0.4949
12960/25000 [==============>...............] - ETA: 31s - loss: 7.7471 - accuracy: 0.4948
12992/25000 [==============>...............] - ETA: 31s - loss: 7.7410 - accuracy: 0.4952
13024/25000 [==============>...............] - ETA: 31s - loss: 7.7431 - accuracy: 0.4950
13056/25000 [==============>...............] - ETA: 31s - loss: 7.7418 - accuracy: 0.4951
13088/25000 [==============>...............] - ETA: 31s - loss: 7.7416 - accuracy: 0.4951
13120/25000 [==============>...............] - ETA: 31s - loss: 7.7426 - accuracy: 0.4950
13152/25000 [==============>...............] - ETA: 31s - loss: 7.7401 - accuracy: 0.4952
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7376 - accuracy: 0.4954
13216/25000 [==============>...............] - ETA: 30s - loss: 7.7409 - accuracy: 0.4952
13248/25000 [==============>...............] - ETA: 30s - loss: 7.7395 - accuracy: 0.4952
13280/25000 [==============>...............] - ETA: 30s - loss: 7.7371 - accuracy: 0.4954
13312/25000 [==============>...............] - ETA: 30s - loss: 7.7346 - accuracy: 0.4956
13344/25000 [===============>..............] - ETA: 30s - loss: 7.7344 - accuracy: 0.4956
13376/25000 [===============>..............] - ETA: 30s - loss: 7.7377 - accuracy: 0.4954
13408/25000 [===============>..............] - ETA: 30s - loss: 7.7341 - accuracy: 0.4956
13440/25000 [===============>..............] - ETA: 30s - loss: 7.7385 - accuracy: 0.4953
13472/25000 [===============>..............] - ETA: 30s - loss: 7.7383 - accuracy: 0.4953
13504/25000 [===============>..............] - ETA: 30s - loss: 7.7325 - accuracy: 0.4957
13536/25000 [===============>..............] - ETA: 30s - loss: 7.7346 - accuracy: 0.4956
13568/25000 [===============>..............] - ETA: 29s - loss: 7.7367 - accuracy: 0.4954
13600/25000 [===============>..............] - ETA: 29s - loss: 7.7309 - accuracy: 0.4958
13632/25000 [===============>..............] - ETA: 29s - loss: 7.7330 - accuracy: 0.4957
13664/25000 [===============>..............] - ETA: 29s - loss: 7.7317 - accuracy: 0.4958
13696/25000 [===============>..............] - ETA: 29s - loss: 7.7372 - accuracy: 0.4954
13728/25000 [===============>..............] - ETA: 29s - loss: 7.7348 - accuracy: 0.4956
13760/25000 [===============>..............] - ETA: 29s - loss: 7.7379 - accuracy: 0.4953
13792/25000 [===============>..............] - ETA: 29s - loss: 7.7422 - accuracy: 0.4951
13824/25000 [===============>..............] - ETA: 29s - loss: 7.7465 - accuracy: 0.4948
13856/25000 [===============>..............] - ETA: 29s - loss: 7.7419 - accuracy: 0.4951
13888/25000 [===============>..............] - ETA: 29s - loss: 7.7439 - accuracy: 0.4950
13920/25000 [===============>..............] - ETA: 29s - loss: 7.7415 - accuracy: 0.4951
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7435 - accuracy: 0.4950
13984/25000 [===============>..............] - ETA: 28s - loss: 7.7467 - accuracy: 0.4948
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7410 - accuracy: 0.4951
14048/25000 [===============>..............] - ETA: 28s - loss: 7.7452 - accuracy: 0.4949
14080/25000 [===============>..............] - ETA: 28s - loss: 7.7439 - accuracy: 0.4950
14112/25000 [===============>..............] - ETA: 28s - loss: 7.7372 - accuracy: 0.4954
14144/25000 [===============>..............] - ETA: 28s - loss: 7.7338 - accuracy: 0.4956
14176/25000 [================>.............] - ETA: 28s - loss: 7.7304 - accuracy: 0.4958
14208/25000 [================>.............] - ETA: 28s - loss: 7.7314 - accuracy: 0.4958
14240/25000 [================>.............] - ETA: 28s - loss: 7.7355 - accuracy: 0.4955
14272/25000 [================>.............] - ETA: 28s - loss: 7.7311 - accuracy: 0.4958
14304/25000 [================>.............] - ETA: 27s - loss: 7.7352 - accuracy: 0.4955
14336/25000 [================>.............] - ETA: 27s - loss: 7.7297 - accuracy: 0.4959
14368/25000 [================>.............] - ETA: 27s - loss: 7.7285 - accuracy: 0.4960
14400/25000 [================>.............] - ETA: 27s - loss: 7.7284 - accuracy: 0.4960
14432/25000 [================>.............] - ETA: 27s - loss: 7.7261 - accuracy: 0.4961
14464/25000 [================>.............] - ETA: 27s - loss: 7.7281 - accuracy: 0.4960
14496/25000 [================>.............] - ETA: 27s - loss: 7.7322 - accuracy: 0.4957
14528/25000 [================>.............] - ETA: 27s - loss: 7.7373 - accuracy: 0.4954
14560/25000 [================>.............] - ETA: 27s - loss: 7.7330 - accuracy: 0.4957
14592/25000 [================>.............] - ETA: 27s - loss: 7.7349 - accuracy: 0.4955
14624/25000 [================>.............] - ETA: 27s - loss: 7.7337 - accuracy: 0.4956
14656/25000 [================>.............] - ETA: 27s - loss: 7.7367 - accuracy: 0.4954
14688/25000 [================>.............] - ETA: 26s - loss: 7.7366 - accuracy: 0.4954
14720/25000 [================>.............] - ETA: 26s - loss: 7.7375 - accuracy: 0.4954
14752/25000 [================>.............] - ETA: 26s - loss: 7.7373 - accuracy: 0.4954
14784/25000 [================>.............] - ETA: 26s - loss: 7.7361 - accuracy: 0.4955
14816/25000 [================>.............] - ETA: 26s - loss: 7.7349 - accuracy: 0.4955
14848/25000 [================>.............] - ETA: 26s - loss: 7.7348 - accuracy: 0.4956
14880/25000 [================>.............] - ETA: 26s - loss: 7.7377 - accuracy: 0.4954
14912/25000 [================>.............] - ETA: 26s - loss: 7.7417 - accuracy: 0.4951
14944/25000 [================>.............] - ETA: 26s - loss: 7.7405 - accuracy: 0.4952
14976/25000 [================>.............] - ETA: 26s - loss: 7.7424 - accuracy: 0.4951
15008/25000 [=================>............] - ETA: 26s - loss: 7.7473 - accuracy: 0.4947
15040/25000 [=================>............] - ETA: 26s - loss: 7.7482 - accuracy: 0.4947
15072/25000 [=================>............] - ETA: 25s - loss: 7.7500 - accuracy: 0.4946
15104/25000 [=================>............] - ETA: 25s - loss: 7.7488 - accuracy: 0.4946
15136/25000 [=================>............] - ETA: 25s - loss: 7.7426 - accuracy: 0.4950
15168/25000 [=================>............] - ETA: 25s - loss: 7.7414 - accuracy: 0.4951
15200/25000 [=================>............] - ETA: 25s - loss: 7.7433 - accuracy: 0.4950
15232/25000 [=================>............] - ETA: 25s - loss: 7.7411 - accuracy: 0.4951
15264/25000 [=================>............] - ETA: 25s - loss: 7.7430 - accuracy: 0.4950
15296/25000 [=================>............] - ETA: 25s - loss: 7.7478 - accuracy: 0.4947
15328/25000 [=================>............] - ETA: 25s - loss: 7.7476 - accuracy: 0.4947
15360/25000 [=================>............] - ETA: 25s - loss: 7.7495 - accuracy: 0.4946
15392/25000 [=================>............] - ETA: 25s - loss: 7.7443 - accuracy: 0.4949
15424/25000 [=================>............] - ETA: 25s - loss: 7.7452 - accuracy: 0.4949
15456/25000 [=================>............] - ETA: 24s - loss: 7.7440 - accuracy: 0.4950
15488/25000 [=================>............] - ETA: 24s - loss: 7.7458 - accuracy: 0.4948
15520/25000 [=================>............] - ETA: 24s - loss: 7.7417 - accuracy: 0.4951
15552/25000 [=================>............] - ETA: 24s - loss: 7.7445 - accuracy: 0.4949
15584/25000 [=================>............] - ETA: 24s - loss: 7.7414 - accuracy: 0.4951
15616/25000 [=================>............] - ETA: 24s - loss: 7.7383 - accuracy: 0.4953
15648/25000 [=================>............] - ETA: 24s - loss: 7.7382 - accuracy: 0.4953
15680/25000 [=================>............] - ETA: 24s - loss: 7.7390 - accuracy: 0.4953
15712/25000 [=================>............] - ETA: 24s - loss: 7.7437 - accuracy: 0.4950
15744/25000 [=================>............] - ETA: 24s - loss: 7.7416 - accuracy: 0.4951
15776/25000 [=================>............] - ETA: 24s - loss: 7.7385 - accuracy: 0.4953
15808/25000 [=================>............] - ETA: 24s - loss: 7.7423 - accuracy: 0.4951
15840/25000 [==================>...........] - ETA: 23s - loss: 7.7460 - accuracy: 0.4948
15872/25000 [==================>...........] - ETA: 23s - loss: 7.7478 - accuracy: 0.4947
15904/25000 [==================>...........] - ETA: 23s - loss: 7.7437 - accuracy: 0.4950
15936/25000 [==================>...........] - ETA: 23s - loss: 7.7455 - accuracy: 0.4949
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7444 - accuracy: 0.4949
16000/25000 [==================>...........] - ETA: 23s - loss: 7.7462 - accuracy: 0.4948
16032/25000 [==================>...........] - ETA: 23s - loss: 7.7460 - accuracy: 0.4948
16064/25000 [==================>...........] - ETA: 23s - loss: 7.7458 - accuracy: 0.4948
16096/25000 [==================>...........] - ETA: 23s - loss: 7.7457 - accuracy: 0.4948
16128/25000 [==================>...........] - ETA: 23s - loss: 7.7455 - accuracy: 0.4949
16160/25000 [==================>...........] - ETA: 23s - loss: 7.7425 - accuracy: 0.4950
16192/25000 [==================>...........] - ETA: 23s - loss: 7.7424 - accuracy: 0.4951
16224/25000 [==================>...........] - ETA: 22s - loss: 7.7403 - accuracy: 0.4952
16256/25000 [==================>...........] - ETA: 22s - loss: 7.7374 - accuracy: 0.4954
16288/25000 [==================>...........] - ETA: 22s - loss: 7.7372 - accuracy: 0.4954
16320/25000 [==================>...........] - ETA: 22s - loss: 7.7371 - accuracy: 0.4954
16352/25000 [==================>...........] - ETA: 22s - loss: 7.7341 - accuracy: 0.4956
16384/25000 [==================>...........] - ETA: 22s - loss: 7.7331 - accuracy: 0.4957
16416/25000 [==================>...........] - ETA: 22s - loss: 7.7292 - accuracy: 0.4959
16448/25000 [==================>...........] - ETA: 22s - loss: 7.7281 - accuracy: 0.4960
16480/25000 [==================>...........] - ETA: 22s - loss: 7.7290 - accuracy: 0.4959
16512/25000 [==================>...........] - ETA: 22s - loss: 7.7270 - accuracy: 0.4961
16544/25000 [==================>...........] - ETA: 22s - loss: 7.7269 - accuracy: 0.4961
16576/25000 [==================>...........] - ETA: 21s - loss: 7.7240 - accuracy: 0.4963
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7202 - accuracy: 0.4965
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7201 - accuracy: 0.4965
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7200 - accuracy: 0.4965
16704/25000 [===================>..........] - ETA: 21s - loss: 7.7217 - accuracy: 0.4964
16736/25000 [===================>..........] - ETA: 21s - loss: 7.7170 - accuracy: 0.4967
16768/25000 [===================>..........] - ETA: 21s - loss: 7.7187 - accuracy: 0.4966
16800/25000 [===================>..........] - ETA: 21s - loss: 7.7177 - accuracy: 0.4967
16832/25000 [===================>..........] - ETA: 21s - loss: 7.7167 - accuracy: 0.4967
16864/25000 [===================>..........] - ETA: 21s - loss: 7.7194 - accuracy: 0.4966
16896/25000 [===================>..........] - ETA: 21s - loss: 7.7165 - accuracy: 0.4967
16928/25000 [===================>..........] - ETA: 21s - loss: 7.7173 - accuracy: 0.4967
16960/25000 [===================>..........] - ETA: 20s - loss: 7.7172 - accuracy: 0.4967
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7199 - accuracy: 0.4965
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7126 - accuracy: 0.4970
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7161 - accuracy: 0.4968
17088/25000 [===================>..........] - ETA: 20s - loss: 7.7142 - accuracy: 0.4969
17120/25000 [===================>..........] - ETA: 20s - loss: 7.7114 - accuracy: 0.4971
17152/25000 [===================>..........] - ETA: 20s - loss: 7.7158 - accuracy: 0.4968
17184/25000 [===================>..........] - ETA: 20s - loss: 7.7166 - accuracy: 0.4967
17216/25000 [===================>..........] - ETA: 20s - loss: 7.7147 - accuracy: 0.4969
17248/25000 [===================>..........] - ETA: 20s - loss: 7.7173 - accuracy: 0.4967
17280/25000 [===================>..........] - ETA: 20s - loss: 7.7181 - accuracy: 0.4966
17312/25000 [===================>..........] - ETA: 20s - loss: 7.7189 - accuracy: 0.4966
17344/25000 [===================>..........] - ETA: 19s - loss: 7.7188 - accuracy: 0.4966
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7231 - accuracy: 0.4963
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7248 - accuracy: 0.4962
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7238 - accuracy: 0.4963
17472/25000 [===================>..........] - ETA: 19s - loss: 7.7175 - accuracy: 0.4967
17504/25000 [====================>.........] - ETA: 19s - loss: 7.7174 - accuracy: 0.4967
17536/25000 [====================>.........] - ETA: 19s - loss: 7.7165 - accuracy: 0.4967
17568/25000 [====================>.........] - ETA: 19s - loss: 7.7146 - accuracy: 0.4969
17600/25000 [====================>.........] - ETA: 19s - loss: 7.7154 - accuracy: 0.4968
17632/25000 [====================>.........] - ETA: 19s - loss: 7.7127 - accuracy: 0.4970
17664/25000 [====================>.........] - ETA: 19s - loss: 7.7161 - accuracy: 0.4968
17696/25000 [====================>.........] - ETA: 19s - loss: 7.7117 - accuracy: 0.4971
17728/25000 [====================>.........] - ETA: 18s - loss: 7.7081 - accuracy: 0.4973
17760/25000 [====================>.........] - ETA: 18s - loss: 7.7072 - accuracy: 0.4974
17792/25000 [====================>.........] - ETA: 18s - loss: 7.7071 - accuracy: 0.4974
17824/25000 [====================>.........] - ETA: 18s - loss: 7.7010 - accuracy: 0.4978
17856/25000 [====================>.........] - ETA: 18s - loss: 7.7061 - accuracy: 0.4974
17888/25000 [====================>.........] - ETA: 18s - loss: 7.7060 - accuracy: 0.4974
17920/25000 [====================>.........] - ETA: 18s - loss: 7.7094 - accuracy: 0.4972
17952/25000 [====================>.........] - ETA: 18s - loss: 7.7076 - accuracy: 0.4973
17984/25000 [====================>.........] - ETA: 18s - loss: 7.7050 - accuracy: 0.4975
18016/25000 [====================>.........] - ETA: 18s - loss: 7.7032 - accuracy: 0.4976
18048/25000 [====================>.........] - ETA: 18s - loss: 7.7032 - accuracy: 0.4976
18080/25000 [====================>.........] - ETA: 18s - loss: 7.7022 - accuracy: 0.4977
18112/25000 [====================>.........] - ETA: 17s - loss: 7.7013 - accuracy: 0.4977
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6970 - accuracy: 0.4980
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6987 - accuracy: 0.4979
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6995 - accuracy: 0.4979
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6969 - accuracy: 0.4980
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6952 - accuracy: 0.4981
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6959 - accuracy: 0.4981
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6984 - accuracy: 0.4979
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6983 - accuracy: 0.4979
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6966 - accuracy: 0.4980
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6966 - accuracy: 0.4980
18464/25000 [=====================>........] - ETA: 17s - loss: 7.6924 - accuracy: 0.4983
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6940 - accuracy: 0.4982
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6906 - accuracy: 0.4984
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6864 - accuracy: 0.4987
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6881 - accuracy: 0.4986
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6880 - accuracy: 0.4986
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6929 - accuracy: 0.4983
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6970 - accuracy: 0.4980
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6977 - accuracy: 0.4980
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6985 - accuracy: 0.4979
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6960 - accuracy: 0.4981
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6960 - accuracy: 0.4981
18848/25000 [=====================>........] - ETA: 16s - loss: 7.6975 - accuracy: 0.4980
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6983 - accuracy: 0.4979
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6974 - accuracy: 0.4980
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6917 - accuracy: 0.4984
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6909 - accuracy: 0.4984
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6908 - accuracy: 0.4984
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6924 - accuracy: 0.4983
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6915 - accuracy: 0.4984
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6907 - accuracy: 0.4984
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6907 - accuracy: 0.4984
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6906 - accuracy: 0.4984
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6906 - accuracy: 0.4984
19232/25000 [======================>.......] - ETA: 15s - loss: 7.6913 - accuracy: 0.4984
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6849 - accuracy: 0.4988
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6841 - accuracy: 0.4989
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6841 - accuracy: 0.4989
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6864 - accuracy: 0.4987
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6895 - accuracy: 0.4985
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6903 - accuracy: 0.4985
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6903 - accuracy: 0.4985
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6894 - accuracy: 0.4985
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6910 - accuracy: 0.4984
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6894 - accuracy: 0.4985
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6878 - accuracy: 0.4986
19616/25000 [======================>.......] - ETA: 14s - loss: 7.6916 - accuracy: 0.4984
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6916 - accuracy: 0.4984
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6861 - accuracy: 0.4987
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6900 - accuracy: 0.4985
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6899 - accuracy: 0.4985
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6883 - accuracy: 0.4986
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6891 - accuracy: 0.4985
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6906 - accuracy: 0.4984
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6921 - accuracy: 0.4983
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6920 - accuracy: 0.4983
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6897 - accuracy: 0.4985
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6874 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 13s - loss: 7.6858 - accuracy: 0.4988
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6873 - accuracy: 0.4987
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6842 - accuracy: 0.4989
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6865 - accuracy: 0.4987
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6849 - accuracy: 0.4988
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6879 - accuracy: 0.4986
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6864 - accuracy: 0.4987
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6878 - accuracy: 0.4986
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6818 - accuracy: 0.4990
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6825 - accuracy: 0.4990
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6825 - accuracy: 0.4990
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6832 - accuracy: 0.4989
20384/25000 [=======================>......] - ETA: 12s - loss: 7.6869 - accuracy: 0.4987
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6861 - accuracy: 0.4987
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6854 - accuracy: 0.4988
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6831 - accuracy: 0.4989
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6831 - accuracy: 0.4989
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6815 - accuracy: 0.4990
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6830 - accuracy: 0.4989
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6822 - accuracy: 0.4990
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6815 - accuracy: 0.4990
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6829 - accuracy: 0.4989
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6837 - accuracy: 0.4989
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6821 - accuracy: 0.4990
20768/25000 [=======================>......] - ETA: 11s - loss: 7.6829 - accuracy: 0.4989
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6850 - accuracy: 0.4988
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6850 - accuracy: 0.4988
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6821 - accuracy: 0.4990
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6835 - accuracy: 0.4989
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6805 - accuracy: 0.4991
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6798 - accuracy: 0.4991
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6790 - accuracy: 0.4992
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6761 - accuracy: 0.4994
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6790 - accuracy: 0.4992
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6811 - accuracy: 0.4991
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6833 - accuracy: 0.4989
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6811 - accuracy: 0.4991 
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6782 - accuracy: 0.4992
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6774 - accuracy: 0.4993
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6753 - accuracy: 0.4994
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6731 - accuracy: 0.4996
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6724 - accuracy: 0.4996
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6709 - accuracy: 0.4997
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21536/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6631 - accuracy: 0.5002
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6624 - accuracy: 0.5003
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6631 - accuracy: 0.5002
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6631 - accuracy: 0.5002
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6617 - accuracy: 0.5003
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6638 - accuracy: 0.5002
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6631 - accuracy: 0.5002
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6659 - accuracy: 0.5000
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6632 - accuracy: 0.5002
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6611 - accuracy: 0.5004
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6597 - accuracy: 0.5004
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6591 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6577 - accuracy: 0.5006
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6571 - accuracy: 0.5006
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6537 - accuracy: 0.5008
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6517 - accuracy: 0.5010
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6510 - accuracy: 0.5010
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6511 - accuracy: 0.5010
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6531 - accuracy: 0.5009
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6497 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6532 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6539 - accuracy: 0.5008
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6546 - accuracy: 0.5008
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6546 - accuracy: 0.5008
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6526 - accuracy: 0.5009
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6506 - accuracy: 0.5010
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6526 - accuracy: 0.5009
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6553 - accuracy: 0.5007
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6560 - accuracy: 0.5007
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6587 - accuracy: 0.5005
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6594 - accuracy: 0.5005
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6627 - accuracy: 0.5003
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6620 - accuracy: 0.5003
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6581 - accuracy: 0.5006
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6594 - accuracy: 0.5005
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6561 - accuracy: 0.5007
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6542 - accuracy: 0.5008
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6562 - accuracy: 0.5007
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6562 - accuracy: 0.5007
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6562 - accuracy: 0.5007
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6575 - accuracy: 0.5006
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5003
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24192/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24224/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 1s - loss: 7.6673 - accuracy: 0.5000
24288/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24320/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24448/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 77s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

