
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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
 40%|████      | 2/5 [00:56<01:24, 28.26s/it] 40%|████      | 2/5 [00:56<01:24, 28.26s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.38594848989308633, 'embedding_size_factor': 1.1152444305928642, 'layers.choice': 1, 'learning_rate': 0.00038756034620889994, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.04240401971805843} and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd8\xb3aK\x82\x02UX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xd8\n\x8bG\x14tX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?9f/\x03\x1f\x1f\xb7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa5\xb5\xfa\xcb\xd0\xe5du.' and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd8\xb3aK\x82\x02UX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xd8\n\x8bG\x14tX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?9f/\x03\x1f\x1f\xb7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa5\xb5\xfa\xcb\xd0\xe5du.' and reward: 0.3732
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.41052103042603
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -42.42s of remaining time.
Ensemble size: 4
Ensemble weights: 
[0.75 0.25]
	0.391	 = Validation accuracy score
	0.96s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 163.42s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f515d2d99e8> 

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
 [-0.01100719  0.17323628  0.00140251 -0.05467196  0.02690583  0.05321748]
 [ 0.0419086   0.06460232 -0.05082988  0.2753146  -0.12415525  0.06364498]
 [-0.23204733  0.20879871 -0.09094369  0.1387883  -0.08874978  0.2434074 ]
 [ 0.0565645  -0.00846442 -0.03938613  0.16488247  0.1875692   0.07985894]
 [-0.08695517  0.28003904 -0.33075625 -0.0616355   0.30940393 -0.0412188 ]
 [ 0.34520501  0.00402107 -0.05386268  0.38850552 -0.14218935  0.27300617]
 [-0.27549461  1.1504246  -0.18623957  0.26593179  0.17950603  0.41059971]
 [ 0.26333421  0.51843846  0.06348023 -0.01708677 -0.16463831  0.05341707]
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
{'loss': 0.5714426785707474, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 05:17:43.799785: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.3744090907275677, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 05:17:45.074166: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
   90112/17464789 [..............................] - ETA: 38s
  196608/17464789 [..............................] - ETA: 23s
  385024/17464789 [..............................] - ETA: 14s
  753664/17464789 [>.............................] - ETA: 8s 
 1515520/17464789 [=>............................] - ETA: 4s
 3031040/17464789 [====>.........................] - ETA: 2s
 6045696/17464789 [=========>....................] - ETA: 1s
 9043968/17464789 [==============>...............] - ETA: 0s
12091392/17464789 [===================>..........] - ETA: 0s
15171584/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 05:17:58.377629: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 05:17:58.382012: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-15 05:17:58.382187: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557481c06f20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 05:17:58.382205: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:47 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 3:03 - loss: 7.9062 - accuracy: 0.4844 
   96/25000 [..............................] - ETA: 2:28 - loss: 8.4652 - accuracy: 0.4479
  128/25000 [..............................] - ETA: 2:13 - loss: 8.6249 - accuracy: 0.4375
  160/25000 [..............................] - ETA: 2:02 - loss: 8.6249 - accuracy: 0.4375
  192/25000 [..............................] - ETA: 1:56 - loss: 8.2256 - accuracy: 0.4635
  224/25000 [..............................] - ETA: 1:50 - loss: 8.3511 - accuracy: 0.4554
  256/25000 [..............................] - ETA: 1:47 - loss: 8.2057 - accuracy: 0.4648
  288/25000 [..............................] - ETA: 1:44 - loss: 8.0925 - accuracy: 0.4722
  320/25000 [..............................] - ETA: 1:41 - loss: 7.8104 - accuracy: 0.4906
  352/25000 [..............................] - ETA: 1:39 - loss: 7.7973 - accuracy: 0.4915
  384/25000 [..............................] - ETA: 1:37 - loss: 7.7864 - accuracy: 0.4922
  416/25000 [..............................] - ETA: 1:36 - loss: 7.9246 - accuracy: 0.4832
  448/25000 [..............................] - ETA: 1:35 - loss: 7.7693 - accuracy: 0.4933
  480/25000 [..............................] - ETA: 1:35 - loss: 7.6666 - accuracy: 0.5000
  512/25000 [..............................] - ETA: 1:34 - loss: 7.7265 - accuracy: 0.4961
  544/25000 [..............................] - ETA: 1:33 - loss: 7.8921 - accuracy: 0.4853
  576/25000 [..............................] - ETA: 1:32 - loss: 7.7465 - accuracy: 0.4948
  608/25000 [..............................] - ETA: 1:31 - loss: 7.7423 - accuracy: 0.4951
  640/25000 [..............................] - ETA: 1:30 - loss: 7.7864 - accuracy: 0.4922
  672/25000 [..............................] - ETA: 1:29 - loss: 7.9176 - accuracy: 0.4836
  704/25000 [..............................] - ETA: 1:28 - loss: 8.0151 - accuracy: 0.4773
  736/25000 [..............................] - ETA: 1:28 - loss: 8.0000 - accuracy: 0.4783
  768/25000 [..............................] - ETA: 1:27 - loss: 7.8862 - accuracy: 0.4857
  800/25000 [..............................] - ETA: 1:27 - loss: 7.8775 - accuracy: 0.4863
  832/25000 [..............................] - ETA: 1:26 - loss: 7.9062 - accuracy: 0.4844
  864/25000 [>.............................] - ETA: 1:26 - loss: 7.9328 - accuracy: 0.4826
  896/25000 [>.............................] - ETA: 1:25 - loss: 7.9404 - accuracy: 0.4821
  928/25000 [>.............................] - ETA: 1:25 - loss: 7.8979 - accuracy: 0.4849
  960/25000 [>.............................] - ETA: 1:25 - loss: 7.9062 - accuracy: 0.4844
  992/25000 [>.............................] - ETA: 1:24 - loss: 7.9758 - accuracy: 0.4798
 1024/25000 [>.............................] - ETA: 1:24 - loss: 7.9361 - accuracy: 0.4824
 1056/25000 [>.............................] - ETA: 1:24 - loss: 7.9280 - accuracy: 0.4830
 1088/25000 [>.............................] - ETA: 1:23 - loss: 7.9626 - accuracy: 0.4807
 1120/25000 [>.............................] - ETA: 1:23 - loss: 7.9404 - accuracy: 0.4821
 1152/25000 [>.............................] - ETA: 1:23 - loss: 7.9195 - accuracy: 0.4835
 1184/25000 [>.............................] - ETA: 1:23 - loss: 7.8738 - accuracy: 0.4865
 1216/25000 [>.............................] - ETA: 1:22 - loss: 7.9188 - accuracy: 0.4836
 1248/25000 [>.............................] - ETA: 1:22 - loss: 7.8509 - accuracy: 0.4880
 1280/25000 [>.............................] - ETA: 1:22 - loss: 7.8343 - accuracy: 0.4891
 1312/25000 [>.............................] - ETA: 1:22 - loss: 7.8069 - accuracy: 0.4909
 1344/25000 [>.............................] - ETA: 1:21 - loss: 7.8035 - accuracy: 0.4911
 1376/25000 [>.............................] - ETA: 1:21 - loss: 7.8226 - accuracy: 0.4898
 1408/25000 [>.............................] - ETA: 1:21 - loss: 7.7973 - accuracy: 0.4915
 1440/25000 [>.............................] - ETA: 1:21 - loss: 7.7944 - accuracy: 0.4917
 1472/25000 [>.............................] - ETA: 1:21 - loss: 7.7708 - accuracy: 0.4932
 1504/25000 [>.............................] - ETA: 1:21 - loss: 7.7788 - accuracy: 0.4927
 1536/25000 [>.............................] - ETA: 1:21 - loss: 7.7864 - accuracy: 0.4922
 1568/25000 [>.............................] - ETA: 1:21 - loss: 7.7937 - accuracy: 0.4917
 1600/25000 [>.............................] - ETA: 1:21 - loss: 7.8391 - accuracy: 0.4888
 1632/25000 [>.............................] - ETA: 1:21 - loss: 7.8169 - accuracy: 0.4902
 1664/25000 [>.............................] - ETA: 1:20 - loss: 7.7956 - accuracy: 0.4916
 1696/25000 [=>............................] - ETA: 1:20 - loss: 7.7932 - accuracy: 0.4917
 1728/25000 [=>............................] - ETA: 1:20 - loss: 7.8175 - accuracy: 0.4902
 1760/25000 [=>............................] - ETA: 1:20 - loss: 7.8147 - accuracy: 0.4903
 1792/25000 [=>............................] - ETA: 1:19 - loss: 7.7950 - accuracy: 0.4916
 1824/25000 [=>............................] - ETA: 1:19 - loss: 7.7759 - accuracy: 0.4929
 1856/25000 [=>............................] - ETA: 1:19 - loss: 7.7823 - accuracy: 0.4925
 1888/25000 [=>............................] - ETA: 1:19 - loss: 7.7966 - accuracy: 0.4915
 1920/25000 [=>............................] - ETA: 1:19 - loss: 7.7944 - accuracy: 0.4917
 1952/25000 [=>............................] - ETA: 1:19 - loss: 7.8316 - accuracy: 0.4892
 1984/25000 [=>............................] - ETA: 1:18 - loss: 7.8212 - accuracy: 0.4899
 2016/25000 [=>............................] - ETA: 1:18 - loss: 7.8035 - accuracy: 0.4911
 2048/25000 [=>............................] - ETA: 1:18 - loss: 7.8014 - accuracy: 0.4912
 2080/25000 [=>............................] - ETA: 1:18 - loss: 7.7919 - accuracy: 0.4918
 2112/25000 [=>............................] - ETA: 1:18 - loss: 7.7973 - accuracy: 0.4915
 2144/25000 [=>............................] - ETA: 1:18 - loss: 7.7596 - accuracy: 0.4939
 2176/25000 [=>............................] - ETA: 1:17 - loss: 7.7441 - accuracy: 0.4949
 2208/25000 [=>............................] - ETA: 1:17 - loss: 7.7430 - accuracy: 0.4950
 2240/25000 [=>............................] - ETA: 1:17 - loss: 7.7488 - accuracy: 0.4946
 2272/25000 [=>............................] - ETA: 1:17 - loss: 7.7611 - accuracy: 0.4938
 2304/25000 [=>............................] - ETA: 1:17 - loss: 7.7398 - accuracy: 0.4952
 2336/25000 [=>............................] - ETA: 1:17 - loss: 7.7388 - accuracy: 0.4953
 2368/25000 [=>............................] - ETA: 1:16 - loss: 7.7508 - accuracy: 0.4945
 2400/25000 [=>............................] - ETA: 1:16 - loss: 7.7688 - accuracy: 0.4933
 2432/25000 [=>............................] - ETA: 1:16 - loss: 7.8053 - accuracy: 0.4910
 2464/25000 [=>............................] - ETA: 1:16 - loss: 7.8346 - accuracy: 0.4890
 2496/25000 [=>............................] - ETA: 1:16 - loss: 7.8141 - accuracy: 0.4904
 2528/25000 [==>...........................] - ETA: 1:15 - loss: 7.8243 - accuracy: 0.4897
 2560/25000 [==>...........................] - ETA: 1:15 - loss: 7.8283 - accuracy: 0.4895
 2592/25000 [==>...........................] - ETA: 1:15 - loss: 7.8204 - accuracy: 0.4900
 2624/25000 [==>...........................] - ETA: 1:15 - loss: 7.8478 - accuracy: 0.4882
 2656/25000 [==>...........................] - ETA: 1:15 - loss: 7.8629 - accuracy: 0.4872
 2688/25000 [==>...........................] - ETA: 1:14 - loss: 7.8549 - accuracy: 0.4877
 2720/25000 [==>...........................] - ETA: 1:14 - loss: 7.8696 - accuracy: 0.4868
 2752/25000 [==>...........................] - ETA: 1:14 - loss: 7.8728 - accuracy: 0.4866
 2784/25000 [==>...........................] - ETA: 1:14 - loss: 7.8704 - accuracy: 0.4867
 2816/25000 [==>...........................] - ETA: 1:14 - loss: 7.8790 - accuracy: 0.4862
 2848/25000 [==>...........................] - ETA: 1:14 - loss: 7.8497 - accuracy: 0.4881
 2880/25000 [==>...........................] - ETA: 1:14 - loss: 7.8476 - accuracy: 0.4882
 2912/25000 [==>...........................] - ETA: 1:14 - loss: 7.8667 - accuracy: 0.4870
 2944/25000 [==>...........................] - ETA: 1:13 - loss: 7.8750 - accuracy: 0.4864
 2976/25000 [==>...........................] - ETA: 1:13 - loss: 7.8779 - accuracy: 0.4862
 3008/25000 [==>...........................] - ETA: 1:13 - loss: 7.8705 - accuracy: 0.4867
 3040/25000 [==>...........................] - ETA: 1:13 - loss: 7.8583 - accuracy: 0.4875
 3072/25000 [==>...........................] - ETA: 1:13 - loss: 7.8513 - accuracy: 0.4880
 3104/25000 [==>...........................] - ETA: 1:13 - loss: 7.8494 - accuracy: 0.4881
 3136/25000 [==>...........................] - ETA: 1:13 - loss: 7.8377 - accuracy: 0.4888
 3168/25000 [==>...........................] - ETA: 1:12 - loss: 7.8409 - accuracy: 0.4886
 3200/25000 [==>...........................] - ETA: 1:12 - loss: 7.8439 - accuracy: 0.4884
 3232/25000 [==>...........................] - ETA: 1:12 - loss: 7.8611 - accuracy: 0.4873
 3264/25000 [==>...........................] - ETA: 1:12 - loss: 7.8639 - accuracy: 0.4871
 3296/25000 [==>...........................] - ETA: 1:12 - loss: 7.8667 - accuracy: 0.4870
 3328/25000 [==>...........................] - ETA: 1:12 - loss: 7.8693 - accuracy: 0.4868
 3360/25000 [===>..........................] - ETA: 1:12 - loss: 7.8720 - accuracy: 0.4866
 3392/25000 [===>..........................] - ETA: 1:12 - loss: 7.9017 - accuracy: 0.4847
 3424/25000 [===>..........................] - ETA: 1:11 - loss: 7.9084 - accuracy: 0.4842
 3456/25000 [===>..........................] - ETA: 1:11 - loss: 7.8840 - accuracy: 0.4858
 3488/25000 [===>..........................] - ETA: 1:11 - loss: 7.8776 - accuracy: 0.4862
 3520/25000 [===>..........................] - ETA: 1:11 - loss: 7.8626 - accuracy: 0.4872
 3552/25000 [===>..........................] - ETA: 1:11 - loss: 7.8522 - accuracy: 0.4879
 3584/25000 [===>..........................] - ETA: 1:11 - loss: 7.8292 - accuracy: 0.4894
 3616/25000 [===>..........................] - ETA: 1:11 - loss: 7.8193 - accuracy: 0.4900
 3648/25000 [===>..........................] - ETA: 1:11 - loss: 7.8137 - accuracy: 0.4904
 3680/25000 [===>..........................] - ETA: 1:11 - loss: 7.8333 - accuracy: 0.4891
 3712/25000 [===>..........................] - ETA: 1:10 - loss: 7.8442 - accuracy: 0.4884
 3744/25000 [===>..........................] - ETA: 1:10 - loss: 7.8509 - accuracy: 0.4880
 3776/25000 [===>..........................] - ETA: 1:10 - loss: 7.8372 - accuracy: 0.4889
 3808/25000 [===>..........................] - ETA: 1:10 - loss: 7.8277 - accuracy: 0.4895
 3840/25000 [===>..........................] - ETA: 1:10 - loss: 7.8223 - accuracy: 0.4898
 3872/25000 [===>..........................] - ETA: 1:10 - loss: 7.8092 - accuracy: 0.4907
 3904/25000 [===>..........................] - ETA: 1:10 - loss: 7.8119 - accuracy: 0.4905
 3936/25000 [===>..........................] - ETA: 1:10 - loss: 7.8108 - accuracy: 0.4906
 3968/25000 [===>..........................] - ETA: 1:09 - loss: 7.8057 - accuracy: 0.4909
 4000/25000 [===>..........................] - ETA: 1:09 - loss: 7.8123 - accuracy: 0.4905
 4032/25000 [===>..........................] - ETA: 1:09 - loss: 7.8187 - accuracy: 0.4901
 4064/25000 [===>..........................] - ETA: 1:09 - loss: 7.8439 - accuracy: 0.4884
 4096/25000 [===>..........................] - ETA: 1:09 - loss: 7.8313 - accuracy: 0.4893
 4128/25000 [===>..........................] - ETA: 1:09 - loss: 7.8078 - accuracy: 0.4908
 4160/25000 [===>..........................] - ETA: 1:09 - loss: 7.7993 - accuracy: 0.4913
 4192/25000 [====>.........................] - ETA: 1:09 - loss: 7.7946 - accuracy: 0.4917
 4224/25000 [====>.........................] - ETA: 1:09 - loss: 7.8046 - accuracy: 0.4910
 4256/25000 [====>.........................] - ETA: 1:08 - loss: 7.7999 - accuracy: 0.4913
 4288/25000 [====>.........................] - ETA: 1:08 - loss: 7.7989 - accuracy: 0.4914
 4320/25000 [====>.........................] - ETA: 1:08 - loss: 7.8015 - accuracy: 0.4912
 4352/25000 [====>.........................] - ETA: 1:08 - loss: 7.8005 - accuracy: 0.4913
 4384/25000 [====>.........................] - ETA: 1:08 - loss: 7.7925 - accuracy: 0.4918
 4416/25000 [====>.........................] - ETA: 1:08 - loss: 7.7847 - accuracy: 0.4923
 4448/25000 [====>.........................] - ETA: 1:08 - loss: 7.7907 - accuracy: 0.4919
 4480/25000 [====>.........................] - ETA: 1:08 - loss: 7.7967 - accuracy: 0.4915
 4512/25000 [====>.........................] - ETA: 1:08 - loss: 7.7890 - accuracy: 0.4920
 4544/25000 [====>.........................] - ETA: 1:07 - loss: 7.7746 - accuracy: 0.4930
 4576/25000 [====>.........................] - ETA: 1:07 - loss: 7.7671 - accuracy: 0.4934
 4608/25000 [====>.........................] - ETA: 1:07 - loss: 7.7664 - accuracy: 0.4935
 4640/25000 [====>.........................] - ETA: 1:07 - loss: 7.7658 - accuracy: 0.4935
 4672/25000 [====>.........................] - ETA: 1:07 - loss: 7.7749 - accuracy: 0.4929
 4704/25000 [====>.........................] - ETA: 1:07 - loss: 7.7742 - accuracy: 0.4930
 4736/25000 [====>.........................] - ETA: 1:07 - loss: 7.7799 - accuracy: 0.4926
 4768/25000 [====>.........................] - ETA: 1:07 - loss: 7.7888 - accuracy: 0.4920
 4800/25000 [====>.........................] - ETA: 1:06 - loss: 7.7752 - accuracy: 0.4929
 4832/25000 [====>.........................] - ETA: 1:06 - loss: 7.7872 - accuracy: 0.4921
 4864/25000 [====>.........................] - ETA: 1:06 - loss: 7.7864 - accuracy: 0.4922
 4896/25000 [====>.........................] - ETA: 1:06 - loss: 7.7731 - accuracy: 0.4931
 4928/25000 [====>.........................] - ETA: 1:06 - loss: 7.7693 - accuracy: 0.4933
 4960/25000 [====>.........................] - ETA: 1:06 - loss: 7.7686 - accuracy: 0.4933
 4992/25000 [====>.........................] - ETA: 1:06 - loss: 7.7588 - accuracy: 0.4940
 5024/25000 [=====>........................] - ETA: 1:06 - loss: 7.7582 - accuracy: 0.4940
 5056/25000 [=====>........................] - ETA: 1:06 - loss: 7.7515 - accuracy: 0.4945
 5088/25000 [=====>........................] - ETA: 1:05 - loss: 7.7570 - accuracy: 0.4941
 5120/25000 [=====>........................] - ETA: 1:05 - loss: 7.7595 - accuracy: 0.4939
 5152/25000 [=====>........................] - ETA: 1:05 - loss: 7.7648 - accuracy: 0.4936
 5184/25000 [=====>........................] - ETA: 1:05 - loss: 7.7642 - accuracy: 0.4936
 5216/25000 [=====>........................] - ETA: 1:05 - loss: 7.7636 - accuracy: 0.4937
 5248/25000 [=====>........................] - ETA: 1:05 - loss: 7.7630 - accuracy: 0.4937
 5280/25000 [=====>........................] - ETA: 1:05 - loss: 7.7625 - accuracy: 0.4938
 5312/25000 [=====>........................] - ETA: 1:05 - loss: 7.7734 - accuracy: 0.4930
 5344/25000 [=====>........................] - ETA: 1:05 - loss: 7.7757 - accuracy: 0.4929
 5376/25000 [=====>........................] - ETA: 1:04 - loss: 7.7664 - accuracy: 0.4935
 5408/25000 [=====>........................] - ETA: 1:04 - loss: 7.7800 - accuracy: 0.4926
 5440/25000 [=====>........................] - ETA: 1:04 - loss: 7.7653 - accuracy: 0.4936
 5472/25000 [=====>........................] - ETA: 1:04 - loss: 7.7563 - accuracy: 0.4942
 5504/25000 [=====>........................] - ETA: 1:04 - loss: 7.7697 - accuracy: 0.4933
 5536/25000 [=====>........................] - ETA: 1:04 - loss: 7.7719 - accuracy: 0.4931
 5568/25000 [=====>........................] - ETA: 1:04 - loss: 7.7630 - accuracy: 0.4937
 5600/25000 [=====>........................] - ETA: 1:04 - loss: 7.7625 - accuracy: 0.4938
 5632/25000 [=====>........................] - ETA: 1:04 - loss: 7.7592 - accuracy: 0.4940
 5664/25000 [=====>........................] - ETA: 1:03 - loss: 7.7587 - accuracy: 0.4940
 5696/25000 [=====>........................] - ETA: 1:03 - loss: 7.7662 - accuracy: 0.4935
 5728/25000 [=====>........................] - ETA: 1:03 - loss: 7.7550 - accuracy: 0.4942
 5760/25000 [=====>........................] - ETA: 1:03 - loss: 7.7518 - accuracy: 0.4944
 5792/25000 [=====>........................] - ETA: 1:03 - loss: 7.7619 - accuracy: 0.4938
 5824/25000 [=====>........................] - ETA: 1:03 - loss: 7.7693 - accuracy: 0.4933
 5856/25000 [======>.......................] - ETA: 1:03 - loss: 7.7635 - accuracy: 0.4937
 5888/25000 [======>.......................] - ETA: 1:03 - loss: 7.7708 - accuracy: 0.4932
 5920/25000 [======>.......................] - ETA: 1:02 - loss: 7.7599 - accuracy: 0.4939
 5952/25000 [======>.......................] - ETA: 1:02 - loss: 7.7516 - accuracy: 0.4945
 5984/25000 [======>.......................] - ETA: 1:02 - loss: 7.7461 - accuracy: 0.4948
 6016/25000 [======>.......................] - ETA: 1:02 - loss: 7.7507 - accuracy: 0.4945
 6048/25000 [======>.......................] - ETA: 1:02 - loss: 7.7477 - accuracy: 0.4947
 6080/25000 [======>.......................] - ETA: 1:02 - loss: 7.7524 - accuracy: 0.4944
 6112/25000 [======>.......................] - ETA: 1:02 - loss: 7.7569 - accuracy: 0.4941
 6144/25000 [======>.......................] - ETA: 1:02 - loss: 7.7590 - accuracy: 0.4940
 6176/25000 [======>.......................] - ETA: 1:02 - loss: 7.7684 - accuracy: 0.4934
 6208/25000 [======>.......................] - ETA: 1:01 - loss: 7.7802 - accuracy: 0.4926
 6240/25000 [======>.......................] - ETA: 1:01 - loss: 7.7674 - accuracy: 0.4934
 6272/25000 [======>.......................] - ETA: 1:01 - loss: 7.7620 - accuracy: 0.4938
 6304/25000 [======>.......................] - ETA: 1:01 - loss: 7.7590 - accuracy: 0.4940
 6336/25000 [======>.......................] - ETA: 1:01 - loss: 7.7513 - accuracy: 0.4945
 6368/25000 [======>.......................] - ETA: 1:01 - loss: 7.7389 - accuracy: 0.4953
 6400/25000 [======>.......................] - ETA: 1:01 - loss: 7.7433 - accuracy: 0.4950
 6432/25000 [======>.......................] - ETA: 1:01 - loss: 7.7405 - accuracy: 0.4952
 6464/25000 [======>.......................] - ETA: 1:01 - loss: 7.7425 - accuracy: 0.4950
 6496/25000 [======>.......................] - ETA: 1:00 - loss: 7.7374 - accuracy: 0.4954
 6528/25000 [======>.......................] - ETA: 1:00 - loss: 7.7277 - accuracy: 0.4960
 6560/25000 [======>.......................] - ETA: 1:00 - loss: 7.7391 - accuracy: 0.4953
 6592/25000 [======>.......................] - ETA: 1:00 - loss: 7.7271 - accuracy: 0.4961
 6624/25000 [======>.......................] - ETA: 1:00 - loss: 7.7268 - accuracy: 0.4961
 6656/25000 [======>.......................] - ETA: 1:00 - loss: 7.7357 - accuracy: 0.4955
 6688/25000 [=======>......................] - ETA: 1:00 - loss: 7.7469 - accuracy: 0.4948
 6720/25000 [=======>......................] - ETA: 1:00 - loss: 7.7442 - accuracy: 0.4949
 6752/25000 [=======>......................] - ETA: 1:00 - loss: 7.7484 - accuracy: 0.4947
 6784/25000 [=======>......................] - ETA: 59s - loss: 7.7525 - accuracy: 0.4944 
 6816/25000 [=======>......................] - ETA: 59s - loss: 7.7566 - accuracy: 0.4941
 6848/25000 [=======>......................] - ETA: 59s - loss: 7.7584 - accuracy: 0.4940
 6880/25000 [=======>......................] - ETA: 59s - loss: 7.7580 - accuracy: 0.4940
 6912/25000 [=======>......................] - ETA: 59s - loss: 7.7576 - accuracy: 0.4941
 6944/25000 [=======>......................] - ETA: 59s - loss: 7.7616 - accuracy: 0.4938
 6976/25000 [=======>......................] - ETA: 59s - loss: 7.7655 - accuracy: 0.4935
 7008/25000 [=======>......................] - ETA: 59s - loss: 7.7629 - accuracy: 0.4937
 7040/25000 [=======>......................] - ETA: 59s - loss: 7.7668 - accuracy: 0.4935
 7072/25000 [=======>......................] - ETA: 59s - loss: 7.7599 - accuracy: 0.4939
 7104/25000 [=======>......................] - ETA: 58s - loss: 7.7530 - accuracy: 0.4944
 7136/25000 [=======>......................] - ETA: 58s - loss: 7.7590 - accuracy: 0.4940
 7168/25000 [=======>......................] - ETA: 58s - loss: 7.7629 - accuracy: 0.4937
 7200/25000 [=======>......................] - ETA: 58s - loss: 7.7603 - accuracy: 0.4939
 7232/25000 [=======>......................] - ETA: 58s - loss: 7.7557 - accuracy: 0.4942
 7264/25000 [=======>......................] - ETA: 58s - loss: 7.7574 - accuracy: 0.4941
 7296/25000 [=======>......................] - ETA: 58s - loss: 7.7528 - accuracy: 0.4944
 7328/25000 [=======>......................] - ETA: 58s - loss: 7.7524 - accuracy: 0.4944
 7360/25000 [=======>......................] - ETA: 58s - loss: 7.7500 - accuracy: 0.4946
 7392/25000 [=======>......................] - ETA: 58s - loss: 7.7434 - accuracy: 0.4950
 7424/25000 [=======>......................] - ETA: 57s - loss: 7.7410 - accuracy: 0.4952
 7456/25000 [=======>......................] - ETA: 57s - loss: 7.7386 - accuracy: 0.4953
 7488/25000 [=======>......................] - ETA: 57s - loss: 7.7321 - accuracy: 0.4957
 7520/25000 [========>.....................] - ETA: 57s - loss: 7.7298 - accuracy: 0.4959
 7552/25000 [========>.....................] - ETA: 57s - loss: 7.7397 - accuracy: 0.4952
 7584/25000 [========>.....................] - ETA: 57s - loss: 7.7394 - accuracy: 0.4953
 7616/25000 [========>.....................] - ETA: 57s - loss: 7.7270 - accuracy: 0.4961
 7648/25000 [========>.....................] - ETA: 57s - loss: 7.7208 - accuracy: 0.4965
 7680/25000 [========>.....................] - ETA: 57s - loss: 7.7165 - accuracy: 0.4967
 7712/25000 [========>.....................] - ETA: 56s - loss: 7.7203 - accuracy: 0.4965
 7744/25000 [========>.....................] - ETA: 56s - loss: 7.7240 - accuracy: 0.4963
 7776/25000 [========>.....................] - ETA: 56s - loss: 7.7120 - accuracy: 0.4970
 7808/25000 [========>.....................] - ETA: 56s - loss: 7.7079 - accuracy: 0.4973
 7840/25000 [========>.....................] - ETA: 56s - loss: 7.7096 - accuracy: 0.4972
 7872/25000 [========>.....................] - ETA: 56s - loss: 7.7153 - accuracy: 0.4968
 7904/25000 [========>.....................] - ETA: 56s - loss: 7.7171 - accuracy: 0.4967
 7936/25000 [========>.....................] - ETA: 56s - loss: 7.7111 - accuracy: 0.4971
 7968/25000 [========>.....................] - ETA: 56s - loss: 7.7128 - accuracy: 0.4970
 8000/25000 [========>.....................] - ETA: 56s - loss: 7.7165 - accuracy: 0.4967
 8032/25000 [========>.....................] - ETA: 55s - loss: 7.7182 - accuracy: 0.4966
 8064/25000 [========>.....................] - ETA: 55s - loss: 7.7180 - accuracy: 0.4967
 8096/25000 [========>.....................] - ETA: 55s - loss: 7.7178 - accuracy: 0.4967
 8128/25000 [========>.....................] - ETA: 55s - loss: 7.7157 - accuracy: 0.4968
 8160/25000 [========>.....................] - ETA: 55s - loss: 7.7155 - accuracy: 0.4968
 8192/25000 [========>.....................] - ETA: 55s - loss: 7.7209 - accuracy: 0.4965
 8224/25000 [========>.....................] - ETA: 55s - loss: 7.7188 - accuracy: 0.4966
 8256/25000 [========>.....................] - ETA: 55s - loss: 7.7205 - accuracy: 0.4965
 8288/25000 [========>.....................] - ETA: 55s - loss: 7.7221 - accuracy: 0.4964
 8320/25000 [========>.....................] - ETA: 54s - loss: 7.7238 - accuracy: 0.4963
 8352/25000 [=========>....................] - ETA: 54s - loss: 7.7217 - accuracy: 0.4964
 8384/25000 [=========>....................] - ETA: 54s - loss: 7.7306 - accuracy: 0.4958
 8416/25000 [=========>....................] - ETA: 54s - loss: 7.7304 - accuracy: 0.4958
 8448/25000 [=========>....................] - ETA: 54s - loss: 7.7429 - accuracy: 0.4950
 8480/25000 [=========>....................] - ETA: 54s - loss: 7.7426 - accuracy: 0.4950
 8512/25000 [=========>....................] - ETA: 54s - loss: 7.7387 - accuracy: 0.4953
 8544/25000 [=========>....................] - ETA: 54s - loss: 7.7330 - accuracy: 0.4957
 8576/25000 [=========>....................] - ETA: 54s - loss: 7.7399 - accuracy: 0.4952
 8608/25000 [=========>....................] - ETA: 53s - loss: 7.7432 - accuracy: 0.4950
 8640/25000 [=========>....................] - ETA: 53s - loss: 7.7412 - accuracy: 0.4951
 8672/25000 [=========>....................] - ETA: 53s - loss: 7.7426 - accuracy: 0.4950
 8704/25000 [=========>....................] - ETA: 53s - loss: 7.7477 - accuracy: 0.4947
 8736/25000 [=========>....................] - ETA: 53s - loss: 7.7474 - accuracy: 0.4947
 8768/25000 [=========>....................] - ETA: 53s - loss: 7.7471 - accuracy: 0.4948
 8800/25000 [=========>....................] - ETA: 53s - loss: 7.7520 - accuracy: 0.4944
 8832/25000 [=========>....................] - ETA: 53s - loss: 7.7447 - accuracy: 0.4949
 8864/25000 [=========>....................] - ETA: 53s - loss: 7.7427 - accuracy: 0.4950
 8896/25000 [=========>....................] - ETA: 52s - loss: 7.7356 - accuracy: 0.4955
 8928/25000 [=========>....................] - ETA: 52s - loss: 7.7319 - accuracy: 0.4957
 8960/25000 [=========>....................] - ETA: 52s - loss: 7.7385 - accuracy: 0.4953
 8992/25000 [=========>....................] - ETA: 52s - loss: 7.7365 - accuracy: 0.4954
 9024/25000 [=========>....................] - ETA: 52s - loss: 7.7380 - accuracy: 0.4953
 9056/25000 [=========>....................] - ETA: 52s - loss: 7.7411 - accuracy: 0.4951
 9088/25000 [=========>....................] - ETA: 52s - loss: 7.7375 - accuracy: 0.4954
 9120/25000 [=========>....................] - ETA: 52s - loss: 7.7339 - accuracy: 0.4956
 9152/25000 [=========>....................] - ETA: 52s - loss: 7.7336 - accuracy: 0.4956
 9184/25000 [==========>...................] - ETA: 51s - loss: 7.7367 - accuracy: 0.4954
 9216/25000 [==========>...................] - ETA: 51s - loss: 7.7365 - accuracy: 0.4954
 9248/25000 [==========>...................] - ETA: 51s - loss: 7.7313 - accuracy: 0.4958
 9280/25000 [==========>...................] - ETA: 51s - loss: 7.7311 - accuracy: 0.4958
 9312/25000 [==========>...................] - ETA: 51s - loss: 7.7325 - accuracy: 0.4957
 9344/25000 [==========>...................] - ETA: 51s - loss: 7.7388 - accuracy: 0.4953
 9376/25000 [==========>...................] - ETA: 51s - loss: 7.7304 - accuracy: 0.4958
 9408/25000 [==========>...................] - ETA: 51s - loss: 7.7237 - accuracy: 0.4963
 9440/25000 [==========>...................] - ETA: 51s - loss: 7.7186 - accuracy: 0.4966
 9472/25000 [==========>...................] - ETA: 51s - loss: 7.7119 - accuracy: 0.4970
 9504/25000 [==========>...................] - ETA: 50s - loss: 7.7118 - accuracy: 0.4971
 9536/25000 [==========>...................] - ETA: 50s - loss: 7.7100 - accuracy: 0.4972
 9568/25000 [==========>...................] - ETA: 50s - loss: 7.7099 - accuracy: 0.4972
 9600/25000 [==========>...................] - ETA: 50s - loss: 7.7050 - accuracy: 0.4975
 9632/25000 [==========>...................] - ETA: 50s - loss: 7.7080 - accuracy: 0.4973
 9664/25000 [==========>...................] - ETA: 50s - loss: 7.7110 - accuracy: 0.4971
 9696/25000 [==========>...................] - ETA: 50s - loss: 7.7093 - accuracy: 0.4972
 9728/25000 [==========>...................] - ETA: 50s - loss: 7.7139 - accuracy: 0.4969
 9760/25000 [==========>...................] - ETA: 50s - loss: 7.7169 - accuracy: 0.4967
 9792/25000 [==========>...................] - ETA: 49s - loss: 7.7136 - accuracy: 0.4969
 9824/25000 [==========>...................] - ETA: 49s - loss: 7.7103 - accuracy: 0.4971
 9856/25000 [==========>...................] - ETA: 49s - loss: 7.7102 - accuracy: 0.4972
 9888/25000 [==========>...................] - ETA: 49s - loss: 7.7147 - accuracy: 0.4969
 9920/25000 [==========>...................] - ETA: 49s - loss: 7.7176 - accuracy: 0.4967
 9952/25000 [==========>...................] - ETA: 49s - loss: 7.7175 - accuracy: 0.4967
 9984/25000 [==========>...................] - ETA: 49s - loss: 7.7188 - accuracy: 0.4966
10016/25000 [===========>..................] - ETA: 49s - loss: 7.7187 - accuracy: 0.4966
10048/25000 [===========>..................] - ETA: 49s - loss: 7.7231 - accuracy: 0.4963
10080/25000 [===========>..................] - ETA: 49s - loss: 7.7168 - accuracy: 0.4967
10112/25000 [===========>..................] - ETA: 48s - loss: 7.7167 - accuracy: 0.4967
10144/25000 [===========>..................] - ETA: 48s - loss: 7.7165 - accuracy: 0.4967
10176/25000 [===========>..................] - ETA: 48s - loss: 7.7118 - accuracy: 0.4971
10208/25000 [===========>..................] - ETA: 48s - loss: 7.7147 - accuracy: 0.4969
10240/25000 [===========>..................] - ETA: 48s - loss: 7.7145 - accuracy: 0.4969
10272/25000 [===========>..................] - ETA: 48s - loss: 7.7084 - accuracy: 0.4973
10304/25000 [===========>..................] - ETA: 48s - loss: 7.7128 - accuracy: 0.4970
10336/25000 [===========>..................] - ETA: 48s - loss: 7.7037 - accuracy: 0.4976
10368/25000 [===========>..................] - ETA: 48s - loss: 7.7065 - accuracy: 0.4974
10400/25000 [===========>..................] - ETA: 47s - loss: 7.7167 - accuracy: 0.4967
10432/25000 [===========>..................] - ETA: 47s - loss: 7.7166 - accuracy: 0.4967
10464/25000 [===========>..................] - ETA: 47s - loss: 7.7164 - accuracy: 0.4968
10496/25000 [===========>..................] - ETA: 47s - loss: 7.7090 - accuracy: 0.4972
10528/25000 [===========>..................] - ETA: 47s - loss: 7.7045 - accuracy: 0.4975
10560/25000 [===========>..................] - ETA: 47s - loss: 7.7116 - accuracy: 0.4971
10592/25000 [===========>..................] - ETA: 47s - loss: 7.7173 - accuracy: 0.4967
10624/25000 [===========>..................] - ETA: 47s - loss: 7.7215 - accuracy: 0.4964
10656/25000 [===========>..................] - ETA: 47s - loss: 7.7199 - accuracy: 0.4965
10688/25000 [===========>..................] - ETA: 46s - loss: 7.7168 - accuracy: 0.4967
10720/25000 [===========>..................] - ETA: 46s - loss: 7.7181 - accuracy: 0.4966
10752/25000 [===========>..................] - ETA: 46s - loss: 7.7108 - accuracy: 0.4971
10784/25000 [===========>..................] - ETA: 46s - loss: 7.7079 - accuracy: 0.4973
10816/25000 [===========>..................] - ETA: 46s - loss: 7.7077 - accuracy: 0.4973
10848/25000 [============>.................] - ETA: 46s - loss: 7.7048 - accuracy: 0.4975
10880/25000 [============>.................] - ETA: 46s - loss: 7.7047 - accuracy: 0.4975
10912/25000 [============>.................] - ETA: 46s - loss: 7.7116 - accuracy: 0.4971
10944/25000 [============>.................] - ETA: 46s - loss: 7.7101 - accuracy: 0.4972
10976/25000 [============>.................] - ETA: 45s - loss: 7.7071 - accuracy: 0.4974
11008/25000 [============>.................] - ETA: 45s - loss: 7.7126 - accuracy: 0.4970
11040/25000 [============>.................] - ETA: 45s - loss: 7.7111 - accuracy: 0.4971
11072/25000 [============>.................] - ETA: 45s - loss: 7.7026 - accuracy: 0.4977
11104/25000 [============>.................] - ETA: 45s - loss: 7.7094 - accuracy: 0.4972
11136/25000 [============>.................] - ETA: 45s - loss: 7.7121 - accuracy: 0.4970
11168/25000 [============>.................] - ETA: 45s - loss: 7.7106 - accuracy: 0.4971
11200/25000 [============>.................] - ETA: 45s - loss: 7.7145 - accuracy: 0.4969
11232/25000 [============>.................] - ETA: 45s - loss: 7.7130 - accuracy: 0.4970
11264/25000 [============>.................] - ETA: 45s - loss: 7.7115 - accuracy: 0.4971
11296/25000 [============>.................] - ETA: 44s - loss: 7.7087 - accuracy: 0.4973
11328/25000 [============>.................] - ETA: 44s - loss: 7.7045 - accuracy: 0.4975
11360/25000 [============>.................] - ETA: 44s - loss: 7.7031 - accuracy: 0.4976
11392/25000 [============>.................] - ETA: 44s - loss: 7.7043 - accuracy: 0.4975
11424/25000 [============>.................] - ETA: 44s - loss: 7.7123 - accuracy: 0.4970
11456/25000 [============>.................] - ETA: 44s - loss: 7.7175 - accuracy: 0.4967
11488/25000 [============>.................] - ETA: 44s - loss: 7.7147 - accuracy: 0.4969
11520/25000 [============>.................] - ETA: 44s - loss: 7.7119 - accuracy: 0.4970
11552/25000 [============>.................] - ETA: 44s - loss: 7.7184 - accuracy: 0.4966
11584/25000 [============>.................] - ETA: 43s - loss: 7.7262 - accuracy: 0.4961
11616/25000 [============>.................] - ETA: 43s - loss: 7.7300 - accuracy: 0.4959
11648/25000 [============>.................] - ETA: 43s - loss: 7.7272 - accuracy: 0.4961
11680/25000 [=============>................] - ETA: 43s - loss: 7.7270 - accuracy: 0.4961
11712/25000 [=============>................] - ETA: 43s - loss: 7.7295 - accuracy: 0.4959
11744/25000 [=============>................] - ETA: 43s - loss: 7.7267 - accuracy: 0.4961
11776/25000 [=============>................] - ETA: 43s - loss: 7.7304 - accuracy: 0.4958
11808/25000 [=============>................] - ETA: 43s - loss: 7.7315 - accuracy: 0.4958
11840/25000 [=============>................] - ETA: 43s - loss: 7.7314 - accuracy: 0.4958
11872/25000 [=============>................] - ETA: 43s - loss: 7.7312 - accuracy: 0.4958
11904/25000 [=============>................] - ETA: 42s - loss: 7.7336 - accuracy: 0.4956
11936/25000 [=============>................] - ETA: 42s - loss: 7.7334 - accuracy: 0.4956
11968/25000 [=============>................] - ETA: 42s - loss: 7.7358 - accuracy: 0.4955
12000/25000 [=============>................] - ETA: 42s - loss: 7.7382 - accuracy: 0.4953
12032/25000 [=============>................] - ETA: 42s - loss: 7.7393 - accuracy: 0.4953
12064/25000 [=============>................] - ETA: 42s - loss: 7.7416 - accuracy: 0.4951
12096/25000 [=============>................] - ETA: 42s - loss: 7.7376 - accuracy: 0.4954
12128/25000 [=============>................] - ETA: 42s - loss: 7.7273 - accuracy: 0.4960
12160/25000 [=============>................] - ETA: 42s - loss: 7.7284 - accuracy: 0.4960
12192/25000 [=============>................] - ETA: 41s - loss: 7.7345 - accuracy: 0.4956
12224/25000 [=============>................] - ETA: 41s - loss: 7.7394 - accuracy: 0.4953
12256/25000 [=============>................] - ETA: 41s - loss: 7.7442 - accuracy: 0.4949
12288/25000 [=============>................] - ETA: 41s - loss: 7.7477 - accuracy: 0.4947
12320/25000 [=============>................] - ETA: 41s - loss: 7.7500 - accuracy: 0.4946
12352/25000 [=============>................] - ETA: 41s - loss: 7.7523 - accuracy: 0.4944
12384/25000 [=============>................] - ETA: 41s - loss: 7.7545 - accuracy: 0.4943
12416/25000 [=============>................] - ETA: 41s - loss: 7.7506 - accuracy: 0.4945
12448/25000 [=============>................] - ETA: 41s - loss: 7.7528 - accuracy: 0.4944
12480/25000 [=============>................] - ETA: 40s - loss: 7.7563 - accuracy: 0.4942
12512/25000 [==============>...............] - ETA: 40s - loss: 7.7524 - accuracy: 0.4944
12544/25000 [==============>...............] - ETA: 40s - loss: 7.7473 - accuracy: 0.4947
12576/25000 [==============>...............] - ETA: 40s - loss: 7.7447 - accuracy: 0.4949
12608/25000 [==============>...............] - ETA: 40s - loss: 7.7457 - accuracy: 0.4948
12640/25000 [==============>...............] - ETA: 40s - loss: 7.7479 - accuracy: 0.4947
12672/25000 [==============>...............] - ETA: 40s - loss: 7.7416 - accuracy: 0.4951
12704/25000 [==============>...............] - ETA: 40s - loss: 7.7463 - accuracy: 0.4948
12736/25000 [==============>...............] - ETA: 40s - loss: 7.7413 - accuracy: 0.4951
12768/25000 [==============>...............] - ETA: 40s - loss: 7.7423 - accuracy: 0.4951
12800/25000 [==============>...............] - ETA: 39s - loss: 7.7457 - accuracy: 0.4948
12832/25000 [==============>...............] - ETA: 39s - loss: 7.7371 - accuracy: 0.4954
12864/25000 [==============>...............] - ETA: 39s - loss: 7.7405 - accuracy: 0.4952
12896/25000 [==============>...............] - ETA: 39s - loss: 7.7463 - accuracy: 0.4948
12928/25000 [==============>...............] - ETA: 39s - loss: 7.7413 - accuracy: 0.4951
12960/25000 [==============>...............] - ETA: 39s - loss: 7.7423 - accuracy: 0.4951
12992/25000 [==============>...............] - ETA: 39s - loss: 7.7457 - accuracy: 0.4948
13024/25000 [==============>...............] - ETA: 39s - loss: 7.7467 - accuracy: 0.4948
13056/25000 [==============>...............] - ETA: 39s - loss: 7.7453 - accuracy: 0.4949
13088/25000 [==============>...............] - ETA: 38s - loss: 7.7416 - accuracy: 0.4951
13120/25000 [==============>...............] - ETA: 38s - loss: 7.7379 - accuracy: 0.4954
13152/25000 [==============>...............] - ETA: 38s - loss: 7.7389 - accuracy: 0.4953
13184/25000 [==============>...............] - ETA: 38s - loss: 7.7364 - accuracy: 0.4954
13216/25000 [==============>...............] - ETA: 38s - loss: 7.7386 - accuracy: 0.4953
13248/25000 [==============>...............] - ETA: 38s - loss: 7.7419 - accuracy: 0.4951
13280/25000 [==============>...............] - ETA: 38s - loss: 7.7417 - accuracy: 0.4951
13312/25000 [==============>...............] - ETA: 38s - loss: 7.7403 - accuracy: 0.4952
13344/25000 [===============>..............] - ETA: 38s - loss: 7.7390 - accuracy: 0.4953
13376/25000 [===============>..............] - ETA: 38s - loss: 7.7377 - accuracy: 0.4954
13408/25000 [===============>..............] - ETA: 37s - loss: 7.7398 - accuracy: 0.4952
13440/25000 [===============>..............] - ETA: 37s - loss: 7.7362 - accuracy: 0.4955
13472/25000 [===============>..............] - ETA: 37s - loss: 7.7235 - accuracy: 0.4963
13504/25000 [===============>..............] - ETA: 37s - loss: 7.7189 - accuracy: 0.4966
13536/25000 [===============>..............] - ETA: 37s - loss: 7.7199 - accuracy: 0.4965
13568/25000 [===============>..............] - ETA: 37s - loss: 7.7175 - accuracy: 0.4967
13600/25000 [===============>..............] - ETA: 37s - loss: 7.7196 - accuracy: 0.4965
13632/25000 [===============>..............] - ETA: 37s - loss: 7.7206 - accuracy: 0.4965
13664/25000 [===============>..............] - ETA: 37s - loss: 7.7171 - accuracy: 0.4967
13696/25000 [===============>..............] - ETA: 36s - loss: 7.7136 - accuracy: 0.4969
13728/25000 [===============>..............] - ETA: 36s - loss: 7.7113 - accuracy: 0.4971
13760/25000 [===============>..............] - ETA: 36s - loss: 7.7101 - accuracy: 0.4972
13792/25000 [===============>..............] - ETA: 36s - loss: 7.7100 - accuracy: 0.4972
13824/25000 [===============>..............] - ETA: 36s - loss: 7.7054 - accuracy: 0.4975
13856/25000 [===============>..............] - ETA: 36s - loss: 7.7020 - accuracy: 0.4977
13888/25000 [===============>..............] - ETA: 36s - loss: 7.6942 - accuracy: 0.4982
13920/25000 [===============>..............] - ETA: 36s - loss: 7.6909 - accuracy: 0.4984
13952/25000 [===============>..............] - ETA: 36s - loss: 7.6908 - accuracy: 0.4984
13984/25000 [===============>..............] - ETA: 36s - loss: 7.6918 - accuracy: 0.4984
14016/25000 [===============>..............] - ETA: 35s - loss: 7.6896 - accuracy: 0.4985
14048/25000 [===============>..............] - ETA: 35s - loss: 7.6884 - accuracy: 0.4986
14080/25000 [===============>..............] - ETA: 35s - loss: 7.6873 - accuracy: 0.4987
14112/25000 [===============>..............] - ETA: 35s - loss: 7.6884 - accuracy: 0.4986
14144/25000 [===============>..............] - ETA: 35s - loss: 7.6894 - accuracy: 0.4985
14176/25000 [================>.............] - ETA: 35s - loss: 7.6947 - accuracy: 0.4982
14208/25000 [================>.............] - ETA: 35s - loss: 7.6936 - accuracy: 0.4982
14240/25000 [================>.............] - ETA: 35s - loss: 7.6935 - accuracy: 0.4982
14272/25000 [================>.............] - ETA: 35s - loss: 7.6946 - accuracy: 0.4982
14304/25000 [================>.............] - ETA: 34s - loss: 7.6945 - accuracy: 0.4982
14336/25000 [================>.............] - ETA: 34s - loss: 7.6944 - accuracy: 0.4982
14368/25000 [================>.............] - ETA: 34s - loss: 7.6997 - accuracy: 0.4978
14400/25000 [================>.............] - ETA: 34s - loss: 7.6986 - accuracy: 0.4979
14432/25000 [================>.............] - ETA: 34s - loss: 7.6942 - accuracy: 0.4982
14464/25000 [================>.............] - ETA: 34s - loss: 7.6963 - accuracy: 0.4981
14496/25000 [================>.............] - ETA: 34s - loss: 7.6920 - accuracy: 0.4983
14528/25000 [================>.............] - ETA: 34s - loss: 7.6951 - accuracy: 0.4981
14560/25000 [================>.............] - ETA: 34s - loss: 7.6929 - accuracy: 0.4983
14592/25000 [================>.............] - ETA: 34s - loss: 7.6876 - accuracy: 0.4986
14624/25000 [================>.............] - ETA: 33s - loss: 7.6907 - accuracy: 0.4984
14656/25000 [================>.............] - ETA: 33s - loss: 7.6928 - accuracy: 0.4983
14688/25000 [================>.............] - ETA: 33s - loss: 7.6917 - accuracy: 0.4984
14720/25000 [================>.............] - ETA: 33s - loss: 7.6937 - accuracy: 0.4982
14752/25000 [================>.............] - ETA: 33s - loss: 7.6947 - accuracy: 0.4982
14784/25000 [================>.............] - ETA: 33s - loss: 7.6946 - accuracy: 0.4982
14816/25000 [================>.............] - ETA: 33s - loss: 7.6904 - accuracy: 0.4984
14848/25000 [================>.............] - ETA: 33s - loss: 7.6904 - accuracy: 0.4985
14880/25000 [================>.............] - ETA: 33s - loss: 7.6914 - accuracy: 0.4984
14912/25000 [================>.............] - ETA: 32s - loss: 7.6913 - accuracy: 0.4984
14944/25000 [================>.............] - ETA: 32s - loss: 7.6892 - accuracy: 0.4985
14976/25000 [================>.............] - ETA: 32s - loss: 7.6902 - accuracy: 0.4985
15008/25000 [=================>............] - ETA: 32s - loss: 7.6911 - accuracy: 0.4984
15040/25000 [=================>............] - ETA: 32s - loss: 7.6901 - accuracy: 0.4985
15072/25000 [=================>............] - ETA: 32s - loss: 7.6900 - accuracy: 0.4985
15104/25000 [=================>............] - ETA: 32s - loss: 7.6890 - accuracy: 0.4985
15136/25000 [=================>............] - ETA: 32s - loss: 7.6828 - accuracy: 0.4989
15168/25000 [=================>............] - ETA: 32s - loss: 7.6798 - accuracy: 0.4991
15200/25000 [=================>............] - ETA: 32s - loss: 7.6797 - accuracy: 0.4991
15232/25000 [=================>............] - ETA: 31s - loss: 7.6807 - accuracy: 0.4991
15264/25000 [=================>............] - ETA: 31s - loss: 7.6817 - accuracy: 0.4990
15296/25000 [=================>............] - ETA: 31s - loss: 7.6817 - accuracy: 0.4990
15328/25000 [=================>............] - ETA: 31s - loss: 7.6806 - accuracy: 0.4991
15360/25000 [=================>............] - ETA: 31s - loss: 7.6786 - accuracy: 0.4992
15392/25000 [=================>............] - ETA: 31s - loss: 7.6776 - accuracy: 0.4993
15424/25000 [=================>............] - ETA: 31s - loss: 7.6746 - accuracy: 0.4995
15456/25000 [=================>............] - ETA: 31s - loss: 7.6726 - accuracy: 0.4996
15488/25000 [=================>............] - ETA: 31s - loss: 7.6755 - accuracy: 0.4994
15520/25000 [=================>............] - ETA: 30s - loss: 7.6745 - accuracy: 0.4995
15552/25000 [=================>............] - ETA: 30s - loss: 7.6785 - accuracy: 0.4992
15584/25000 [=================>............] - ETA: 30s - loss: 7.6774 - accuracy: 0.4993
15616/25000 [=================>............] - ETA: 30s - loss: 7.6745 - accuracy: 0.4995
15648/25000 [=================>............] - ETA: 30s - loss: 7.6696 - accuracy: 0.4998
15680/25000 [=================>............] - ETA: 30s - loss: 7.6705 - accuracy: 0.4997
15712/25000 [=================>............] - ETA: 30s - loss: 7.6695 - accuracy: 0.4998
15744/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15776/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15808/25000 [=================>............] - ETA: 30s - loss: 7.6608 - accuracy: 0.5004
15840/25000 [==================>...........] - ETA: 29s - loss: 7.6550 - accuracy: 0.5008
15872/25000 [==================>...........] - ETA: 29s - loss: 7.6531 - accuracy: 0.5009
15904/25000 [==================>...........] - ETA: 29s - loss: 7.6541 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 29s - loss: 7.6551 - accuracy: 0.5008
15968/25000 [==================>...........] - ETA: 29s - loss: 7.6541 - accuracy: 0.5008
16000/25000 [==================>...........] - ETA: 29s - loss: 7.6532 - accuracy: 0.5009
16032/25000 [==================>...........] - ETA: 29s - loss: 7.6532 - accuracy: 0.5009
16064/25000 [==================>...........] - ETA: 29s - loss: 7.6571 - accuracy: 0.5006
16096/25000 [==================>...........] - ETA: 29s - loss: 7.6590 - accuracy: 0.5005
16128/25000 [==================>...........] - ETA: 29s - loss: 7.6571 - accuracy: 0.5006
16160/25000 [==================>...........] - ETA: 28s - loss: 7.6609 - accuracy: 0.5004
16192/25000 [==================>...........] - ETA: 28s - loss: 7.6628 - accuracy: 0.5002
16224/25000 [==================>...........] - ETA: 28s - loss: 7.6657 - accuracy: 0.5001
16256/25000 [==================>...........] - ETA: 28s - loss: 7.6694 - accuracy: 0.4998
16288/25000 [==================>...........] - ETA: 28s - loss: 7.6713 - accuracy: 0.4997
16320/25000 [==================>...........] - ETA: 28s - loss: 7.6694 - accuracy: 0.4998
16352/25000 [==================>...........] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
16384/25000 [==================>...........] - ETA: 28s - loss: 7.6619 - accuracy: 0.5003
16416/25000 [==================>...........] - ETA: 28s - loss: 7.6610 - accuracy: 0.5004
16448/25000 [==================>...........] - ETA: 27s - loss: 7.6601 - accuracy: 0.5004
16480/25000 [==================>...........] - ETA: 27s - loss: 7.6601 - accuracy: 0.5004
16512/25000 [==================>...........] - ETA: 27s - loss: 7.6564 - accuracy: 0.5007
16544/25000 [==================>...........] - ETA: 27s - loss: 7.6564 - accuracy: 0.5007
16576/25000 [==================>...........] - ETA: 27s - loss: 7.6564 - accuracy: 0.5007
16608/25000 [==================>...........] - ETA: 27s - loss: 7.6583 - accuracy: 0.5005
16640/25000 [==================>...........] - ETA: 27s - loss: 7.6611 - accuracy: 0.5004
16672/25000 [===================>..........] - ETA: 27s - loss: 7.6620 - accuracy: 0.5003
16704/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
16736/25000 [===================>..........] - ETA: 27s - loss: 7.6675 - accuracy: 0.4999
16768/25000 [===================>..........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
16800/25000 [===================>..........] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
16832/25000 [===================>..........] - ETA: 26s - loss: 7.6648 - accuracy: 0.5001
16864/25000 [===================>..........] - ETA: 26s - loss: 7.6657 - accuracy: 0.5001
16896/25000 [===================>..........] - ETA: 26s - loss: 7.6630 - accuracy: 0.5002
16928/25000 [===================>..........] - ETA: 26s - loss: 7.6639 - accuracy: 0.5002
16960/25000 [===================>..........] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
16992/25000 [===================>..........] - ETA: 26s - loss: 7.6657 - accuracy: 0.5001
17024/25000 [===================>..........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
17056/25000 [===================>..........] - ETA: 25s - loss: 7.6693 - accuracy: 0.4998
17088/25000 [===================>..........] - ETA: 25s - loss: 7.6720 - accuracy: 0.4996
17120/25000 [===================>..........] - ETA: 25s - loss: 7.6684 - accuracy: 0.4999
17152/25000 [===================>..........] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
17184/25000 [===================>..........] - ETA: 25s - loss: 7.6684 - accuracy: 0.4999
17216/25000 [===================>..........] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
17248/25000 [===================>..........] - ETA: 25s - loss: 7.6657 - accuracy: 0.5001
17280/25000 [===================>..........] - ETA: 25s - loss: 7.6640 - accuracy: 0.5002
17312/25000 [===================>..........] - ETA: 25s - loss: 7.6657 - accuracy: 0.5001
17344/25000 [===================>..........] - ETA: 25s - loss: 7.6657 - accuracy: 0.5001
17376/25000 [===================>..........] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
17408/25000 [===================>..........] - ETA: 24s - loss: 7.6649 - accuracy: 0.5001
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6657 - accuracy: 0.5001
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6657 - accuracy: 0.5001
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6675 - accuracy: 0.4999
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6649 - accuracy: 0.5001
17568/25000 [====================>.........] - ETA: 24s - loss: 7.6631 - accuracy: 0.5002
17600/25000 [====================>.........] - ETA: 24s - loss: 7.6631 - accuracy: 0.5002
17632/25000 [====================>.........] - ETA: 24s - loss: 7.6614 - accuracy: 0.5003
17664/25000 [====================>.........] - ETA: 24s - loss: 7.6623 - accuracy: 0.5003
17696/25000 [====================>.........] - ETA: 23s - loss: 7.6684 - accuracy: 0.4999
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6692 - accuracy: 0.4998
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6692 - accuracy: 0.4998
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6718 - accuracy: 0.4997
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6743 - accuracy: 0.4995
17888/25000 [====================>.........] - ETA: 23s - loss: 7.6769 - accuracy: 0.4993
17920/25000 [====================>.........] - ETA: 23s - loss: 7.6760 - accuracy: 0.4994
17952/25000 [====================>.........] - ETA: 23s - loss: 7.6752 - accuracy: 0.4994
17984/25000 [====================>.........] - ETA: 22s - loss: 7.6760 - accuracy: 0.4994
18016/25000 [====================>.........] - ETA: 22s - loss: 7.6734 - accuracy: 0.4996
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6751 - accuracy: 0.4994
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6751 - accuracy: 0.4994
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6751 - accuracy: 0.4994
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6734 - accuracy: 0.4996
18176/25000 [====================>.........] - ETA: 22s - loss: 7.6742 - accuracy: 0.4995
18208/25000 [====================>.........] - ETA: 22s - loss: 7.6708 - accuracy: 0.4997
18240/25000 [====================>.........] - ETA: 22s - loss: 7.6683 - accuracy: 0.4999
18272/25000 [====================>.........] - ETA: 22s - loss: 7.6683 - accuracy: 0.4999
18304/25000 [====================>.........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6649 - accuracy: 0.5001
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6641 - accuracy: 0.5002
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6633 - accuracy: 0.5002
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6608 - accuracy: 0.5004
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6583 - accuracy: 0.5005
18496/25000 [=====================>........] - ETA: 21s - loss: 7.6542 - accuracy: 0.5008
18528/25000 [=====================>........] - ETA: 21s - loss: 7.6501 - accuracy: 0.5011
18560/25000 [=====================>........] - ETA: 21s - loss: 7.6509 - accuracy: 0.5010
18592/25000 [=====================>........] - ETA: 20s - loss: 7.6518 - accuracy: 0.5010
18624/25000 [=====================>........] - ETA: 20s - loss: 7.6518 - accuracy: 0.5010
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6494 - accuracy: 0.5011
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6502 - accuracy: 0.5011
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6502 - accuracy: 0.5011
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6494 - accuracy: 0.5011
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6519 - accuracy: 0.5010
18816/25000 [=====================>........] - ETA: 20s - loss: 7.6520 - accuracy: 0.5010
18848/25000 [=====================>........] - ETA: 20s - loss: 7.6536 - accuracy: 0.5008
18880/25000 [=====================>........] - ETA: 20s - loss: 7.6536 - accuracy: 0.5008
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6504 - accuracy: 0.5011
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6529 - accuracy: 0.5009
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6505 - accuracy: 0.5011
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6521 - accuracy: 0.5009
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6513 - accuracy: 0.5010
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6521 - accuracy: 0.5009
19104/25000 [=====================>........] - ETA: 19s - loss: 7.6498 - accuracy: 0.5011
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6490 - accuracy: 0.5011
19168/25000 [======================>.......] - ETA: 19s - loss: 7.6498 - accuracy: 0.5011
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6506 - accuracy: 0.5010
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6531 - accuracy: 0.5009
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6507 - accuracy: 0.5010
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6491 - accuracy: 0.5011
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6523 - accuracy: 0.5009
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6516 - accuracy: 0.5010
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6540 - accuracy: 0.5008
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6524 - accuracy: 0.5009
19456/25000 [======================>.......] - ETA: 18s - loss: 7.6501 - accuracy: 0.5011
19488/25000 [======================>.......] - ETA: 18s - loss: 7.6469 - accuracy: 0.5013
19520/25000 [======================>.......] - ETA: 17s - loss: 7.6493 - accuracy: 0.5011
19552/25000 [======================>.......] - ETA: 17s - loss: 7.6470 - accuracy: 0.5013
19584/25000 [======================>.......] - ETA: 17s - loss: 7.6463 - accuracy: 0.5013
19616/25000 [======================>.......] - ETA: 17s - loss: 7.6455 - accuracy: 0.5014
19648/25000 [======================>.......] - ETA: 17s - loss: 7.6455 - accuracy: 0.5014
19680/25000 [======================>.......] - ETA: 17s - loss: 7.6456 - accuracy: 0.5014
19712/25000 [======================>.......] - ETA: 17s - loss: 7.6433 - accuracy: 0.5015
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6457 - accuracy: 0.5014
19776/25000 [======================>.......] - ETA: 17s - loss: 7.6449 - accuracy: 0.5014
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6434 - accuracy: 0.5015
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6442 - accuracy: 0.5015
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6450 - accuracy: 0.5014
19904/25000 [======================>.......] - ETA: 16s - loss: 7.6443 - accuracy: 0.5015
19936/25000 [======================>.......] - ETA: 16s - loss: 7.6451 - accuracy: 0.5014
19968/25000 [======================>.......] - ETA: 16s - loss: 7.6459 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 16s - loss: 7.6475 - accuracy: 0.5013
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6467 - accuracy: 0.5013
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6467 - accuracy: 0.5013
20096/25000 [=======================>......] - ETA: 16s - loss: 7.6475 - accuracy: 0.5012
20128/25000 [=======================>......] - ETA: 15s - loss: 7.6521 - accuracy: 0.5009
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6537 - accuracy: 0.5008
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6560 - accuracy: 0.5007
20224/25000 [=======================>......] - ETA: 15s - loss: 7.6560 - accuracy: 0.5007
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6583 - accuracy: 0.5005
20288/25000 [=======================>......] - ETA: 15s - loss: 7.6583 - accuracy: 0.5005
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6613 - accuracy: 0.5003
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6629 - accuracy: 0.5002
20384/25000 [=======================>......] - ETA: 15s - loss: 7.6651 - accuracy: 0.5001
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6636 - accuracy: 0.5002
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6621 - accuracy: 0.5003
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6636 - accuracy: 0.5002
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6651 - accuracy: 0.5001
20544/25000 [=======================>......] - ETA: 14s - loss: 7.6614 - accuracy: 0.5003
20576/25000 [=======================>......] - ETA: 14s - loss: 7.6614 - accuracy: 0.5003
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6607 - accuracy: 0.5004
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6622 - accuracy: 0.5003
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6607 - accuracy: 0.5004
20704/25000 [=======================>......] - ETA: 14s - loss: 7.6592 - accuracy: 0.5005
20736/25000 [=======================>......] - ETA: 13s - loss: 7.6600 - accuracy: 0.5004
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6607 - accuracy: 0.5004
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6607 - accuracy: 0.5004
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6607 - accuracy: 0.5004
20864/25000 [========================>.....] - ETA: 13s - loss: 7.6615 - accuracy: 0.5003
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6593 - accuracy: 0.5005
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6586 - accuracy: 0.5005
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6608 - accuracy: 0.5004
21024/25000 [========================>.....] - ETA: 12s - loss: 7.6586 - accuracy: 0.5005
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6601 - accuracy: 0.5004
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6579 - accuracy: 0.5006
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6572 - accuracy: 0.5006
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6594 - accuracy: 0.5005
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6587 - accuracy: 0.5005
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6601 - accuracy: 0.5004
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6587 - accuracy: 0.5005
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6601 - accuracy: 0.5004
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6587 - accuracy: 0.5005
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6551 - accuracy: 0.5007
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6552 - accuracy: 0.5007
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6595 - accuracy: 0.5005
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6623 - accuracy: 0.5003
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6623 - accuracy: 0.5003
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6638 - accuracy: 0.5002
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6645 - accuracy: 0.5001
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6624 - accuracy: 0.5003 
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6617 - accuracy: 0.5003
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6583 - accuracy: 0.5005
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6590 - accuracy: 0.5005
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6583 - accuracy: 0.5005
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6555 - accuracy: 0.5007
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6548 - accuracy: 0.5008
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6583 - accuracy: 0.5005
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6583 - accuracy: 0.5005
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6583 - accuracy: 0.5005
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6584 - accuracy: 0.5005
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6536 - accuracy: 0.5009
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6536 - accuracy: 0.5009
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6550 - accuracy: 0.5008
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6536 - accuracy: 0.5008
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6496 - accuracy: 0.5011
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6496 - accuracy: 0.5011
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6489 - accuracy: 0.5012
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6517 - accuracy: 0.5010
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6537 - accuracy: 0.5008
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6551 - accuracy: 0.5008
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6538 - accuracy: 0.5008
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6531 - accuracy: 0.5009
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6531 - accuracy: 0.5009
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6525 - accuracy: 0.5009
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6559 - accuracy: 0.5007
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6592 - accuracy: 0.5005
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6599 - accuracy: 0.5004
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6606 - accuracy: 0.5004
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6620 - accuracy: 0.5003
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6613 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24192/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24224/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24352/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24384/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24512/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24608/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 1s - loss: 7.6648 - accuracy: 0.5001
24704/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24768/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24800/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
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

