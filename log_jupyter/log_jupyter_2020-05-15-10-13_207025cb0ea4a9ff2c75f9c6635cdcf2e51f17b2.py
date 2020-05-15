
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
 40%|████      | 2/5 [00:49<01:14, 24.81s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.2610015159878141, 'embedding_size_factor': 1.1538966000308002, 'layers.choice': 0, 'learning_rate': 0.00012207301289485754, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 6.31924850492472e-09} and reward: 0.296
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd0\xb4?\xb3\xd7\xf0\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2v\\H\x01\x90\xeaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G? \x00\x172;(\xc6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>;$\x16Sd\x84\x9du.' and reward: 0.296
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd0\xb4?\xb3\xd7\xf0\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2v\\H\x01\x90\xeaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G? \x00\x172;(\xc6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>;$\x16Sd\x84\x9du.' and reward: 0.296
 60%|██████    | 3/5 [01:39<01:04, 32.28s/it] 60%|██████    | 3/5 [01:39<01:06, 33.12s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.0598482280324238, 'embedding_size_factor': 1.213762136772901, 'layers.choice': 2, 'learning_rate': 0.0020789589206433937, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.5572965142601106e-08} and reward: 0.3884
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xae\xa4mL>\xe1\nX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3k\x91\xd8\xa9\x00\xc6X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?a\x07\xe4\x92]PPX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>[uqq%\xddNu.' and reward: 0.3884
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xae\xa4mL>\xe1\nX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3k\x91\xd8\xa9\x00\xc6X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?a\x07\xe4\x92]PPX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>[uqq%\xddNu.' and reward: 0.3884
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 150.38751864433289
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.0598482280324238, 'embedding_size_factor': 1.213762136772901, 'layers.choice': 2, 'learning_rate': 0.0020789589206433937, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.5572965142601106e-08}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -32.91s of remaining time.
Ensemble size: 72
Ensemble weights: 
[0.48611111 0.25       0.26388889]
	0.3922	 = Validation accuracy score
	1.04s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 153.99s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fec15fbca90> 

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
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 6.04594536e-02  4.15104441e-02  8.41841381e-03  6.51260912e-02
   8.18756595e-02  1.25632867e-01]
 [ 8.45183730e-02  1.04473166e-01 -7.83751607e-02  1.60291106e-01
   2.05471739e-01  2.65858043e-02]
 [-1.28148064e-01 -3.72951180e-02  6.52569756e-02  8.26811120e-02
  -9.12253745e-05  6.88945800e-02]
 [ 4.37709410e-03 -9.19762254e-02 -9.25176665e-02  3.35674703e-01
   4.38614756e-01 -6.35324642e-02]
 [-2.09759548e-01  8.79245028e-02 -2.94018537e-01 -1.06748730e-01
   2.54852980e-01 -1.36350438e-01]
 [ 1.10339150e-01  1.07718691e-01  3.71407181e-01  3.94036233e-01
  -3.52399886e-01 -2.18380421e-01]
 [ 3.44522476e-01 -6.35208040e-02  6.64064944e-01 -2.43064329e-01
   2.43804231e-01 -1.90333277e-01]
 [ 1.53613582e-01  3.13383877e-01 -2.65212525e-02 -2.07046419e-01
  -3.53838652e-01  5.25744736e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]

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
{'loss': 0.3936995416879654, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 10:16:55.280954: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.48548508435487747, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 10:16:56.374721: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
   49152/17464789 [..............................] - ETA: 43s
   81920/17464789 [..............................] - ETA: 39s
  155648/17464789 [..............................] - ETA: 27s
  294912/17464789 [..............................] - ETA: 18s
  573440/17464789 [..............................] - ETA: 10s
 1114112/17464789 [>.............................] - ETA: 6s 
 2211840/17464789 [==>...........................] - ETA: 3s
 4374528/17464789 [======>.......................] - ETA: 1s
 7454720/17464789 [===========>..................] - ETA: 0s
10485760/17464789 [=================>............] - ETA: 0s
13565952/17464789 [======================>.......] - ETA: 0s
16678912/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 10:17:08.510081: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 10:17:08.514494: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 10:17:08.514627: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5623e4045820 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 10:17:08.514640: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:50 - loss: 5.7500 - accuracy: 0.6250
   64/25000 [..............................] - ETA: 2:57 - loss: 6.4687 - accuracy: 0.5781
   96/25000 [..............................] - ETA: 2:16 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 1:57 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:46 - loss: 7.7625 - accuracy: 0.4938
  192/25000 [..............................] - ETA: 1:38 - loss: 7.6666 - accuracy: 0.5000
  224/25000 [..............................] - ETA: 1:32 - loss: 7.1875 - accuracy: 0.5312
  256/25000 [..............................] - ETA: 1:28 - loss: 7.1875 - accuracy: 0.5312
  288/25000 [..............................] - ETA: 1:25 - loss: 7.1342 - accuracy: 0.5347
  320/25000 [..............................] - ETA: 1:22 - loss: 7.0916 - accuracy: 0.5375
  352/25000 [..............................] - ETA: 1:19 - loss: 7.1875 - accuracy: 0.5312
  384/25000 [..............................] - ETA: 1:17 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:16 - loss: 7.3349 - accuracy: 0.5216
  448/25000 [..............................] - ETA: 1:15 - loss: 7.2901 - accuracy: 0.5246
  480/25000 [..............................] - ETA: 1:14 - loss: 7.3152 - accuracy: 0.5229
  512/25000 [..............................] - ETA: 1:13 - loss: 7.3072 - accuracy: 0.5234
  544/25000 [..............................] - ETA: 1:12 - loss: 7.2156 - accuracy: 0.5294
  576/25000 [..............................] - ETA: 1:11 - loss: 7.2407 - accuracy: 0.5278
  608/25000 [..............................] - ETA: 1:10 - loss: 7.3388 - accuracy: 0.5214
  640/25000 [..............................] - ETA: 1:09 - loss: 7.3312 - accuracy: 0.5219
  672/25000 [..............................] - ETA: 1:09 - loss: 7.3015 - accuracy: 0.5238
  704/25000 [..............................] - ETA: 1:08 - loss: 7.4270 - accuracy: 0.5156
  736/25000 [..............................] - ETA: 1:08 - loss: 7.3333 - accuracy: 0.5217
  768/25000 [..............................] - ETA: 1:07 - loss: 7.2274 - accuracy: 0.5286
  800/25000 [..............................] - ETA: 1:07 - loss: 7.2066 - accuracy: 0.5300
  832/25000 [..............................] - ETA: 1:06 - loss: 7.1690 - accuracy: 0.5325
  864/25000 [>.............................] - ETA: 1:06 - loss: 7.1520 - accuracy: 0.5336
  896/25000 [>.............................] - ETA: 1:06 - loss: 7.1190 - accuracy: 0.5357
  928/25000 [>.............................] - ETA: 1:05 - loss: 7.1875 - accuracy: 0.5312
  960/25000 [>.............................] - ETA: 1:05 - loss: 7.2194 - accuracy: 0.5292
  992/25000 [>.............................] - ETA: 1:04 - loss: 7.2184 - accuracy: 0.5292
 1024/25000 [>.............................] - ETA: 1:04 - loss: 7.2324 - accuracy: 0.5283
 1056/25000 [>.............................] - ETA: 1:04 - loss: 7.2601 - accuracy: 0.5265
 1088/25000 [>.............................] - ETA: 1:04 - loss: 7.3002 - accuracy: 0.5239
 1120/25000 [>.............................] - ETA: 1:04 - loss: 7.2559 - accuracy: 0.5268
 1152/25000 [>.............................] - ETA: 1:03 - loss: 7.2939 - accuracy: 0.5243
 1184/25000 [>.............................] - ETA: 1:03 - loss: 7.2263 - accuracy: 0.5287
 1216/25000 [>.............................] - ETA: 1:03 - loss: 7.2505 - accuracy: 0.5271
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.2489 - accuracy: 0.5272
 1280/25000 [>.............................] - ETA: 1:03 - loss: 7.2713 - accuracy: 0.5258
 1312/25000 [>.............................] - ETA: 1:03 - loss: 7.3043 - accuracy: 0.5236
 1344/25000 [>.............................] - ETA: 1:03 - loss: 7.3130 - accuracy: 0.5231
 1376/25000 [>.............................] - ETA: 1:02 - loss: 7.2655 - accuracy: 0.5262
 1408/25000 [>.............................] - ETA: 1:02 - loss: 7.2092 - accuracy: 0.5298
 1440/25000 [>.............................] - ETA: 1:02 - loss: 7.2620 - accuracy: 0.5264
 1472/25000 [>.............................] - ETA: 1:02 - loss: 7.2916 - accuracy: 0.5245
 1504/25000 [>.............................] - ETA: 1:02 - loss: 7.3098 - accuracy: 0.5233
 1536/25000 [>.............................] - ETA: 1:01 - loss: 7.3172 - accuracy: 0.5228
 1568/25000 [>.............................] - ETA: 1:01 - loss: 7.3341 - accuracy: 0.5217
 1600/25000 [>.............................] - ETA: 1:01 - loss: 7.3408 - accuracy: 0.5213
 1632/25000 [>.............................] - ETA: 1:01 - loss: 7.3002 - accuracy: 0.5239
 1664/25000 [>.............................] - ETA: 1:01 - loss: 7.3072 - accuracy: 0.5234
 1696/25000 [=>............................] - ETA: 1:00 - loss: 7.3231 - accuracy: 0.5224
 1728/25000 [=>............................] - ETA: 1:00 - loss: 7.3294 - accuracy: 0.5220
 1760/25000 [=>............................] - ETA: 1:00 - loss: 7.3530 - accuracy: 0.5205
 1792/25000 [=>............................] - ETA: 1:00 - loss: 7.3586 - accuracy: 0.5201
 1824/25000 [=>............................] - ETA: 1:00 - loss: 7.3640 - accuracy: 0.5197
 1856/25000 [=>............................] - ETA: 59s - loss: 7.3857 - accuracy: 0.5183 
 1888/25000 [=>............................] - ETA: 59s - loss: 7.3905 - accuracy: 0.5180
 1920/25000 [=>............................] - ETA: 59s - loss: 7.3871 - accuracy: 0.5182
 1952/25000 [=>............................] - ETA: 59s - loss: 7.3995 - accuracy: 0.5174
 1984/25000 [=>............................] - ETA: 59s - loss: 7.3961 - accuracy: 0.5176
 2016/25000 [=>............................] - ETA: 59s - loss: 7.4232 - accuracy: 0.5159
 2048/25000 [=>............................] - ETA: 59s - loss: 7.4046 - accuracy: 0.5171
 2080/25000 [=>............................] - ETA: 58s - loss: 7.4455 - accuracy: 0.5144
 2112/25000 [=>............................] - ETA: 58s - loss: 7.4488 - accuracy: 0.5142
 2144/25000 [=>............................] - ETA: 58s - loss: 7.4664 - accuracy: 0.5131
 2176/25000 [=>............................] - ETA: 58s - loss: 7.4975 - accuracy: 0.5110
 2208/25000 [=>............................] - ETA: 58s - loss: 7.4652 - accuracy: 0.5131
 2240/25000 [=>............................] - ETA: 58s - loss: 7.4476 - accuracy: 0.5143
 2272/25000 [=>............................] - ETA: 58s - loss: 7.4777 - accuracy: 0.5123
 2304/25000 [=>............................] - ETA: 58s - loss: 7.4803 - accuracy: 0.5122
 2336/25000 [=>............................] - ETA: 57s - loss: 7.4828 - accuracy: 0.5120
 2368/25000 [=>............................] - ETA: 57s - loss: 7.4788 - accuracy: 0.5122
 2400/25000 [=>............................] - ETA: 57s - loss: 7.4877 - accuracy: 0.5117
 2432/25000 [=>............................] - ETA: 57s - loss: 7.4838 - accuracy: 0.5119
 2464/25000 [=>............................] - ETA: 57s - loss: 7.4737 - accuracy: 0.5126
 2496/25000 [=>............................] - ETA: 57s - loss: 7.4639 - accuracy: 0.5132
 2528/25000 [==>...........................] - ETA: 57s - loss: 7.4847 - accuracy: 0.5119
 2560/25000 [==>...........................] - ETA: 57s - loss: 7.4929 - accuracy: 0.5113
 2592/25000 [==>...........................] - ETA: 56s - loss: 7.4892 - accuracy: 0.5116
 2624/25000 [==>...........................] - ETA: 56s - loss: 7.5205 - accuracy: 0.5095
 2656/25000 [==>...........................] - ETA: 56s - loss: 7.5338 - accuracy: 0.5087
 2688/25000 [==>...........................] - ETA: 56s - loss: 7.5183 - accuracy: 0.5097
 2720/25000 [==>...........................] - ETA: 56s - loss: 7.4975 - accuracy: 0.5110
 2752/25000 [==>...........................] - ETA: 56s - loss: 7.4995 - accuracy: 0.5109
 2784/25000 [==>...........................] - ETA: 56s - loss: 7.5014 - accuracy: 0.5108
 2816/25000 [==>...........................] - ETA: 56s - loss: 7.4760 - accuracy: 0.5124
 2848/25000 [==>...........................] - ETA: 56s - loss: 7.4943 - accuracy: 0.5112
 2880/25000 [==>...........................] - ETA: 56s - loss: 7.4909 - accuracy: 0.5115
 2912/25000 [==>...........................] - ETA: 56s - loss: 7.4929 - accuracy: 0.5113
 2944/25000 [==>...........................] - ETA: 55s - loss: 7.4895 - accuracy: 0.5115
 2976/25000 [==>...........................] - ETA: 55s - loss: 7.4811 - accuracy: 0.5121
 3008/25000 [==>...........................] - ETA: 55s - loss: 7.4780 - accuracy: 0.5123
 3040/25000 [==>...........................] - ETA: 55s - loss: 7.5002 - accuracy: 0.5109
 3072/25000 [==>...........................] - ETA: 55s - loss: 7.4919 - accuracy: 0.5114
 3104/25000 [==>...........................] - ETA: 55s - loss: 7.4789 - accuracy: 0.5122
 3136/25000 [==>...........................] - ETA: 55s - loss: 7.4662 - accuracy: 0.5131
 3168/25000 [==>...........................] - ETA: 55s - loss: 7.4827 - accuracy: 0.5120
 3200/25000 [==>...........................] - ETA: 54s - loss: 7.4606 - accuracy: 0.5134
 3232/25000 [==>...........................] - ETA: 54s - loss: 7.4579 - accuracy: 0.5136
 3264/25000 [==>...........................] - ETA: 54s - loss: 7.4787 - accuracy: 0.5123
 3296/25000 [==>...........................] - ETA: 54s - loss: 7.4991 - accuracy: 0.5109
 3328/25000 [==>...........................] - ETA: 54s - loss: 7.5100 - accuracy: 0.5102
 3360/25000 [===>..........................] - ETA: 54s - loss: 7.5206 - accuracy: 0.5095
 3392/25000 [===>..........................] - ETA: 54s - loss: 7.5174 - accuracy: 0.5097
 3424/25000 [===>..........................] - ETA: 54s - loss: 7.5412 - accuracy: 0.5082
 3456/25000 [===>..........................] - ETA: 54s - loss: 7.5380 - accuracy: 0.5084
 3488/25000 [===>..........................] - ETA: 54s - loss: 7.5259 - accuracy: 0.5092
 3520/25000 [===>..........................] - ETA: 54s - loss: 7.5272 - accuracy: 0.5091
 3552/25000 [===>..........................] - ETA: 53s - loss: 7.5112 - accuracy: 0.5101
 3584/25000 [===>..........................] - ETA: 53s - loss: 7.5212 - accuracy: 0.5095
 3616/25000 [===>..........................] - ETA: 53s - loss: 7.5224 - accuracy: 0.5094
 3648/25000 [===>..........................] - ETA: 53s - loss: 7.5363 - accuracy: 0.5085
 3680/25000 [===>..........................] - ETA: 53s - loss: 7.5291 - accuracy: 0.5090
 3712/25000 [===>..........................] - ETA: 53s - loss: 7.5427 - accuracy: 0.5081
 3744/25000 [===>..........................] - ETA: 53s - loss: 7.5479 - accuracy: 0.5077
 3776/25000 [===>..........................] - ETA: 53s - loss: 7.5286 - accuracy: 0.5090
 3808/25000 [===>..........................] - ETA: 53s - loss: 7.5297 - accuracy: 0.5089
 3840/25000 [===>..........................] - ETA: 53s - loss: 7.5388 - accuracy: 0.5083
 3872/25000 [===>..........................] - ETA: 53s - loss: 7.5518 - accuracy: 0.5075
 3904/25000 [===>..........................] - ETA: 53s - loss: 7.5606 - accuracy: 0.5069
 3936/25000 [===>..........................] - ETA: 52s - loss: 7.5614 - accuracy: 0.5069
 3968/25000 [===>..........................] - ETA: 52s - loss: 7.5661 - accuracy: 0.5066
 4000/25000 [===>..........................] - ETA: 52s - loss: 7.5746 - accuracy: 0.5060
 4032/25000 [===>..........................] - ETA: 52s - loss: 7.5601 - accuracy: 0.5069
 4064/25000 [===>..........................] - ETA: 52s - loss: 7.5572 - accuracy: 0.5071
 4096/25000 [===>..........................] - ETA: 52s - loss: 7.5655 - accuracy: 0.5066
 4128/25000 [===>..........................] - ETA: 52s - loss: 7.5663 - accuracy: 0.5065
 4160/25000 [===>..........................] - ETA: 52s - loss: 7.5560 - accuracy: 0.5072
 4192/25000 [====>.........................] - ETA: 52s - loss: 7.5496 - accuracy: 0.5076
 4224/25000 [====>.........................] - ETA: 52s - loss: 7.5541 - accuracy: 0.5073
 4256/25000 [====>.........................] - ETA: 51s - loss: 7.5802 - accuracy: 0.5056
 4288/25000 [====>.........................] - ETA: 51s - loss: 7.5880 - accuracy: 0.5051
 4320/25000 [====>.........................] - ETA: 51s - loss: 7.5850 - accuracy: 0.5053
 4352/25000 [====>.........................] - ETA: 51s - loss: 7.5856 - accuracy: 0.5053
 4384/25000 [====>.........................] - ETA: 51s - loss: 7.5827 - accuracy: 0.5055
 4416/25000 [====>.........................] - ETA: 51s - loss: 7.5798 - accuracy: 0.5057
 4448/25000 [====>.........................] - ETA: 51s - loss: 7.5977 - accuracy: 0.5045
 4480/25000 [====>.........................] - ETA: 51s - loss: 7.5947 - accuracy: 0.5047
 4512/25000 [====>.........................] - ETA: 51s - loss: 7.5817 - accuracy: 0.5055
 4544/25000 [====>.........................] - ETA: 51s - loss: 7.5890 - accuracy: 0.5051
 4576/25000 [====>.........................] - ETA: 51s - loss: 7.5828 - accuracy: 0.5055
 4608/25000 [====>.........................] - ETA: 51s - loss: 7.5768 - accuracy: 0.5059
 4640/25000 [====>.........................] - ETA: 51s - loss: 7.5840 - accuracy: 0.5054
 4672/25000 [====>.........................] - ETA: 50s - loss: 7.5682 - accuracy: 0.5064
 4704/25000 [====>.........................] - ETA: 50s - loss: 7.5721 - accuracy: 0.5062
 4736/25000 [====>.........................] - ETA: 50s - loss: 7.5663 - accuracy: 0.5065
 4768/25000 [====>.........................] - ETA: 50s - loss: 7.5573 - accuracy: 0.5071
 4800/25000 [====>.........................] - ETA: 50s - loss: 7.5676 - accuracy: 0.5065
 4832/25000 [====>.........................] - ETA: 50s - loss: 7.5746 - accuracy: 0.5060
 4864/25000 [====>.........................] - ETA: 50s - loss: 7.5815 - accuracy: 0.5056
 4896/25000 [====>.........................] - ETA: 50s - loss: 7.5852 - accuracy: 0.5053
 4928/25000 [====>.........................] - ETA: 50s - loss: 7.5857 - accuracy: 0.5053
 4960/25000 [====>.........................] - ETA: 50s - loss: 7.5862 - accuracy: 0.5052
 4992/25000 [====>.........................] - ETA: 50s - loss: 7.5868 - accuracy: 0.5052
 5024/25000 [=====>........................] - ETA: 49s - loss: 7.5751 - accuracy: 0.5060
 5056/25000 [=====>........................] - ETA: 49s - loss: 7.5696 - accuracy: 0.5063
 5088/25000 [=====>........................] - ETA: 49s - loss: 7.5642 - accuracy: 0.5067
 5120/25000 [=====>........................] - ETA: 49s - loss: 7.5738 - accuracy: 0.5061
 5152/25000 [=====>........................] - ETA: 49s - loss: 7.5744 - accuracy: 0.5060
 5184/25000 [=====>........................] - ETA: 49s - loss: 7.5720 - accuracy: 0.5062
 5216/25000 [=====>........................] - ETA: 49s - loss: 7.5814 - accuracy: 0.5056
 5248/25000 [=====>........................] - ETA: 49s - loss: 7.5760 - accuracy: 0.5059
 5280/25000 [=====>........................] - ETA: 49s - loss: 7.5795 - accuracy: 0.5057
 5312/25000 [=====>........................] - ETA: 49s - loss: 7.5743 - accuracy: 0.5060
 5344/25000 [=====>........................] - ETA: 49s - loss: 7.5748 - accuracy: 0.5060
 5376/25000 [=====>........................] - ETA: 48s - loss: 7.5782 - accuracy: 0.5058
 5408/25000 [=====>........................] - ETA: 48s - loss: 7.5702 - accuracy: 0.5063
 5440/25000 [=====>........................] - ETA: 48s - loss: 7.5680 - accuracy: 0.5064
 5472/25000 [=====>........................] - ETA: 48s - loss: 7.5657 - accuracy: 0.5066
 5504/25000 [=====>........................] - ETA: 48s - loss: 7.5719 - accuracy: 0.5062
 5536/25000 [=====>........................] - ETA: 48s - loss: 7.5669 - accuracy: 0.5065
 5568/25000 [=====>........................] - ETA: 48s - loss: 7.5730 - accuracy: 0.5061
 5600/25000 [=====>........................] - ETA: 48s - loss: 7.5680 - accuracy: 0.5064
 5632/25000 [=====>........................] - ETA: 48s - loss: 7.5795 - accuracy: 0.5057
 5664/25000 [=====>........................] - ETA: 48s - loss: 7.5827 - accuracy: 0.5055
 5696/25000 [=====>........................] - ETA: 48s - loss: 7.5805 - accuracy: 0.5056
 5728/25000 [=====>........................] - ETA: 48s - loss: 7.5729 - accuracy: 0.5061
 5760/25000 [=====>........................] - ETA: 47s - loss: 7.5708 - accuracy: 0.5063
 5792/25000 [=====>........................] - ETA: 47s - loss: 7.5634 - accuracy: 0.5067
 5824/25000 [=====>........................] - ETA: 47s - loss: 7.5534 - accuracy: 0.5074
 5856/25000 [======>.......................] - ETA: 47s - loss: 7.5593 - accuracy: 0.5070
 5888/25000 [======>.......................] - ETA: 47s - loss: 7.5651 - accuracy: 0.5066
 5920/25000 [======>.......................] - ETA: 47s - loss: 7.5682 - accuracy: 0.5064
 5952/25000 [======>.......................] - ETA: 47s - loss: 7.5687 - accuracy: 0.5064
 5984/25000 [======>.......................] - ETA: 47s - loss: 7.5692 - accuracy: 0.5064
 6016/25000 [======>.......................] - ETA: 47s - loss: 7.5749 - accuracy: 0.5060
 6048/25000 [======>.......................] - ETA: 47s - loss: 7.5652 - accuracy: 0.5066
 6080/25000 [======>.......................] - ETA: 47s - loss: 7.5758 - accuracy: 0.5059
 6112/25000 [======>.......................] - ETA: 47s - loss: 7.5888 - accuracy: 0.5051
 6144/25000 [======>.......................] - ETA: 46s - loss: 7.5992 - accuracy: 0.5044
 6176/25000 [======>.......................] - ETA: 46s - loss: 7.5971 - accuracy: 0.5045
 6208/25000 [======>.......................] - ETA: 46s - loss: 7.5975 - accuracy: 0.5045
 6240/25000 [======>.......................] - ETA: 46s - loss: 7.5855 - accuracy: 0.5053
 6272/25000 [======>.......................] - ETA: 46s - loss: 7.5859 - accuracy: 0.5053
 6304/25000 [======>.......................] - ETA: 46s - loss: 7.5839 - accuracy: 0.5054
 6336/25000 [======>.......................] - ETA: 46s - loss: 7.5964 - accuracy: 0.5046
 6368/25000 [======>.......................] - ETA: 46s - loss: 7.5992 - accuracy: 0.5044
 6400/25000 [======>.......................] - ETA: 46s - loss: 7.5971 - accuracy: 0.5045
 6432/25000 [======>.......................] - ETA: 46s - loss: 7.6023 - accuracy: 0.5042
 6464/25000 [======>.......................] - ETA: 46s - loss: 7.5955 - accuracy: 0.5046
 6496/25000 [======>.......................] - ETA: 46s - loss: 7.6005 - accuracy: 0.5043
 6528/25000 [======>.......................] - ETA: 45s - loss: 7.6009 - accuracy: 0.5043
 6560/25000 [======>.......................] - ETA: 45s - loss: 7.6058 - accuracy: 0.5040
 6592/25000 [======>.......................] - ETA: 45s - loss: 7.6038 - accuracy: 0.5041
 6624/25000 [======>.......................] - ETA: 45s - loss: 7.6064 - accuracy: 0.5039
 6656/25000 [======>.......................] - ETA: 45s - loss: 7.6044 - accuracy: 0.5041
 6688/25000 [=======>......................] - ETA: 45s - loss: 7.6047 - accuracy: 0.5040
 6720/25000 [=======>......................] - ETA: 45s - loss: 7.6027 - accuracy: 0.5042
 6752/25000 [=======>......................] - ETA: 45s - loss: 7.6030 - accuracy: 0.5041
 6784/25000 [=======>......................] - ETA: 45s - loss: 7.6033 - accuracy: 0.5041
 6816/25000 [=======>......................] - ETA: 45s - loss: 7.5991 - accuracy: 0.5044
 6848/25000 [=======>......................] - ETA: 45s - loss: 7.5860 - accuracy: 0.5053
 6880/25000 [=======>......................] - ETA: 45s - loss: 7.5819 - accuracy: 0.5055
 6912/25000 [=======>......................] - ETA: 44s - loss: 7.5757 - accuracy: 0.5059
 6944/25000 [=======>......................] - ETA: 44s - loss: 7.5761 - accuracy: 0.5059
 6976/25000 [=======>......................] - ETA: 44s - loss: 7.5809 - accuracy: 0.5056
 7008/25000 [=======>......................] - ETA: 44s - loss: 7.5879 - accuracy: 0.5051
 7040/25000 [=======>......................] - ETA: 44s - loss: 7.5969 - accuracy: 0.5045
 7072/25000 [=======>......................] - ETA: 44s - loss: 7.5951 - accuracy: 0.5047
 7104/25000 [=======>......................] - ETA: 44s - loss: 7.5954 - accuracy: 0.5046
 7136/25000 [=======>......................] - ETA: 44s - loss: 7.5936 - accuracy: 0.5048
 7168/25000 [=======>......................] - ETA: 44s - loss: 7.5853 - accuracy: 0.5053
 7200/25000 [=======>......................] - ETA: 44s - loss: 7.5814 - accuracy: 0.5056
 7232/25000 [=======>......................] - ETA: 44s - loss: 7.5818 - accuracy: 0.5055
 7264/25000 [=======>......................] - ETA: 43s - loss: 7.5759 - accuracy: 0.5059
 7296/25000 [=======>......................] - ETA: 43s - loss: 7.5784 - accuracy: 0.5058
 7328/25000 [=======>......................] - ETA: 43s - loss: 7.5683 - accuracy: 0.5064
 7360/25000 [=======>......................] - ETA: 43s - loss: 7.5625 - accuracy: 0.5068
 7392/25000 [=======>......................] - ETA: 43s - loss: 7.5546 - accuracy: 0.5073
 7424/25000 [=======>......................] - ETA: 43s - loss: 7.5654 - accuracy: 0.5066
 7456/25000 [=======>......................] - ETA: 43s - loss: 7.5700 - accuracy: 0.5063
 7488/25000 [=======>......................] - ETA: 43s - loss: 7.5786 - accuracy: 0.5057
 7520/25000 [========>.....................] - ETA: 43s - loss: 7.5789 - accuracy: 0.5057
 7552/25000 [========>.....................] - ETA: 43s - loss: 7.5773 - accuracy: 0.5058
 7584/25000 [========>.....................] - ETA: 43s - loss: 7.5837 - accuracy: 0.5054
 7616/25000 [========>.....................] - ETA: 43s - loss: 7.5962 - accuracy: 0.5046
 7648/25000 [========>.....................] - ETA: 42s - loss: 7.5964 - accuracy: 0.5046
 7680/25000 [========>.....................] - ETA: 42s - loss: 7.6047 - accuracy: 0.5040
 7712/25000 [========>.....................] - ETA: 42s - loss: 7.5931 - accuracy: 0.5048
 7744/25000 [========>.....................] - ETA: 42s - loss: 7.5953 - accuracy: 0.5046
 7776/25000 [========>.....................] - ETA: 42s - loss: 7.6015 - accuracy: 0.5042
 7808/25000 [========>.....................] - ETA: 42s - loss: 7.6077 - accuracy: 0.5038
 7840/25000 [========>.....................] - ETA: 42s - loss: 7.6099 - accuracy: 0.5037
 7872/25000 [========>.....................] - ETA: 42s - loss: 7.6082 - accuracy: 0.5038
 7904/25000 [========>.....................] - ETA: 42s - loss: 7.6084 - accuracy: 0.5038
 7936/25000 [========>.....................] - ETA: 42s - loss: 7.6048 - accuracy: 0.5040
 7968/25000 [========>.....................] - ETA: 42s - loss: 7.5993 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 42s - loss: 7.5880 - accuracy: 0.5051
 8032/25000 [========>.....................] - ETA: 41s - loss: 7.5960 - accuracy: 0.5046
 8064/25000 [========>.....................] - ETA: 41s - loss: 7.5963 - accuracy: 0.5046
 8096/25000 [========>.....................] - ETA: 41s - loss: 7.6060 - accuracy: 0.5040
 8128/25000 [========>.....................] - ETA: 41s - loss: 7.6025 - accuracy: 0.5042
 8160/25000 [========>.....................] - ETA: 41s - loss: 7.6102 - accuracy: 0.5037
 8192/25000 [========>.....................] - ETA: 41s - loss: 7.5974 - accuracy: 0.5045
 8224/25000 [========>.....................] - ETA: 41s - loss: 7.5958 - accuracy: 0.5046
 8256/25000 [========>.....................] - ETA: 41s - loss: 7.5979 - accuracy: 0.5045
 8288/25000 [========>.....................] - ETA: 41s - loss: 7.6056 - accuracy: 0.5040
 8320/25000 [========>.....................] - ETA: 41s - loss: 7.6169 - accuracy: 0.5032
 8352/25000 [=========>....................] - ETA: 41s - loss: 7.6207 - accuracy: 0.5030
 8384/25000 [=========>....................] - ETA: 41s - loss: 7.6209 - accuracy: 0.5030
 8416/25000 [=========>....................] - ETA: 41s - loss: 7.6284 - accuracy: 0.5025
 8448/25000 [=========>....................] - ETA: 40s - loss: 7.6321 - accuracy: 0.5022
 8480/25000 [=========>....................] - ETA: 40s - loss: 7.6250 - accuracy: 0.5027
 8512/25000 [=========>....................] - ETA: 40s - loss: 7.6342 - accuracy: 0.5021
 8544/25000 [=========>....................] - ETA: 40s - loss: 7.6343 - accuracy: 0.5021
 8576/25000 [=========>....................] - ETA: 40s - loss: 7.6434 - accuracy: 0.5015
 8608/25000 [=========>....................] - ETA: 40s - loss: 7.6381 - accuracy: 0.5019
 8640/25000 [=========>....................] - ETA: 40s - loss: 7.6382 - accuracy: 0.5019
 8672/25000 [=========>....................] - ETA: 40s - loss: 7.6224 - accuracy: 0.5029
 8704/25000 [=========>....................] - ETA: 40s - loss: 7.6296 - accuracy: 0.5024
 8736/25000 [=========>....................] - ETA: 40s - loss: 7.6280 - accuracy: 0.5025
 8768/25000 [=========>....................] - ETA: 40s - loss: 7.6264 - accuracy: 0.5026
 8800/25000 [=========>....................] - ETA: 40s - loss: 7.6283 - accuracy: 0.5025
 8832/25000 [=========>....................] - ETA: 40s - loss: 7.6250 - accuracy: 0.5027
 8864/25000 [=========>....................] - ETA: 39s - loss: 7.6251 - accuracy: 0.5027
 8896/25000 [=========>....................] - ETA: 39s - loss: 7.6235 - accuracy: 0.5028
 8928/25000 [=========>....................] - ETA: 39s - loss: 7.6254 - accuracy: 0.5027
 8960/25000 [=========>....................] - ETA: 39s - loss: 7.6221 - accuracy: 0.5029
 8992/25000 [=========>....................] - ETA: 39s - loss: 7.6308 - accuracy: 0.5023
 9024/25000 [=========>....................] - ETA: 39s - loss: 7.6292 - accuracy: 0.5024
 9056/25000 [=========>....................] - ETA: 39s - loss: 7.6311 - accuracy: 0.5023
 9088/25000 [=========>....................] - ETA: 39s - loss: 7.6346 - accuracy: 0.5021
 9120/25000 [=========>....................] - ETA: 39s - loss: 7.6313 - accuracy: 0.5023
 9152/25000 [=========>....................] - ETA: 39s - loss: 7.6365 - accuracy: 0.5020
 9184/25000 [==========>...................] - ETA: 39s - loss: 7.6349 - accuracy: 0.5021
 9216/25000 [==========>...................] - ETA: 39s - loss: 7.6333 - accuracy: 0.5022
 9248/25000 [==========>...................] - ETA: 38s - loss: 7.6351 - accuracy: 0.5021
 9280/25000 [==========>...................] - ETA: 38s - loss: 7.6385 - accuracy: 0.5018
 9312/25000 [==========>...................] - ETA: 38s - loss: 7.6353 - accuracy: 0.5020
 9344/25000 [==========>...................] - ETA: 38s - loss: 7.6387 - accuracy: 0.5018
 9376/25000 [==========>...................] - ETA: 38s - loss: 7.6421 - accuracy: 0.5016
 9408/25000 [==========>...................] - ETA: 38s - loss: 7.6422 - accuracy: 0.5016
 9440/25000 [==========>...................] - ETA: 38s - loss: 7.6374 - accuracy: 0.5019
 9472/25000 [==========>...................] - ETA: 38s - loss: 7.6375 - accuracy: 0.5019
 9504/25000 [==========>...................] - ETA: 38s - loss: 7.6408 - accuracy: 0.5017
 9536/25000 [==========>...................] - ETA: 38s - loss: 7.6393 - accuracy: 0.5018
 9568/25000 [==========>...................] - ETA: 38s - loss: 7.6394 - accuracy: 0.5018
 9600/25000 [==========>...................] - ETA: 38s - loss: 7.6395 - accuracy: 0.5018
 9632/25000 [==========>...................] - ETA: 38s - loss: 7.6380 - accuracy: 0.5019
 9664/25000 [==========>...................] - ETA: 37s - loss: 7.6333 - accuracy: 0.5022
 9696/25000 [==========>...................] - ETA: 37s - loss: 7.6366 - accuracy: 0.5020
 9728/25000 [==========>...................] - ETA: 37s - loss: 7.6367 - accuracy: 0.5020
 9760/25000 [==========>...................] - ETA: 37s - loss: 7.6431 - accuracy: 0.5015
 9792/25000 [==========>...................] - ETA: 37s - loss: 7.6416 - accuracy: 0.5016
 9824/25000 [==========>...................] - ETA: 37s - loss: 7.6401 - accuracy: 0.5017
 9856/25000 [==========>...................] - ETA: 37s - loss: 7.6355 - accuracy: 0.5020
 9888/25000 [==========>...................] - ETA: 37s - loss: 7.6356 - accuracy: 0.5020
 9920/25000 [==========>...................] - ETA: 37s - loss: 7.6419 - accuracy: 0.5016
 9952/25000 [==========>...................] - ETA: 37s - loss: 7.6435 - accuracy: 0.5015
 9984/25000 [==========>...................] - ETA: 37s - loss: 7.6482 - accuracy: 0.5012
10016/25000 [===========>..................] - ETA: 37s - loss: 7.6559 - accuracy: 0.5007
10048/25000 [===========>..................] - ETA: 37s - loss: 7.6590 - accuracy: 0.5005
10080/25000 [===========>..................] - ETA: 36s - loss: 7.6575 - accuracy: 0.5006
10112/25000 [===========>..................] - ETA: 36s - loss: 7.6545 - accuracy: 0.5008
10144/25000 [===========>..................] - ETA: 36s - loss: 7.6576 - accuracy: 0.5006
10176/25000 [===========>..................] - ETA: 36s - loss: 7.6561 - accuracy: 0.5007
10208/25000 [===========>..................] - ETA: 36s - loss: 7.6606 - accuracy: 0.5004
10240/25000 [===========>..................] - ETA: 36s - loss: 7.6591 - accuracy: 0.5005
10272/25000 [===========>..................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10304/25000 [===========>..................] - ETA: 36s - loss: 7.6755 - accuracy: 0.4994
10336/25000 [===========>..................] - ETA: 36s - loss: 7.6800 - accuracy: 0.4991
10368/25000 [===========>..................] - ETA: 36s - loss: 7.6888 - accuracy: 0.4986
10400/25000 [===========>..................] - ETA: 36s - loss: 7.6887 - accuracy: 0.4986
10432/25000 [===========>..................] - ETA: 36s - loss: 7.6945 - accuracy: 0.4982
10464/25000 [===========>..................] - ETA: 35s - loss: 7.6974 - accuracy: 0.4980
10496/25000 [===========>..................] - ETA: 35s - loss: 7.7017 - accuracy: 0.4977
10528/25000 [===========>..................] - ETA: 35s - loss: 7.6899 - accuracy: 0.4985
10560/25000 [===========>..................] - ETA: 35s - loss: 7.6899 - accuracy: 0.4985
10592/25000 [===========>..................] - ETA: 35s - loss: 7.6927 - accuracy: 0.4983
10624/25000 [===========>..................] - ETA: 35s - loss: 7.6883 - accuracy: 0.4986
10656/25000 [===========>..................] - ETA: 35s - loss: 7.6911 - accuracy: 0.4984
10688/25000 [===========>..................] - ETA: 35s - loss: 7.7011 - accuracy: 0.4978
10720/25000 [===========>..................] - ETA: 35s - loss: 7.6952 - accuracy: 0.4981
10752/25000 [===========>..................] - ETA: 35s - loss: 7.6937 - accuracy: 0.4982
10784/25000 [===========>..................] - ETA: 35s - loss: 7.6879 - accuracy: 0.4986
10816/25000 [===========>..................] - ETA: 35s - loss: 7.6893 - accuracy: 0.4985
10848/25000 [============>.................] - ETA: 35s - loss: 7.6836 - accuracy: 0.4989
10880/25000 [============>.................] - ETA: 34s - loss: 7.6821 - accuracy: 0.4990
10912/25000 [============>.................] - ETA: 34s - loss: 7.6849 - accuracy: 0.4988
10944/25000 [============>.................] - ETA: 34s - loss: 7.6778 - accuracy: 0.4993
10976/25000 [============>.................] - ETA: 34s - loss: 7.6778 - accuracy: 0.4993
11008/25000 [============>.................] - ETA: 34s - loss: 7.6778 - accuracy: 0.4993
11040/25000 [============>.................] - ETA: 34s - loss: 7.6736 - accuracy: 0.4995
11072/25000 [============>.................] - ETA: 34s - loss: 7.6722 - accuracy: 0.4996
11104/25000 [============>.................] - ETA: 34s - loss: 7.6735 - accuracy: 0.4995
11136/25000 [============>.................] - ETA: 34s - loss: 7.6776 - accuracy: 0.4993
11168/25000 [============>.................] - ETA: 34s - loss: 7.6762 - accuracy: 0.4994
11200/25000 [============>.................] - ETA: 34s - loss: 7.6789 - accuracy: 0.4992
11232/25000 [============>.................] - ETA: 34s - loss: 7.6803 - accuracy: 0.4991
11264/25000 [============>.................] - ETA: 33s - loss: 7.6748 - accuracy: 0.4995
11296/25000 [============>.................] - ETA: 33s - loss: 7.6734 - accuracy: 0.4996
11328/25000 [============>.................] - ETA: 33s - loss: 7.6747 - accuracy: 0.4995
11360/25000 [============>.................] - ETA: 33s - loss: 7.6707 - accuracy: 0.4997
11392/25000 [============>.................] - ETA: 33s - loss: 7.6680 - accuracy: 0.4999
11424/25000 [============>.................] - ETA: 33s - loss: 7.6706 - accuracy: 0.4997
11456/25000 [============>.................] - ETA: 33s - loss: 7.6693 - accuracy: 0.4998
11488/25000 [============>.................] - ETA: 33s - loss: 7.6626 - accuracy: 0.5003
11520/25000 [============>.................] - ETA: 33s - loss: 7.6626 - accuracy: 0.5003
11552/25000 [============>.................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
11584/25000 [============>.................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
11616/25000 [============>.................] - ETA: 33s - loss: 7.6640 - accuracy: 0.5002
11648/25000 [============>.................] - ETA: 33s - loss: 7.6627 - accuracy: 0.5003
11680/25000 [=============>................] - ETA: 32s - loss: 7.6561 - accuracy: 0.5007
11712/25000 [=============>................] - ETA: 32s - loss: 7.6575 - accuracy: 0.5006
11744/25000 [=============>................] - ETA: 32s - loss: 7.6562 - accuracy: 0.5007
11776/25000 [=============>................] - ETA: 32s - loss: 7.6562 - accuracy: 0.5007
11808/25000 [=============>................] - ETA: 32s - loss: 7.6588 - accuracy: 0.5005
11840/25000 [=============>................] - ETA: 32s - loss: 7.6601 - accuracy: 0.5004
11872/25000 [=============>................] - ETA: 32s - loss: 7.6602 - accuracy: 0.5004
11904/25000 [=============>................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
11936/25000 [=============>................] - ETA: 32s - loss: 7.6718 - accuracy: 0.4997
11968/25000 [=============>................] - ETA: 32s - loss: 7.6782 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 32s - loss: 7.6730 - accuracy: 0.4996
12032/25000 [=============>................] - ETA: 32s - loss: 7.6704 - accuracy: 0.4998
12064/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
12096/25000 [=============>................] - ETA: 31s - loss: 7.6692 - accuracy: 0.4998
12128/25000 [=============>................] - ETA: 31s - loss: 7.6729 - accuracy: 0.4996
12160/25000 [=============>................] - ETA: 31s - loss: 7.6691 - accuracy: 0.4998
12192/25000 [=============>................] - ETA: 31s - loss: 7.6628 - accuracy: 0.5002
12224/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
12256/25000 [=============>................] - ETA: 31s - loss: 7.6691 - accuracy: 0.4998
12288/25000 [=============>................] - ETA: 31s - loss: 7.6691 - accuracy: 0.4998
12320/25000 [=============>................] - ETA: 31s - loss: 7.6629 - accuracy: 0.5002
12352/25000 [=============>................] - ETA: 31s - loss: 7.6592 - accuracy: 0.5005
12384/25000 [=============>................] - ETA: 31s - loss: 7.6567 - accuracy: 0.5006
12416/25000 [=============>................] - ETA: 31s - loss: 7.6530 - accuracy: 0.5009
12448/25000 [=============>................] - ETA: 31s - loss: 7.6555 - accuracy: 0.5007
12480/25000 [=============>................] - ETA: 30s - loss: 7.6556 - accuracy: 0.5007
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6531 - accuracy: 0.5009
12544/25000 [==============>...............] - ETA: 30s - loss: 7.6532 - accuracy: 0.5009
12576/25000 [==============>...............] - ETA: 30s - loss: 7.6496 - accuracy: 0.5011
12608/25000 [==============>...............] - ETA: 30s - loss: 7.6508 - accuracy: 0.5010
12640/25000 [==============>...............] - ETA: 30s - loss: 7.6521 - accuracy: 0.5009
12672/25000 [==============>...............] - ETA: 30s - loss: 7.6581 - accuracy: 0.5006
12704/25000 [==============>...............] - ETA: 30s - loss: 7.6606 - accuracy: 0.5004
12736/25000 [==============>...............] - ETA: 30s - loss: 7.6594 - accuracy: 0.5005
12768/25000 [==============>...............] - ETA: 30s - loss: 7.6582 - accuracy: 0.5005
12800/25000 [==============>...............] - ETA: 30s - loss: 7.6618 - accuracy: 0.5003
12832/25000 [==============>...............] - ETA: 30s - loss: 7.6618 - accuracy: 0.5003
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6583 - accuracy: 0.5005
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6595 - accuracy: 0.5005
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6559 - accuracy: 0.5007
12960/25000 [==============>...............] - ETA: 29s - loss: 7.6548 - accuracy: 0.5008
12992/25000 [==============>...............] - ETA: 29s - loss: 7.6525 - accuracy: 0.5009
13024/25000 [==============>...............] - ETA: 29s - loss: 7.6560 - accuracy: 0.5007
13056/25000 [==============>...............] - ETA: 29s - loss: 7.6572 - accuracy: 0.5006
13088/25000 [==============>...............] - ETA: 29s - loss: 7.6631 - accuracy: 0.5002
13120/25000 [==============>...............] - ETA: 29s - loss: 7.6619 - accuracy: 0.5003
13152/25000 [==============>...............] - ETA: 29s - loss: 7.6608 - accuracy: 0.5004
13184/25000 [==============>...............] - ETA: 29s - loss: 7.6573 - accuracy: 0.5006
13216/25000 [==============>...............] - ETA: 29s - loss: 7.6515 - accuracy: 0.5010
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6516 - accuracy: 0.5010
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6505 - accuracy: 0.5011
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6551 - accuracy: 0.5008
13344/25000 [===============>..............] - ETA: 28s - loss: 7.6517 - accuracy: 0.5010
13376/25000 [===============>..............] - ETA: 28s - loss: 7.6506 - accuracy: 0.5010
13408/25000 [===============>..............] - ETA: 28s - loss: 7.6483 - accuracy: 0.5012
13440/25000 [===============>..............] - ETA: 28s - loss: 7.6495 - accuracy: 0.5011
13472/25000 [===============>..............] - ETA: 28s - loss: 7.6507 - accuracy: 0.5010
13504/25000 [===============>..............] - ETA: 28s - loss: 7.6507 - accuracy: 0.5010
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6496 - accuracy: 0.5011
13568/25000 [===============>..............] - ETA: 28s - loss: 7.6519 - accuracy: 0.5010
13600/25000 [===============>..............] - ETA: 28s - loss: 7.6497 - accuracy: 0.5011
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6464 - accuracy: 0.5013
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6464 - accuracy: 0.5013
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6532 - accuracy: 0.5009
13728/25000 [===============>..............] - ETA: 27s - loss: 7.6521 - accuracy: 0.5009
13760/25000 [===============>..............] - ETA: 27s - loss: 7.6521 - accuracy: 0.5009
13792/25000 [===============>..............] - ETA: 27s - loss: 7.6488 - accuracy: 0.5012
13824/25000 [===============>..............] - ETA: 27s - loss: 7.6478 - accuracy: 0.5012
13856/25000 [===============>..............] - ETA: 27s - loss: 7.6467 - accuracy: 0.5013
13888/25000 [===============>..............] - ETA: 27s - loss: 7.6467 - accuracy: 0.5013
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6468 - accuracy: 0.5013
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6479 - accuracy: 0.5012
13984/25000 [===============>..............] - ETA: 27s - loss: 7.6480 - accuracy: 0.5012
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6436 - accuracy: 0.5015
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6437 - accuracy: 0.5015
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6427 - accuracy: 0.5016
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6416 - accuracy: 0.5016
14144/25000 [===============>..............] - ETA: 26s - loss: 7.6417 - accuracy: 0.5016
14176/25000 [================>.............] - ETA: 26s - loss: 7.6439 - accuracy: 0.5015
14208/25000 [================>.............] - ETA: 26s - loss: 7.6472 - accuracy: 0.5013
14240/25000 [================>.............] - ETA: 26s - loss: 7.6483 - accuracy: 0.5012
14272/25000 [================>.............] - ETA: 26s - loss: 7.6505 - accuracy: 0.5011
14304/25000 [================>.............] - ETA: 26s - loss: 7.6505 - accuracy: 0.5010
14336/25000 [================>.............] - ETA: 26s - loss: 7.6559 - accuracy: 0.5007
14368/25000 [================>.............] - ETA: 26s - loss: 7.6624 - accuracy: 0.5003
14400/25000 [================>.............] - ETA: 26s - loss: 7.6581 - accuracy: 0.5006
14432/25000 [================>.............] - ETA: 26s - loss: 7.6528 - accuracy: 0.5009
14464/25000 [================>.............] - ETA: 26s - loss: 7.6613 - accuracy: 0.5003
14496/25000 [================>.............] - ETA: 25s - loss: 7.6624 - accuracy: 0.5003
14528/25000 [================>.............] - ETA: 25s - loss: 7.6635 - accuracy: 0.5002
14560/25000 [================>.............] - ETA: 25s - loss: 7.6582 - accuracy: 0.5005
14592/25000 [================>.............] - ETA: 25s - loss: 7.6561 - accuracy: 0.5007
14624/25000 [================>.............] - ETA: 25s - loss: 7.6582 - accuracy: 0.5005
14656/25000 [================>.............] - ETA: 25s - loss: 7.6603 - accuracy: 0.5004
14688/25000 [================>.............] - ETA: 25s - loss: 7.6593 - accuracy: 0.5005
14720/25000 [================>.............] - ETA: 25s - loss: 7.6583 - accuracy: 0.5005
14752/25000 [================>.............] - ETA: 25s - loss: 7.6573 - accuracy: 0.5006
14784/25000 [================>.............] - ETA: 25s - loss: 7.6562 - accuracy: 0.5007
14816/25000 [================>.............] - ETA: 25s - loss: 7.6532 - accuracy: 0.5009
14848/25000 [================>.............] - ETA: 25s - loss: 7.6522 - accuracy: 0.5009
14880/25000 [================>.............] - ETA: 24s - loss: 7.6522 - accuracy: 0.5009
14912/25000 [================>.............] - ETA: 24s - loss: 7.6512 - accuracy: 0.5010
14944/25000 [================>.............] - ETA: 24s - loss: 7.6440 - accuracy: 0.5015
14976/25000 [================>.............] - ETA: 24s - loss: 7.6461 - accuracy: 0.5013
15008/25000 [=================>............] - ETA: 24s - loss: 7.6482 - accuracy: 0.5012
15040/25000 [=================>............] - ETA: 24s - loss: 7.6462 - accuracy: 0.5013
15072/25000 [=================>............] - ETA: 24s - loss: 7.6432 - accuracy: 0.5015
15104/25000 [=================>............] - ETA: 24s - loss: 7.6392 - accuracy: 0.5018
15136/25000 [=================>............] - ETA: 24s - loss: 7.6433 - accuracy: 0.5015
15168/25000 [=================>............] - ETA: 24s - loss: 7.6434 - accuracy: 0.5015
15200/25000 [=================>............] - ETA: 24s - loss: 7.6454 - accuracy: 0.5014
15232/25000 [=================>............] - ETA: 24s - loss: 7.6485 - accuracy: 0.5012
15264/25000 [=================>............] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
15296/25000 [=================>............] - ETA: 23s - loss: 7.6516 - accuracy: 0.5010
15328/25000 [=================>............] - ETA: 23s - loss: 7.6496 - accuracy: 0.5011
15360/25000 [=================>............] - ETA: 23s - loss: 7.6516 - accuracy: 0.5010
15392/25000 [=================>............] - ETA: 23s - loss: 7.6596 - accuracy: 0.5005
15424/25000 [=================>............] - ETA: 23s - loss: 7.6616 - accuracy: 0.5003
15456/25000 [=================>............] - ETA: 23s - loss: 7.6617 - accuracy: 0.5003
15488/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15520/25000 [=================>............] - ETA: 23s - loss: 7.6637 - accuracy: 0.5002
15552/25000 [=================>............] - ETA: 23s - loss: 7.6686 - accuracy: 0.4999
15584/25000 [=================>............] - ETA: 23s - loss: 7.6745 - accuracy: 0.4995
15616/25000 [=================>............] - ETA: 23s - loss: 7.6745 - accuracy: 0.4995
15648/25000 [=================>............] - ETA: 23s - loss: 7.6725 - accuracy: 0.4996
15680/25000 [=================>............] - ETA: 23s - loss: 7.6735 - accuracy: 0.4996
15712/25000 [=================>............] - ETA: 22s - loss: 7.6803 - accuracy: 0.4991
15744/25000 [=================>............] - ETA: 22s - loss: 7.6793 - accuracy: 0.4992
15776/25000 [=================>............] - ETA: 22s - loss: 7.6763 - accuracy: 0.4994
15808/25000 [=================>............] - ETA: 22s - loss: 7.6802 - accuracy: 0.4991
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6773 - accuracy: 0.4993
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6763 - accuracy: 0.4994
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6792 - accuracy: 0.4992
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6839 - accuracy: 0.4989
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6810 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6839 - accuracy: 0.4989
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6857 - accuracy: 0.4988
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6828 - accuracy: 0.4989
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6838 - accuracy: 0.4989
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6828 - accuracy: 0.4989
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6846 - accuracy: 0.4988
16192/25000 [==================>...........] - ETA: 21s - loss: 7.6846 - accuracy: 0.4988
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6893 - accuracy: 0.4985
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6883 - accuracy: 0.4986
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6911 - accuracy: 0.4984
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6901 - accuracy: 0.4985
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6919 - accuracy: 0.4983
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6919 - accuracy: 0.4984
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6918 - accuracy: 0.4984
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6946 - accuracy: 0.4982
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6917 - accuracy: 0.4984
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6908 - accuracy: 0.4984
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6870 - accuracy: 0.4987
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6888 - accuracy: 0.4986
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6832 - accuracy: 0.4989
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6841 - accuracy: 0.4989
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6905 - accuracy: 0.4984
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6905 - accuracy: 0.4984
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6868 - accuracy: 0.4987
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6858 - accuracy: 0.4987
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6840 - accuracy: 0.4989
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6848 - accuracy: 0.4988
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6903 - accuracy: 0.4985
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6893 - accuracy: 0.4985
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6884 - accuracy: 0.4986
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6901 - accuracy: 0.4985
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6901 - accuracy: 0.4985
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6855 - accuracy: 0.4988
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6873 - accuracy: 0.4987
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6935 - accuracy: 0.4982
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6881 - accuracy: 0.4986
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6881 - accuracy: 0.4986
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6880 - accuracy: 0.4986
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6844 - accuracy: 0.4988
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6844 - accuracy: 0.4988
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6861 - accuracy: 0.4987
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6905 - accuracy: 0.4984
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6931 - accuracy: 0.4983
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6957 - accuracy: 0.4981
17408/25000 [===================>..........] - ETA: 18s - loss: 7.7001 - accuracy: 0.4978
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6956 - accuracy: 0.4981
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6965 - accuracy: 0.4981
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6973 - accuracy: 0.4980
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6963 - accuracy: 0.4981
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6954 - accuracy: 0.4981
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6971 - accuracy: 0.4980
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6979 - accuracy: 0.4980
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6996 - accuracy: 0.4978
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6987 - accuracy: 0.4979
17728/25000 [====================>.........] - ETA: 17s - loss: 7.7004 - accuracy: 0.4978
17760/25000 [====================>.........] - ETA: 17s - loss: 7.7020 - accuracy: 0.4977
17792/25000 [====================>.........] - ETA: 17s - loss: 7.7020 - accuracy: 0.4977
17824/25000 [====================>.........] - ETA: 17s - loss: 7.7045 - accuracy: 0.4975
17856/25000 [====================>.........] - ETA: 17s - loss: 7.7010 - accuracy: 0.4978
17888/25000 [====================>.........] - ETA: 17s - loss: 7.7018 - accuracy: 0.4977
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6966 - accuracy: 0.4980
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6974 - accuracy: 0.4980
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6973 - accuracy: 0.4980
18016/25000 [====================>.........] - ETA: 17s - loss: 7.7024 - accuracy: 0.4977
18048/25000 [====================>.........] - ETA: 17s - loss: 7.7006 - accuracy: 0.4978
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6946 - accuracy: 0.4982
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6920 - accuracy: 0.4983
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6928 - accuracy: 0.4983
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6877 - accuracy: 0.4986
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6885 - accuracy: 0.4986
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6876 - accuracy: 0.4986
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6868 - accuracy: 0.4987
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6867 - accuracy: 0.4987
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6842 - accuracy: 0.4989
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6841 - accuracy: 0.4989
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6841 - accuracy: 0.4989
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6824 - accuracy: 0.4990
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6841 - accuracy: 0.4989
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6799 - accuracy: 0.4991
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6782 - accuracy: 0.4992
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6774 - accuracy: 0.4993
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6798 - accuracy: 0.4991
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6773 - accuracy: 0.4993
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6781 - accuracy: 0.4992
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6765 - accuracy: 0.4994
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6781 - accuracy: 0.4993
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6781 - accuracy: 0.4993
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6756 - accuracy: 0.4994
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6756 - accuracy: 0.4994
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6731 - accuracy: 0.4996
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6739 - accuracy: 0.4995
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6707 - accuracy: 0.4997
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6690 - accuracy: 0.4998
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6690 - accuracy: 0.4998
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6723 - accuracy: 0.4996
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6698 - accuracy: 0.4998
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6690 - accuracy: 0.4998
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6690 - accuracy: 0.4998
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6626 - accuracy: 0.5003
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6674 - accuracy: 0.4999
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6650 - accuracy: 0.5001
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6619 - accuracy: 0.5003
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6619 - accuracy: 0.5003
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6587 - accuracy: 0.5005
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6619 - accuracy: 0.5003
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6635 - accuracy: 0.5002
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6729 - accuracy: 0.4996
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6713 - accuracy: 0.4997
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6721 - accuracy: 0.4996
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6744 - accuracy: 0.4995
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6744 - accuracy: 0.4995
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6775 - accuracy: 0.4993
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6775 - accuracy: 0.4993
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6752 - accuracy: 0.4994
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6751 - accuracy: 0.4994
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6705 - accuracy: 0.4997
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6728 - accuracy: 0.4996
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6697 - accuracy: 0.4998
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6735 - accuracy: 0.4995
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6766 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6751 - accuracy: 0.4994
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6720 - accuracy: 0.4997
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6735 - accuracy: 0.4996
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6742 - accuracy: 0.4995
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6750 - accuracy: 0.4995
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6765 - accuracy: 0.4994
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6765 - accuracy: 0.4994
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6750 - accuracy: 0.4995
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6727 - accuracy: 0.4996
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6795 - accuracy: 0.4992
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6794 - accuracy: 0.4992
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6802 - accuracy: 0.4991
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6809 - accuracy: 0.4991
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6846 - accuracy: 0.4988
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6816 - accuracy: 0.4990
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6838 - accuracy: 0.4989
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6831 - accuracy: 0.4989
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6830 - accuracy: 0.4989
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6823 - accuracy: 0.4990
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6860 - accuracy: 0.4987
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6837 - accuracy: 0.4989
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6837 - accuracy: 0.4989
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6800 - accuracy: 0.4991
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6807 - accuracy: 0.4991
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6806 - accuracy: 0.4991
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6814 - accuracy: 0.4990
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6799 - accuracy: 0.4991
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6747 - accuracy: 0.4995
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6754 - accuracy: 0.4994
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6739 - accuracy: 0.4995
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6776 - accuracy: 0.4993 
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6761 - accuracy: 0.4994
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6768 - accuracy: 0.4993
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6761 - accuracy: 0.4994
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6761 - accuracy: 0.4994
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6753 - accuracy: 0.4994
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6739 - accuracy: 0.4995
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6717 - accuracy: 0.4997
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6731 - accuracy: 0.4996
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6760 - accuracy: 0.4994
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6767 - accuracy: 0.4993
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6759 - accuracy: 0.4994
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6809 - accuracy: 0.4991
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6816 - accuracy: 0.4990
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6809 - accuracy: 0.4991
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6851 - accuracy: 0.4988
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6858 - accuracy: 0.4987
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6872 - accuracy: 0.4987
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6865 - accuracy: 0.4987
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6857 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6842 - accuracy: 0.4989
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6828 - accuracy: 0.4989
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6814 - accuracy: 0.4990
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6821 - accuracy: 0.4990
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6813 - accuracy: 0.4990
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6813 - accuracy: 0.4990
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6806 - accuracy: 0.4991
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6827 - accuracy: 0.4990
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6847 - accuracy: 0.4988
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6826 - accuracy: 0.4990
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6825 - accuracy: 0.4990
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6832 - accuracy: 0.4989
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6832 - accuracy: 0.4989
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6859 - accuracy: 0.4987
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6865 - accuracy: 0.4987
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6824 - accuracy: 0.4990
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6824 - accuracy: 0.4990
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6817 - accuracy: 0.4990
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6803 - accuracy: 0.4991
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6802 - accuracy: 0.4991
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6836 - accuracy: 0.4989
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6843 - accuracy: 0.4988
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6836 - accuracy: 0.4989
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6835 - accuracy: 0.4989
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6862 - accuracy: 0.4987
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6855 - accuracy: 0.4988
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6855 - accuracy: 0.4988
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6848 - accuracy: 0.4988
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6848 - accuracy: 0.4988
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6854 - accuracy: 0.4988
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6867 - accuracy: 0.4987
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6854 - accuracy: 0.4988
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6847 - accuracy: 0.4988
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6833 - accuracy: 0.4989
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6819 - accuracy: 0.4990
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6799 - accuracy: 0.4991
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6839 - accuracy: 0.4989
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6838 - accuracy: 0.4989
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6878 - accuracy: 0.4986
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6851 - accuracy: 0.4988
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6831 - accuracy: 0.4989
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6811 - accuracy: 0.4991
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6817 - accuracy: 0.4990
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6850 - accuracy: 0.4988
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6849 - accuracy: 0.4988
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6810 - accuracy: 0.4991
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6797 - accuracy: 0.4991
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6777 - accuracy: 0.4993
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6783 - accuracy: 0.4992
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6783 - accuracy: 0.4992
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6776 - accuracy: 0.4993
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6783 - accuracy: 0.4992
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6776 - accuracy: 0.4993
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6769 - accuracy: 0.4993
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6788 - accuracy: 0.4992
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6808 - accuracy: 0.4991
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6826 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6839 - accuracy: 0.4989
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6845 - accuracy: 0.4988
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6832 - accuracy: 0.4989
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6838 - accuracy: 0.4989
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6870 - accuracy: 0.4987
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6857 - accuracy: 0.4988
24192/25000 [============================>.] - ETA: 1s - loss: 7.6869 - accuracy: 0.4987
24224/25000 [============================>.] - ETA: 1s - loss: 7.6862 - accuracy: 0.4987
24256/25000 [============================>.] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
24288/25000 [============================>.] - ETA: 1s - loss: 7.6881 - accuracy: 0.4986
24320/25000 [============================>.] - ETA: 1s - loss: 7.6906 - accuracy: 0.4984
24352/25000 [============================>.] - ETA: 1s - loss: 7.6905 - accuracy: 0.4984
24384/25000 [============================>.] - ETA: 1s - loss: 7.6867 - accuracy: 0.4987
24416/25000 [============================>.] - ETA: 1s - loss: 7.6867 - accuracy: 0.4987
24448/25000 [============================>.] - ETA: 1s - loss: 7.6842 - accuracy: 0.4989
24480/25000 [============================>.] - ETA: 1s - loss: 7.6817 - accuracy: 0.4990
24512/25000 [============================>.] - ETA: 1s - loss: 7.6791 - accuracy: 0.4992
24544/25000 [============================>.] - ETA: 1s - loss: 7.6810 - accuracy: 0.4991
24576/25000 [============================>.] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
24608/25000 [============================>.] - ETA: 0s - loss: 7.6797 - accuracy: 0.4991
24640/25000 [============================>.] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24672/25000 [============================>.] - ETA: 0s - loss: 7.6759 - accuracy: 0.4994
24704/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24768/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24896/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 72s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

