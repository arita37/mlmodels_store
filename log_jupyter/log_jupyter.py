
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '203a72830f23a80c3dd3ee4f0d2ce62ae396cb03', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03

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
 40%|████      | 2/5 [00:55<01:23, 27.68s/it] 40%|████      | 2/5 [00:55<01:23, 27.68s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.22412132446496746, 'embedding_size_factor': 1.2071327452324718, 'layers.choice': 0, 'learning_rate': 0.0003713265864338324, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 5.298982895384028e-06} and reward: 0.3664
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcc\xb0\x01\xeft\xe5\x0cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3Pjl\xebDDX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?8U\xd3\x8b|\x82\xa7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd69\xbdS\xc4`fu.' and reward: 0.3664
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcc\xb0\x01\xeft\xe5\x0cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3Pjl\xebDDX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?8U\xd3\x8b|\x82\xa7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd69\xbdS\xc4`fu.' and reward: 0.3664
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 111.87223315238953
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the 6.36s of remaining time.
Ensemble size: 11
Ensemble weights: 
[0.72727273 0.27272727]
	0.3896	 = Validation accuracy score
	0.96s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 114.64s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7ff192834a90> 

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
 [ 0.1069865   0.01018785 -0.03477554 -0.0227038   0.11754055  0.07464302]
 [ 0.27156439  0.27042848  0.10829305 -0.02949878  0.03999641 -0.02579247]
 [-0.12271927  0.17343965 -0.06856058  0.13665791  0.08072576  0.11099711]
 [ 0.05828048  0.12171094  0.06557287  0.38901004  0.1274637  -0.17023298]
 [ 0.32728302  0.66523594 -0.20537612 -0.11239282  0.36393055  0.28684387]
 [ 0.56720281  0.20691928  0.42501375 -0.74359667  0.03168174  0.60681736]
 [ 0.21652928  0.46867618  0.49932671 -0.06514889  0.28193715  0.49405083]
 [ 0.15188682  0.27648762 -0.60381603  0.19518571  0.66824156  0.12812553]
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
{'loss': 0.6874085366725922, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 05:16:52.354842: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.49034763127565384, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 05:16:53.576862: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 2:11
   40960/17464789 [..............................] - ETA: 52s 
   90112/17464789 [..............................] - ETA: 35s
  163840/17464789 [..............................] - ETA: 26s
  319488/17464789 [..............................] - ETA: 16s
  630784/17464789 [>.............................] - ETA: 9s 
 1253376/17464789 [=>............................] - ETA: 5s
 2498560/17464789 [===>..........................] - ETA: 2s
 4956160/17464789 [=======>......................] - ETA: 1s
 7938048/17464789 [============>.................] - ETA: 0s
10870784/17464789 [=================>............] - ETA: 0s
13705216/17464789 [======================>.......] - ETA: 0s
16769024/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-18 05:17:06.630685: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 05:17:06.634377: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-18 05:17:06.635059: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a3f77ed830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 05:17:06.635078: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:30 - loss: 6.7083 - accuracy: 0.5625
   64/25000 [..............................] - ETA: 2:55 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:22 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:05 - loss: 7.6666 - accuracy: 0.5000
  160/25000 [..............................] - ETA: 1:55 - loss: 7.4750 - accuracy: 0.5125
  192/25000 [..............................] - ETA: 1:48 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:43 - loss: 7.5297 - accuracy: 0.5089
  256/25000 [..............................] - ETA: 1:39 - loss: 7.4270 - accuracy: 0.5156
  288/25000 [..............................] - ETA: 1:37 - loss: 7.5069 - accuracy: 0.5104
  320/25000 [..............................] - ETA: 1:34 - loss: 7.7145 - accuracy: 0.4969
  352/25000 [..............................] - ETA: 1:33 - loss: 7.5795 - accuracy: 0.5057
  384/25000 [..............................] - ETA: 1:31 - loss: 7.7465 - accuracy: 0.4948
  416/25000 [..............................] - ETA: 1:29 - loss: 7.6666 - accuracy: 0.5000
  448/25000 [..............................] - ETA: 1:28 - loss: 7.6324 - accuracy: 0.5022
  480/25000 [..............................] - ETA: 1:27 - loss: 7.7305 - accuracy: 0.4958
  512/25000 [..............................] - ETA: 1:27 - loss: 7.7565 - accuracy: 0.4941
  544/25000 [..............................] - ETA: 1:26 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:25 - loss: 7.6932 - accuracy: 0.4983
  608/25000 [..............................] - ETA: 1:24 - loss: 7.6414 - accuracy: 0.5016
  640/25000 [..............................] - ETA: 1:24 - loss: 7.6427 - accuracy: 0.5016
  672/25000 [..............................] - ETA: 1:23 - loss: 7.6438 - accuracy: 0.5015
  704/25000 [..............................] - ETA: 1:22 - loss: 7.5577 - accuracy: 0.5071
  736/25000 [..............................] - ETA: 1:22 - loss: 7.5208 - accuracy: 0.5095
  768/25000 [..............................] - ETA: 1:21 - loss: 7.4670 - accuracy: 0.5130
  800/25000 [..............................] - ETA: 1:21 - loss: 7.4558 - accuracy: 0.5138
  832/25000 [..............................] - ETA: 1:21 - loss: 7.4823 - accuracy: 0.5120
  864/25000 [>.............................] - ETA: 1:20 - loss: 7.5956 - accuracy: 0.5046
  896/25000 [>.............................] - ETA: 1:20 - loss: 7.6324 - accuracy: 0.5022
  928/25000 [>.............................] - ETA: 1:19 - loss: 7.6005 - accuracy: 0.5043
  960/25000 [>.............................] - ETA: 1:19 - loss: 7.6187 - accuracy: 0.5031
  992/25000 [>.............................] - ETA: 1:19 - loss: 7.5893 - accuracy: 0.5050
 1024/25000 [>.............................] - ETA: 1:19 - loss: 7.6367 - accuracy: 0.5020
 1056/25000 [>.............................] - ETA: 1:18 - loss: 7.6231 - accuracy: 0.5028
 1088/25000 [>.............................] - ETA: 1:18 - loss: 7.6243 - accuracy: 0.5028
 1120/25000 [>.............................] - ETA: 1:18 - loss: 7.6529 - accuracy: 0.5009
 1152/25000 [>.............................] - ETA: 1:17 - loss: 7.6134 - accuracy: 0.5035
 1184/25000 [>.............................] - ETA: 1:17 - loss: 7.5889 - accuracy: 0.5051
 1216/25000 [>.............................] - ETA: 1:17 - loss: 7.6288 - accuracy: 0.5025
 1248/25000 [>.............................] - ETA: 1:17 - loss: 7.6420 - accuracy: 0.5016
 1280/25000 [>.............................] - ETA: 1:17 - loss: 7.5947 - accuracy: 0.5047
 1312/25000 [>.............................] - ETA: 1:16 - loss: 7.6199 - accuracy: 0.5030
 1344/25000 [>.............................] - ETA: 1:16 - loss: 7.6096 - accuracy: 0.5037
 1376/25000 [>.............................] - ETA: 1:16 - loss: 7.6443 - accuracy: 0.5015
 1408/25000 [>.............................] - ETA: 1:16 - loss: 7.6557 - accuracy: 0.5007
 1440/25000 [>.............................] - ETA: 1:16 - loss: 7.6347 - accuracy: 0.5021
 1472/25000 [>.............................] - ETA: 1:16 - loss: 7.6354 - accuracy: 0.5020
 1504/25000 [>.............................] - ETA: 1:16 - loss: 7.6156 - accuracy: 0.5033
 1536/25000 [>.............................] - ETA: 1:16 - loss: 7.6067 - accuracy: 0.5039
 1568/25000 [>.............................] - ETA: 1:15 - loss: 7.6177 - accuracy: 0.5032
 1600/25000 [>.............................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 1632/25000 [>.............................] - ETA: 1:15 - loss: 7.6384 - accuracy: 0.5018
 1664/25000 [>.............................] - ETA: 1:15 - loss: 7.6205 - accuracy: 0.5030
 1696/25000 [=>............................] - ETA: 1:15 - loss: 7.6757 - accuracy: 0.4994
 1728/25000 [=>............................] - ETA: 1:14 - loss: 7.7199 - accuracy: 0.4965
 1760/25000 [=>............................] - ETA: 1:14 - loss: 7.7625 - accuracy: 0.4938
 1792/25000 [=>............................] - ETA: 1:14 - loss: 7.7607 - accuracy: 0.4939
 1824/25000 [=>............................] - ETA: 1:14 - loss: 7.7591 - accuracy: 0.4940
 1856/25000 [=>............................] - ETA: 1:14 - loss: 7.7905 - accuracy: 0.4919
 1888/25000 [=>............................] - ETA: 1:14 - loss: 7.7641 - accuracy: 0.4936
 1920/25000 [=>............................] - ETA: 1:13 - loss: 7.7465 - accuracy: 0.4948
 1952/25000 [=>............................] - ETA: 1:13 - loss: 7.7373 - accuracy: 0.4954
 1984/25000 [=>............................] - ETA: 1:13 - loss: 7.7671 - accuracy: 0.4934
 2016/25000 [=>............................] - ETA: 1:13 - loss: 7.7503 - accuracy: 0.4945
 2048/25000 [=>............................] - ETA: 1:13 - loss: 7.7190 - accuracy: 0.4966
 2080/25000 [=>............................] - ETA: 1:13 - loss: 7.7182 - accuracy: 0.4966
 2112/25000 [=>............................] - ETA: 1:13 - loss: 7.6957 - accuracy: 0.4981
 2144/25000 [=>............................] - ETA: 1:13 - loss: 7.7381 - accuracy: 0.4953
 2176/25000 [=>............................] - ETA: 1:12 - loss: 7.7371 - accuracy: 0.4954
 2208/25000 [=>............................] - ETA: 1:12 - loss: 7.7569 - accuracy: 0.4941
 2240/25000 [=>............................] - ETA: 1:12 - loss: 7.7693 - accuracy: 0.4933
 2272/25000 [=>............................] - ETA: 1:12 - loss: 7.7544 - accuracy: 0.4943
 2304/25000 [=>............................] - ETA: 1:12 - loss: 7.7398 - accuracy: 0.4952
 2336/25000 [=>............................] - ETA: 1:12 - loss: 7.7323 - accuracy: 0.4957
 2368/25000 [=>............................] - ETA: 1:12 - loss: 7.7378 - accuracy: 0.4954
 2400/25000 [=>............................] - ETA: 1:11 - loss: 7.7113 - accuracy: 0.4971
 2432/25000 [=>............................] - ETA: 1:11 - loss: 7.6918 - accuracy: 0.4984
 2464/25000 [=>............................] - ETA: 1:11 - loss: 7.6915 - accuracy: 0.4984
 2496/25000 [=>............................] - ETA: 1:11 - loss: 7.7035 - accuracy: 0.4976
 2528/25000 [==>...........................] - ETA: 1:11 - loss: 7.6909 - accuracy: 0.4984
 2560/25000 [==>...........................] - ETA: 1:11 - loss: 7.6906 - accuracy: 0.4984
 2592/25000 [==>...........................] - ETA: 1:11 - loss: 7.6844 - accuracy: 0.4988
 2624/25000 [==>...........................] - ETA: 1:11 - loss: 7.6374 - accuracy: 0.5019
 2656/25000 [==>...........................] - ETA: 1:11 - loss: 7.6262 - accuracy: 0.5026
 2688/25000 [==>...........................] - ETA: 1:10 - loss: 7.6153 - accuracy: 0.5033
 2720/25000 [==>...........................] - ETA: 1:10 - loss: 7.6215 - accuracy: 0.5029
 2752/25000 [==>...........................] - ETA: 1:10 - loss: 7.6555 - accuracy: 0.5007
 2784/25000 [==>...........................] - ETA: 1:10 - loss: 7.6611 - accuracy: 0.5004
 2816/25000 [==>...........................] - ETA: 1:10 - loss: 7.6557 - accuracy: 0.5007
 2848/25000 [==>...........................] - ETA: 1:10 - loss: 7.6612 - accuracy: 0.5004
 2880/25000 [==>...........................] - ETA: 1:10 - loss: 7.6560 - accuracy: 0.5007
 2912/25000 [==>...........................] - ETA: 1:10 - loss: 7.6614 - accuracy: 0.5003
 2944/25000 [==>...........................] - ETA: 1:09 - loss: 7.6510 - accuracy: 0.5010
 2976/25000 [==>...........................] - ETA: 1:09 - loss: 7.6460 - accuracy: 0.5013
 3008/25000 [==>...........................] - ETA: 1:09 - loss: 7.6462 - accuracy: 0.5013
 3040/25000 [==>...........................] - ETA: 1:09 - loss: 7.6212 - accuracy: 0.5030
 3072/25000 [==>...........................] - ETA: 1:09 - loss: 7.6017 - accuracy: 0.5042
 3104/25000 [==>...........................] - ETA: 1:09 - loss: 7.5925 - accuracy: 0.5048
 3136/25000 [==>...........................] - ETA: 1:09 - loss: 7.6128 - accuracy: 0.5035
 3168/25000 [==>...........................] - ETA: 1:09 - loss: 7.6085 - accuracy: 0.5038
 3200/25000 [==>...........................] - ETA: 1:09 - loss: 7.6235 - accuracy: 0.5028
 3232/25000 [==>...........................] - ETA: 1:08 - loss: 7.6192 - accuracy: 0.5031
 3264/25000 [==>...........................] - ETA: 1:08 - loss: 7.6149 - accuracy: 0.5034
 3296/25000 [==>...........................] - ETA: 1:08 - loss: 7.6154 - accuracy: 0.5033
 3328/25000 [==>...........................] - ETA: 1:08 - loss: 7.6252 - accuracy: 0.5027
 3360/25000 [===>..........................] - ETA: 1:08 - loss: 7.6164 - accuracy: 0.5033
 3392/25000 [===>..........................] - ETA: 1:08 - loss: 7.5943 - accuracy: 0.5047
 3424/25000 [===>..........................] - ETA: 1:08 - loss: 7.5905 - accuracy: 0.5050
 3456/25000 [===>..........................] - ETA: 1:08 - loss: 7.6001 - accuracy: 0.5043
 3488/25000 [===>..........................] - ETA: 1:08 - loss: 7.5875 - accuracy: 0.5052
 3520/25000 [===>..........................] - ETA: 1:07 - loss: 7.5664 - accuracy: 0.5065
 3552/25000 [===>..........................] - ETA: 1:07 - loss: 7.5717 - accuracy: 0.5062
 3584/25000 [===>..........................] - ETA: 1:07 - loss: 7.5768 - accuracy: 0.5059
 3616/25000 [===>..........................] - ETA: 1:07 - loss: 7.5733 - accuracy: 0.5061
 3648/25000 [===>..........................] - ETA: 1:07 - loss: 7.5784 - accuracy: 0.5058
 3680/25000 [===>..........................] - ETA: 1:07 - loss: 7.6041 - accuracy: 0.5041
 3712/25000 [===>..........................] - ETA: 1:07 - loss: 7.6047 - accuracy: 0.5040
 3744/25000 [===>..........................] - ETA: 1:07 - loss: 7.5970 - accuracy: 0.5045
 3776/25000 [===>..........................] - ETA: 1:07 - loss: 7.5813 - accuracy: 0.5056
 3808/25000 [===>..........................] - ETA: 1:07 - loss: 7.5780 - accuracy: 0.5058
 3840/25000 [===>..........................] - ETA: 1:06 - loss: 7.5908 - accuracy: 0.5049
 3872/25000 [===>..........................] - ETA: 1:06 - loss: 7.5795 - accuracy: 0.5057
 3904/25000 [===>..........................] - ETA: 1:06 - loss: 7.5841 - accuracy: 0.5054
 3936/25000 [===>..........................] - ETA: 1:06 - loss: 7.5848 - accuracy: 0.5053
 3968/25000 [===>..........................] - ETA: 1:06 - loss: 7.5932 - accuracy: 0.5048
 4000/25000 [===>..........................] - ETA: 1:06 - loss: 7.5938 - accuracy: 0.5048
 4032/25000 [===>..........................] - ETA: 1:06 - loss: 7.5792 - accuracy: 0.5057
 4064/25000 [===>..........................] - ETA: 1:06 - loss: 7.5798 - accuracy: 0.5057
 4096/25000 [===>..........................] - ETA: 1:06 - loss: 7.5805 - accuracy: 0.5056
 4128/25000 [===>..........................] - ETA: 1:06 - loss: 7.5738 - accuracy: 0.5061
 4160/25000 [===>..........................] - ETA: 1:05 - loss: 7.5597 - accuracy: 0.5070
 4192/25000 [====>.........................] - ETA: 1:05 - loss: 7.5569 - accuracy: 0.5072
 4224/25000 [====>.........................] - ETA: 1:05 - loss: 7.5759 - accuracy: 0.5059
 4256/25000 [====>.........................] - ETA: 1:05 - loss: 7.5621 - accuracy: 0.5068
 4288/25000 [====>.........................] - ETA: 1:05 - loss: 7.5736 - accuracy: 0.5061
 4320/25000 [====>.........................] - ETA: 1:05 - loss: 7.5850 - accuracy: 0.5053
 4352/25000 [====>.........................] - ETA: 1:05 - loss: 7.5750 - accuracy: 0.5060
 4384/25000 [====>.........................] - ETA: 1:05 - loss: 7.5862 - accuracy: 0.5052
 4416/25000 [====>.........................] - ETA: 1:05 - loss: 7.5868 - accuracy: 0.5052
 4448/25000 [====>.........................] - ETA: 1:05 - loss: 7.5839 - accuracy: 0.5054
 4480/25000 [====>.........................] - ETA: 1:05 - loss: 7.5913 - accuracy: 0.5049
 4512/25000 [====>.........................] - ETA: 1:04 - loss: 7.5715 - accuracy: 0.5062
 4544/25000 [====>.........................] - ETA: 1:04 - loss: 7.5823 - accuracy: 0.5055
 4576/25000 [====>.........................] - ETA: 1:04 - loss: 7.5694 - accuracy: 0.5063
 4608/25000 [====>.........................] - ETA: 1:04 - loss: 7.5635 - accuracy: 0.5067
 4640/25000 [====>.........................] - ETA: 1:04 - loss: 7.5609 - accuracy: 0.5069
 4672/25000 [====>.........................] - ETA: 1:04 - loss: 7.5747 - accuracy: 0.5060
 4704/25000 [====>.........................] - ETA: 1:04 - loss: 7.5721 - accuracy: 0.5062
 4736/25000 [====>.........................] - ETA: 1:04 - loss: 7.5663 - accuracy: 0.5065
 4768/25000 [====>.........................] - ETA: 1:04 - loss: 7.5830 - accuracy: 0.5055
 4800/25000 [====>.........................] - ETA: 1:03 - loss: 7.5868 - accuracy: 0.5052
 4832/25000 [====>.........................] - ETA: 1:03 - loss: 7.5778 - accuracy: 0.5058
 4864/25000 [====>.........................] - ETA: 1:03 - loss: 7.5847 - accuracy: 0.5053
 4896/25000 [====>.........................] - ETA: 1:03 - loss: 7.5821 - accuracy: 0.5055
 4928/25000 [====>.........................] - ETA: 1:03 - loss: 7.5888 - accuracy: 0.5051
 4960/25000 [====>.........................] - ETA: 1:03 - loss: 7.5801 - accuracy: 0.5056
 4992/25000 [====>.........................] - ETA: 1:03 - loss: 7.5960 - accuracy: 0.5046
 5024/25000 [=====>........................] - ETA: 1:03 - loss: 7.5903 - accuracy: 0.5050
 5056/25000 [=====>........................] - ETA: 1:03 - loss: 7.5908 - accuracy: 0.5049
 5088/25000 [=====>........................] - ETA: 1:02 - loss: 7.5792 - accuracy: 0.5057
 5120/25000 [=====>........................] - ETA: 1:02 - loss: 7.5858 - accuracy: 0.5053
 5152/25000 [=====>........................] - ETA: 1:02 - loss: 7.5922 - accuracy: 0.5049
 5184/25000 [=====>........................] - ETA: 1:02 - loss: 7.6075 - accuracy: 0.5039
 5216/25000 [=====>........................] - ETA: 1:02 - loss: 7.6049 - accuracy: 0.5040
 5248/25000 [=====>........................] - ETA: 1:02 - loss: 7.6199 - accuracy: 0.5030
 5280/25000 [=====>........................] - ETA: 1:02 - loss: 7.6143 - accuracy: 0.5034
 5312/25000 [=====>........................] - ETA: 1:02 - loss: 7.6147 - accuracy: 0.5034
 5344/25000 [=====>........................] - ETA: 1:02 - loss: 7.6121 - accuracy: 0.5036
 5376/25000 [=====>........................] - ETA: 1:02 - loss: 7.6010 - accuracy: 0.5043
 5408/25000 [=====>........................] - ETA: 1:01 - loss: 7.6099 - accuracy: 0.5037
 5440/25000 [=====>........................] - ETA: 1:01 - loss: 7.6074 - accuracy: 0.5039
 5472/25000 [=====>........................] - ETA: 1:01 - loss: 7.6050 - accuracy: 0.5040
 5504/25000 [=====>........................] - ETA: 1:01 - loss: 7.6053 - accuracy: 0.5040
 5536/25000 [=====>........................] - ETA: 1:01 - loss: 7.6029 - accuracy: 0.5042
 5568/25000 [=====>........................] - ETA: 1:01 - loss: 7.6060 - accuracy: 0.5040
 5600/25000 [=====>........................] - ETA: 1:01 - loss: 7.5954 - accuracy: 0.5046
 5632/25000 [=====>........................] - ETA: 1:01 - loss: 7.5877 - accuracy: 0.5051
 5664/25000 [=====>........................] - ETA: 1:00 - loss: 7.5854 - accuracy: 0.5053
 5696/25000 [=====>........................] - ETA: 1:00 - loss: 7.5886 - accuracy: 0.5051
 5728/25000 [=====>........................] - ETA: 1:00 - loss: 7.5810 - accuracy: 0.5056
 5760/25000 [=====>........................] - ETA: 1:00 - loss: 7.5894 - accuracy: 0.5050
 5792/25000 [=====>........................] - ETA: 1:00 - loss: 7.5978 - accuracy: 0.5045
 5824/25000 [=====>........................] - ETA: 1:00 - loss: 7.6034 - accuracy: 0.5041
 5856/25000 [======>.......................] - ETA: 1:00 - loss: 7.6090 - accuracy: 0.5038
 5888/25000 [======>.......................] - ETA: 1:00 - loss: 7.5911 - accuracy: 0.5049
 5920/25000 [======>.......................] - ETA: 1:00 - loss: 7.5993 - accuracy: 0.5044
 5952/25000 [======>.......................] - ETA: 1:00 - loss: 7.6022 - accuracy: 0.5042
 5984/25000 [======>.......................] - ETA: 59s - loss: 7.6102 - accuracy: 0.5037 
 6016/25000 [======>.......................] - ETA: 59s - loss: 7.6207 - accuracy: 0.5030
 6048/25000 [======>.......................] - ETA: 59s - loss: 7.6184 - accuracy: 0.5031
 6080/25000 [======>.......................] - ETA: 59s - loss: 7.6288 - accuracy: 0.5025
 6112/25000 [======>.......................] - ETA: 59s - loss: 7.6290 - accuracy: 0.5025
 6144/25000 [======>.......................] - ETA: 59s - loss: 7.6442 - accuracy: 0.5015
 6176/25000 [======>.......................] - ETA: 59s - loss: 7.6319 - accuracy: 0.5023
 6208/25000 [======>.......................] - ETA: 59s - loss: 7.6296 - accuracy: 0.5024
 6240/25000 [======>.......................] - ETA: 59s - loss: 7.6347 - accuracy: 0.5021
 6272/25000 [======>.......................] - ETA: 58s - loss: 7.6324 - accuracy: 0.5022
 6304/25000 [======>.......................] - ETA: 58s - loss: 7.6301 - accuracy: 0.5024
 6336/25000 [======>.......................] - ETA: 58s - loss: 7.6327 - accuracy: 0.5022
 6368/25000 [======>.......................] - ETA: 58s - loss: 7.6281 - accuracy: 0.5025
 6400/25000 [======>.......................] - ETA: 58s - loss: 7.6379 - accuracy: 0.5019
 6432/25000 [======>.......................] - ETA: 58s - loss: 7.6404 - accuracy: 0.5017
 6464/25000 [======>.......................] - ETA: 58s - loss: 7.6310 - accuracy: 0.5023
 6496/25000 [======>.......................] - ETA: 58s - loss: 7.6265 - accuracy: 0.5026
 6528/25000 [======>.......................] - ETA: 58s - loss: 7.6267 - accuracy: 0.5026
 6560/25000 [======>.......................] - ETA: 57s - loss: 7.6339 - accuracy: 0.5021
 6592/25000 [======>.......................] - ETA: 57s - loss: 7.6341 - accuracy: 0.5021
 6624/25000 [======>.......................] - ETA: 57s - loss: 7.6296 - accuracy: 0.5024
 6656/25000 [======>.......................] - ETA: 57s - loss: 7.6390 - accuracy: 0.5018
 6688/25000 [=======>......................] - ETA: 57s - loss: 7.6368 - accuracy: 0.5019
 6720/25000 [=======>......................] - ETA: 57s - loss: 7.6370 - accuracy: 0.5019
 6752/25000 [=======>......................] - ETA: 57s - loss: 7.6416 - accuracy: 0.5016
 6784/25000 [=======>......................] - ETA: 57s - loss: 7.6440 - accuracy: 0.5015
 6816/25000 [=======>......................] - ETA: 57s - loss: 7.6351 - accuracy: 0.5021
 6848/25000 [=======>......................] - ETA: 57s - loss: 7.6241 - accuracy: 0.5028
 6880/25000 [=======>......................] - ETA: 56s - loss: 7.6310 - accuracy: 0.5023
 6912/25000 [=======>......................] - ETA: 56s - loss: 7.6444 - accuracy: 0.5014
 6944/25000 [=======>......................] - ETA: 56s - loss: 7.6512 - accuracy: 0.5010
 6976/25000 [=======>......................] - ETA: 56s - loss: 7.6600 - accuracy: 0.5004
 7008/25000 [=======>......................] - ETA: 56s - loss: 7.6579 - accuracy: 0.5006
 7040/25000 [=======>......................] - ETA: 56s - loss: 7.6579 - accuracy: 0.5006
 7072/25000 [=======>......................] - ETA: 56s - loss: 7.6601 - accuracy: 0.5004
 7104/25000 [=======>......................] - ETA: 56s - loss: 7.6645 - accuracy: 0.5001
 7136/25000 [=======>......................] - ETA: 56s - loss: 7.6623 - accuracy: 0.5003
 7168/25000 [=======>......................] - ETA: 56s - loss: 7.6581 - accuracy: 0.5006
 7200/25000 [=======>......................] - ETA: 55s - loss: 7.6773 - accuracy: 0.4993
 7232/25000 [=======>......................] - ETA: 55s - loss: 7.6709 - accuracy: 0.4997
 7264/25000 [=======>......................] - ETA: 55s - loss: 7.6751 - accuracy: 0.4994
 7296/25000 [=======>......................] - ETA: 55s - loss: 7.6708 - accuracy: 0.4997
 7328/25000 [=======>......................] - ETA: 55s - loss: 7.6687 - accuracy: 0.4999
 7360/25000 [=======>......................] - ETA: 55s - loss: 7.6708 - accuracy: 0.4997
 7392/25000 [=======>......................] - ETA: 55s - loss: 7.6728 - accuracy: 0.4996
 7424/25000 [=======>......................] - ETA: 55s - loss: 7.6769 - accuracy: 0.4993
 7456/25000 [=======>......................] - ETA: 55s - loss: 7.6790 - accuracy: 0.4992
 7488/25000 [=======>......................] - ETA: 55s - loss: 7.6789 - accuracy: 0.4992
 7520/25000 [========>.....................] - ETA: 54s - loss: 7.6850 - accuracy: 0.4988
 7552/25000 [========>.....................] - ETA: 54s - loss: 7.6788 - accuracy: 0.4992
 7584/25000 [========>.....................] - ETA: 54s - loss: 7.6828 - accuracy: 0.4989
 7616/25000 [========>.....................] - ETA: 54s - loss: 7.6928 - accuracy: 0.4983
 7648/25000 [========>.....................] - ETA: 54s - loss: 7.6887 - accuracy: 0.4986
 7680/25000 [========>.....................] - ETA: 54s - loss: 7.6906 - accuracy: 0.4984
 7712/25000 [========>.....................] - ETA: 54s - loss: 7.6905 - accuracy: 0.4984
 7744/25000 [========>.....................] - ETA: 54s - loss: 7.6884 - accuracy: 0.4986
 7776/25000 [========>.....................] - ETA: 54s - loss: 7.6785 - accuracy: 0.4992
 7808/25000 [========>.....................] - ETA: 54s - loss: 7.6823 - accuracy: 0.4990
 7840/25000 [========>.....................] - ETA: 53s - loss: 7.6764 - accuracy: 0.4994
 7872/25000 [========>.....................] - ETA: 53s - loss: 7.6822 - accuracy: 0.4990
 7904/25000 [========>.....................] - ETA: 53s - loss: 7.6744 - accuracy: 0.4995
 7936/25000 [========>.....................] - ETA: 53s - loss: 7.6686 - accuracy: 0.4999
 7968/25000 [========>.....................] - ETA: 53s - loss: 7.6647 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 53s - loss: 7.6647 - accuracy: 0.5001
 8032/25000 [========>.....................] - ETA: 53s - loss: 7.6628 - accuracy: 0.5002
 8064/25000 [========>.....................] - ETA: 53s - loss: 7.6609 - accuracy: 0.5004
 8096/25000 [========>.....................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 8128/25000 [========>.....................] - ETA: 52s - loss: 7.6779 - accuracy: 0.4993
 8160/25000 [========>.....................] - ETA: 52s - loss: 7.6779 - accuracy: 0.4993
 8192/25000 [========>.....................] - ETA: 52s - loss: 7.6835 - accuracy: 0.4989
 8224/25000 [========>.....................] - ETA: 52s - loss: 7.6815 - accuracy: 0.4990
 8256/25000 [========>.....................] - ETA: 52s - loss: 7.6759 - accuracy: 0.4994
 8288/25000 [========>.....................] - ETA: 52s - loss: 7.6796 - accuracy: 0.4992
 8320/25000 [========>.....................] - ETA: 52s - loss: 7.6869 - accuracy: 0.4987
 8352/25000 [=========>....................] - ETA: 52s - loss: 7.6850 - accuracy: 0.4988
 8384/25000 [=========>....................] - ETA: 52s - loss: 7.6922 - accuracy: 0.4983
 8416/25000 [=========>....................] - ETA: 52s - loss: 7.6885 - accuracy: 0.4986
 8448/25000 [=========>....................] - ETA: 51s - loss: 7.6830 - accuracy: 0.4989
 8480/25000 [=========>....................] - ETA: 51s - loss: 7.6865 - accuracy: 0.4987
 8512/25000 [=========>....................] - ETA: 51s - loss: 7.6864 - accuracy: 0.4987
 8544/25000 [=========>....................] - ETA: 51s - loss: 7.6846 - accuracy: 0.4988
 8576/25000 [=========>....................] - ETA: 51s - loss: 7.6773 - accuracy: 0.4993
 8608/25000 [=========>....................] - ETA: 51s - loss: 7.6791 - accuracy: 0.4992
 8640/25000 [=========>....................] - ETA: 51s - loss: 7.6755 - accuracy: 0.4994
 8672/25000 [=========>....................] - ETA: 51s - loss: 7.6755 - accuracy: 0.4994
 8704/25000 [=========>....................] - ETA: 51s - loss: 7.6754 - accuracy: 0.4994
 8736/25000 [=========>....................] - ETA: 51s - loss: 7.6736 - accuracy: 0.4995
 8768/25000 [=========>....................] - ETA: 50s - loss: 7.6824 - accuracy: 0.4990
 8800/25000 [=========>....................] - ETA: 50s - loss: 7.6771 - accuracy: 0.4993
 8832/25000 [=========>....................] - ETA: 50s - loss: 7.6805 - accuracy: 0.4991
 8864/25000 [=========>....................] - ETA: 50s - loss: 7.6770 - accuracy: 0.4993
 8896/25000 [=========>....................] - ETA: 50s - loss: 7.6752 - accuracy: 0.4994
 8928/25000 [=========>....................] - ETA: 50s - loss: 7.6701 - accuracy: 0.4998
 8960/25000 [=========>....................] - ETA: 50s - loss: 7.6632 - accuracy: 0.5002
 8992/25000 [=========>....................] - ETA: 50s - loss: 7.6598 - accuracy: 0.5004
 9024/25000 [=========>....................] - ETA: 50s - loss: 7.6513 - accuracy: 0.5010
 9056/25000 [=========>....................] - ETA: 50s - loss: 7.6480 - accuracy: 0.5012
 9088/25000 [=========>....................] - ETA: 49s - loss: 7.6531 - accuracy: 0.5009
 9120/25000 [=========>....................] - ETA: 49s - loss: 7.6532 - accuracy: 0.5009
 9152/25000 [=========>....................] - ETA: 49s - loss: 7.6465 - accuracy: 0.5013
 9184/25000 [==========>...................] - ETA: 49s - loss: 7.6533 - accuracy: 0.5009
 9216/25000 [==========>...................] - ETA: 49s - loss: 7.6566 - accuracy: 0.5007
 9248/25000 [==========>...................] - ETA: 49s - loss: 7.6451 - accuracy: 0.5014
 9280/25000 [==========>...................] - ETA: 49s - loss: 7.6336 - accuracy: 0.5022
 9312/25000 [==========>...................] - ETA: 49s - loss: 7.6287 - accuracy: 0.5025
 9344/25000 [==========>...................] - ETA: 49s - loss: 7.6256 - accuracy: 0.5027
 9376/25000 [==========>...................] - ETA: 49s - loss: 7.6208 - accuracy: 0.5030
 9408/25000 [==========>...................] - ETA: 48s - loss: 7.6275 - accuracy: 0.5026
 9440/25000 [==========>...................] - ETA: 48s - loss: 7.6179 - accuracy: 0.5032
 9472/25000 [==========>...................] - ETA: 48s - loss: 7.6229 - accuracy: 0.5029
 9504/25000 [==========>...................] - ETA: 48s - loss: 7.6279 - accuracy: 0.5025
 9536/25000 [==========>...................] - ETA: 48s - loss: 7.6329 - accuracy: 0.5022
 9568/25000 [==========>...................] - ETA: 48s - loss: 7.6298 - accuracy: 0.5024
 9600/25000 [==========>...................] - ETA: 48s - loss: 7.6283 - accuracy: 0.5025
 9632/25000 [==========>...................] - ETA: 48s - loss: 7.6284 - accuracy: 0.5025
 9664/25000 [==========>...................] - ETA: 48s - loss: 7.6365 - accuracy: 0.5020
 9696/25000 [==========>...................] - ETA: 48s - loss: 7.6334 - accuracy: 0.5022
 9728/25000 [==========>...................] - ETA: 47s - loss: 7.6288 - accuracy: 0.5025
 9760/25000 [==========>...................] - ETA: 47s - loss: 7.6352 - accuracy: 0.5020
 9792/25000 [==========>...................] - ETA: 47s - loss: 7.6306 - accuracy: 0.5023
 9824/25000 [==========>...................] - ETA: 47s - loss: 7.6276 - accuracy: 0.5025
 9856/25000 [==========>...................] - ETA: 47s - loss: 7.6371 - accuracy: 0.5019
 9888/25000 [==========>...................] - ETA: 47s - loss: 7.6434 - accuracy: 0.5015
 9920/25000 [==========>...................] - ETA: 47s - loss: 7.6481 - accuracy: 0.5012
 9952/25000 [==========>...................] - ETA: 47s - loss: 7.6497 - accuracy: 0.5011
 9984/25000 [==========>...................] - ETA: 47s - loss: 7.6528 - accuracy: 0.5009
10016/25000 [===========>..................] - ETA: 47s - loss: 7.6544 - accuracy: 0.5008
10048/25000 [===========>..................] - ETA: 46s - loss: 7.6544 - accuracy: 0.5008
10080/25000 [===========>..................] - ETA: 46s - loss: 7.6590 - accuracy: 0.5005
10112/25000 [===========>..................] - ETA: 46s - loss: 7.6651 - accuracy: 0.5001
10144/25000 [===========>..................] - ETA: 46s - loss: 7.6727 - accuracy: 0.4996
10176/25000 [===========>..................] - ETA: 46s - loss: 7.6681 - accuracy: 0.4999
10208/25000 [===========>..................] - ETA: 46s - loss: 7.6726 - accuracy: 0.4996
10240/25000 [===========>..................] - ETA: 46s - loss: 7.6816 - accuracy: 0.4990
10272/25000 [===========>..................] - ETA: 46s - loss: 7.6815 - accuracy: 0.4990
10304/25000 [===========>..................] - ETA: 46s - loss: 7.6845 - accuracy: 0.4988
10336/25000 [===========>..................] - ETA: 46s - loss: 7.6859 - accuracy: 0.4987
10368/25000 [===========>..................] - ETA: 45s - loss: 7.6799 - accuracy: 0.4991
10400/25000 [===========>..................] - ETA: 45s - loss: 7.6740 - accuracy: 0.4995
10432/25000 [===========>..................] - ETA: 45s - loss: 7.6710 - accuracy: 0.4997
10464/25000 [===========>..................] - ETA: 45s - loss: 7.6754 - accuracy: 0.4994
10496/25000 [===========>..................] - ETA: 45s - loss: 7.6856 - accuracy: 0.4988
10528/25000 [===========>..................] - ETA: 45s - loss: 7.6856 - accuracy: 0.4988
10560/25000 [===========>..................] - ETA: 45s - loss: 7.6797 - accuracy: 0.4991
10592/25000 [===========>..................] - ETA: 45s - loss: 7.6854 - accuracy: 0.4988
10624/25000 [===========>..................] - ETA: 45s - loss: 7.6883 - accuracy: 0.4986
10656/25000 [===========>..................] - ETA: 45s - loss: 7.6853 - accuracy: 0.4988
10688/25000 [===========>..................] - ETA: 44s - loss: 7.6867 - accuracy: 0.4987
10720/25000 [===========>..................] - ETA: 44s - loss: 7.6852 - accuracy: 0.4988
10752/25000 [===========>..................] - ETA: 44s - loss: 7.6823 - accuracy: 0.4990
10784/25000 [===========>..................] - ETA: 44s - loss: 7.6865 - accuracy: 0.4987
10816/25000 [===========>..................] - ETA: 44s - loss: 7.6808 - accuracy: 0.4991
10848/25000 [============>.................] - ETA: 44s - loss: 7.6808 - accuracy: 0.4991
10880/25000 [============>.................] - ETA: 44s - loss: 7.6849 - accuracy: 0.4988
10912/25000 [============>.................] - ETA: 44s - loss: 7.6821 - accuracy: 0.4990
10944/25000 [============>.................] - ETA: 44s - loss: 7.6778 - accuracy: 0.4993
10976/25000 [============>.................] - ETA: 43s - loss: 7.6750 - accuracy: 0.4995
11008/25000 [============>.................] - ETA: 43s - loss: 7.6792 - accuracy: 0.4992
11040/25000 [============>.................] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
11072/25000 [============>.................] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
11104/25000 [============>.................] - ETA: 43s - loss: 7.6749 - accuracy: 0.4995
11136/25000 [============>.................] - ETA: 43s - loss: 7.6708 - accuracy: 0.4997
11168/25000 [============>.................] - ETA: 43s - loss: 7.6652 - accuracy: 0.5001
11200/25000 [============>.................] - ETA: 43s - loss: 7.6625 - accuracy: 0.5003
11232/25000 [============>.................] - ETA: 43s - loss: 7.6625 - accuracy: 0.5003
11264/25000 [============>.................] - ETA: 43s - loss: 7.6571 - accuracy: 0.5006
11296/25000 [============>.................] - ETA: 42s - loss: 7.6544 - accuracy: 0.5008
11328/25000 [============>.................] - ETA: 42s - loss: 7.6558 - accuracy: 0.5007
11360/25000 [============>.................] - ETA: 42s - loss: 7.6545 - accuracy: 0.5008
11392/25000 [============>.................] - ETA: 42s - loss: 7.6518 - accuracy: 0.5010
11424/25000 [============>.................] - ETA: 42s - loss: 7.6532 - accuracy: 0.5009
11456/25000 [============>.................] - ETA: 42s - loss: 7.6532 - accuracy: 0.5009
11488/25000 [============>.................] - ETA: 42s - loss: 7.6599 - accuracy: 0.5004
11520/25000 [============>.................] - ETA: 42s - loss: 7.6586 - accuracy: 0.5005
11552/25000 [============>.................] - ETA: 42s - loss: 7.6626 - accuracy: 0.5003
11584/25000 [============>.................] - ETA: 41s - loss: 7.6640 - accuracy: 0.5002
11616/25000 [============>.................] - ETA: 41s - loss: 7.6653 - accuracy: 0.5001
11648/25000 [============>.................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
11680/25000 [=============>................] - ETA: 41s - loss: 7.6706 - accuracy: 0.4997
11712/25000 [=============>................] - ETA: 41s - loss: 7.6732 - accuracy: 0.4996
11744/25000 [=============>................] - ETA: 41s - loss: 7.6745 - accuracy: 0.4995
11776/25000 [=============>................] - ETA: 41s - loss: 7.6783 - accuracy: 0.4992
11808/25000 [=============>................] - ETA: 41s - loss: 7.6757 - accuracy: 0.4994
11840/25000 [=============>................] - ETA: 41s - loss: 7.6731 - accuracy: 0.4996
11872/25000 [=============>................] - ETA: 41s - loss: 7.6718 - accuracy: 0.4997
11904/25000 [=============>................] - ETA: 40s - loss: 7.6718 - accuracy: 0.4997
11936/25000 [=============>................] - ETA: 40s - loss: 7.6692 - accuracy: 0.4998
11968/25000 [=============>................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 40s - loss: 7.6653 - accuracy: 0.5001
12032/25000 [=============>................] - ETA: 40s - loss: 7.6692 - accuracy: 0.4998
12064/25000 [=============>................] - ETA: 40s - loss: 7.6692 - accuracy: 0.4998
12096/25000 [=============>................] - ETA: 40s - loss: 7.6679 - accuracy: 0.4999
12128/25000 [=============>................] - ETA: 40s - loss: 7.6729 - accuracy: 0.4996
12160/25000 [=============>................] - ETA: 40s - loss: 7.6742 - accuracy: 0.4995
12192/25000 [=============>................] - ETA: 40s - loss: 7.6729 - accuracy: 0.4996
12224/25000 [=============>................] - ETA: 39s - loss: 7.6729 - accuracy: 0.4996
12256/25000 [=============>................] - ETA: 39s - loss: 7.6666 - accuracy: 0.5000
12288/25000 [=============>................] - ETA: 39s - loss: 7.6716 - accuracy: 0.4997
12320/25000 [=============>................] - ETA: 39s - loss: 7.6704 - accuracy: 0.4998
12352/25000 [=============>................] - ETA: 39s - loss: 7.6679 - accuracy: 0.4999
12384/25000 [=============>................] - ETA: 39s - loss: 7.6679 - accuracy: 0.4999
12416/25000 [=============>................] - ETA: 39s - loss: 7.6691 - accuracy: 0.4998
12448/25000 [=============>................] - ETA: 39s - loss: 7.6679 - accuracy: 0.4999
12480/25000 [=============>................] - ETA: 39s - loss: 7.6666 - accuracy: 0.5000
12512/25000 [==============>...............] - ETA: 39s - loss: 7.6691 - accuracy: 0.4998
12544/25000 [==============>...............] - ETA: 38s - loss: 7.6678 - accuracy: 0.4999
12576/25000 [==============>...............] - ETA: 38s - loss: 7.6715 - accuracy: 0.4997
12608/25000 [==============>...............] - ETA: 38s - loss: 7.6691 - accuracy: 0.4998
12640/25000 [==============>...............] - ETA: 38s - loss: 7.6654 - accuracy: 0.5001
12672/25000 [==============>...............] - ETA: 38s - loss: 7.6642 - accuracy: 0.5002
12704/25000 [==============>...............] - ETA: 38s - loss: 7.6606 - accuracy: 0.5004
12736/25000 [==============>...............] - ETA: 38s - loss: 7.6606 - accuracy: 0.5004
12768/25000 [==============>...............] - ETA: 38s - loss: 7.6618 - accuracy: 0.5003
12800/25000 [==============>...............] - ETA: 38s - loss: 7.6630 - accuracy: 0.5002
12832/25000 [==============>...............] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
12864/25000 [==============>...............] - ETA: 37s - loss: 7.6619 - accuracy: 0.5003
12896/25000 [==============>...............] - ETA: 37s - loss: 7.6571 - accuracy: 0.5006
12928/25000 [==============>...............] - ETA: 37s - loss: 7.6571 - accuracy: 0.5006
12960/25000 [==============>...............] - ETA: 37s - loss: 7.6583 - accuracy: 0.5005
12992/25000 [==============>...............] - ETA: 37s - loss: 7.6619 - accuracy: 0.5003
13024/25000 [==============>...............] - ETA: 37s - loss: 7.6584 - accuracy: 0.5005
13056/25000 [==============>...............] - ETA: 37s - loss: 7.6607 - accuracy: 0.5004
13088/25000 [==============>...............] - ETA: 37s - loss: 7.6666 - accuracy: 0.5000
13120/25000 [==============>...............] - ETA: 37s - loss: 7.6725 - accuracy: 0.4996
13152/25000 [==============>...............] - ETA: 37s - loss: 7.6713 - accuracy: 0.4997
13184/25000 [==============>...............] - ETA: 36s - loss: 7.6678 - accuracy: 0.4999
13216/25000 [==============>...............] - ETA: 36s - loss: 7.6678 - accuracy: 0.4999
13248/25000 [==============>...............] - ETA: 36s - loss: 7.6655 - accuracy: 0.5001
13280/25000 [==============>...............] - ETA: 36s - loss: 7.6655 - accuracy: 0.5001
13312/25000 [==============>...............] - ETA: 36s - loss: 7.6689 - accuracy: 0.4998
13344/25000 [===============>..............] - ETA: 36s - loss: 7.6643 - accuracy: 0.5001
13376/25000 [===============>..............] - ETA: 36s - loss: 7.6609 - accuracy: 0.5004
13408/25000 [===============>..............] - ETA: 36s - loss: 7.6598 - accuracy: 0.5004
13440/25000 [===============>..............] - ETA: 36s - loss: 7.6564 - accuracy: 0.5007
13472/25000 [===============>..............] - ETA: 35s - loss: 7.6598 - accuracy: 0.5004
13504/25000 [===============>..............] - ETA: 35s - loss: 7.6587 - accuracy: 0.5005
13536/25000 [===============>..............] - ETA: 35s - loss: 7.6576 - accuracy: 0.5006
13568/25000 [===============>..............] - ETA: 35s - loss: 7.6542 - accuracy: 0.5008
13600/25000 [===============>..............] - ETA: 35s - loss: 7.6565 - accuracy: 0.5007
13632/25000 [===============>..............] - ETA: 35s - loss: 7.6520 - accuracy: 0.5010
13664/25000 [===============>..............] - ETA: 35s - loss: 7.6487 - accuracy: 0.5012
13696/25000 [===============>..............] - ETA: 35s - loss: 7.6521 - accuracy: 0.5009
13728/25000 [===============>..............] - ETA: 35s - loss: 7.6555 - accuracy: 0.5007
13760/25000 [===============>..............] - ETA: 35s - loss: 7.6510 - accuracy: 0.5010
13792/25000 [===============>..............] - ETA: 34s - loss: 7.6522 - accuracy: 0.5009
13824/25000 [===============>..............] - ETA: 34s - loss: 7.6500 - accuracy: 0.5011
13856/25000 [===============>..............] - ETA: 34s - loss: 7.6500 - accuracy: 0.5011
13888/25000 [===============>..............] - ETA: 34s - loss: 7.6534 - accuracy: 0.5009
13920/25000 [===============>..............] - ETA: 34s - loss: 7.6534 - accuracy: 0.5009
13952/25000 [===============>..............] - ETA: 34s - loss: 7.6512 - accuracy: 0.5010
13984/25000 [===============>..............] - ETA: 34s - loss: 7.6513 - accuracy: 0.5010
14016/25000 [===============>..............] - ETA: 34s - loss: 7.6568 - accuracy: 0.5006
14048/25000 [===============>..............] - ETA: 34s - loss: 7.6601 - accuracy: 0.5004
14080/25000 [===============>..............] - ETA: 34s - loss: 7.6568 - accuracy: 0.5006
14112/25000 [===============>..............] - ETA: 33s - loss: 7.6503 - accuracy: 0.5011
14144/25000 [===============>..............] - ETA: 33s - loss: 7.6493 - accuracy: 0.5011
14176/25000 [================>.............] - ETA: 33s - loss: 7.6472 - accuracy: 0.5013
14208/25000 [================>.............] - ETA: 33s - loss: 7.6483 - accuracy: 0.5012
14240/25000 [================>.............] - ETA: 33s - loss: 7.6494 - accuracy: 0.5011
14272/25000 [================>.............] - ETA: 33s - loss: 7.6527 - accuracy: 0.5009
14304/25000 [================>.............] - ETA: 33s - loss: 7.6538 - accuracy: 0.5008
14336/25000 [================>.............] - ETA: 33s - loss: 7.6516 - accuracy: 0.5010
14368/25000 [================>.............] - ETA: 33s - loss: 7.6506 - accuracy: 0.5010
14400/25000 [================>.............] - ETA: 33s - loss: 7.6496 - accuracy: 0.5011
14432/25000 [================>.............] - ETA: 32s - loss: 7.6496 - accuracy: 0.5011
14464/25000 [================>.............] - ETA: 32s - loss: 7.6497 - accuracy: 0.5011
14496/25000 [================>.............] - ETA: 32s - loss: 7.6486 - accuracy: 0.5012
14528/25000 [================>.............] - ETA: 32s - loss: 7.6476 - accuracy: 0.5012
14560/25000 [================>.............] - ETA: 32s - loss: 7.6456 - accuracy: 0.5014
14592/25000 [================>.............] - ETA: 32s - loss: 7.6425 - accuracy: 0.5016
14624/25000 [================>.............] - ETA: 32s - loss: 7.6425 - accuracy: 0.5016
14656/25000 [================>.............] - ETA: 32s - loss: 7.6405 - accuracy: 0.5017
14688/25000 [================>.............] - ETA: 32s - loss: 7.6426 - accuracy: 0.5016
14720/25000 [================>.............] - ETA: 32s - loss: 7.6447 - accuracy: 0.5014
14752/25000 [================>.............] - ETA: 31s - loss: 7.6469 - accuracy: 0.5013
14784/25000 [================>.............] - ETA: 31s - loss: 7.6500 - accuracy: 0.5011
14816/25000 [================>.............] - ETA: 31s - loss: 7.6501 - accuracy: 0.5011
14848/25000 [================>.............] - ETA: 31s - loss: 7.6501 - accuracy: 0.5011
14880/25000 [================>.............] - ETA: 31s - loss: 7.6501 - accuracy: 0.5011
14912/25000 [================>.............] - ETA: 31s - loss: 7.6563 - accuracy: 0.5007
14944/25000 [================>.............] - ETA: 31s - loss: 7.6584 - accuracy: 0.5005
14976/25000 [================>.............] - ETA: 31s - loss: 7.6605 - accuracy: 0.5004
15008/25000 [=================>............] - ETA: 31s - loss: 7.6646 - accuracy: 0.5001
15040/25000 [=================>............] - ETA: 31s - loss: 7.6615 - accuracy: 0.5003
15072/25000 [=================>............] - ETA: 30s - loss: 7.6626 - accuracy: 0.5003
15104/25000 [=================>............] - ETA: 30s - loss: 7.6585 - accuracy: 0.5005
15136/25000 [=================>............] - ETA: 30s - loss: 7.6575 - accuracy: 0.5006
15168/25000 [=================>............] - ETA: 30s - loss: 7.6545 - accuracy: 0.5008
15200/25000 [=================>............] - ETA: 30s - loss: 7.6545 - accuracy: 0.5008
15232/25000 [=================>............] - ETA: 30s - loss: 7.6525 - accuracy: 0.5009
15264/25000 [=================>............] - ETA: 30s - loss: 7.6465 - accuracy: 0.5013
15296/25000 [=================>............] - ETA: 30s - loss: 7.6446 - accuracy: 0.5014
15328/25000 [=================>............] - ETA: 30s - loss: 7.6416 - accuracy: 0.5016
15360/25000 [=================>............] - ETA: 30s - loss: 7.6387 - accuracy: 0.5018
15392/25000 [=================>............] - ETA: 29s - loss: 7.6397 - accuracy: 0.5018
15424/25000 [=================>............] - ETA: 29s - loss: 7.6378 - accuracy: 0.5019
15456/25000 [=================>............] - ETA: 29s - loss: 7.6369 - accuracy: 0.5019
15488/25000 [=================>............] - ETA: 29s - loss: 7.6349 - accuracy: 0.5021
15520/25000 [=================>............] - ETA: 29s - loss: 7.6340 - accuracy: 0.5021
15552/25000 [=================>............] - ETA: 29s - loss: 7.6361 - accuracy: 0.5020
15584/25000 [=================>............] - ETA: 29s - loss: 7.6332 - accuracy: 0.5022
15616/25000 [=================>............] - ETA: 29s - loss: 7.6303 - accuracy: 0.5024
15648/25000 [=================>............] - ETA: 29s - loss: 7.6294 - accuracy: 0.5024
15680/25000 [=================>............] - ETA: 29s - loss: 7.6334 - accuracy: 0.5022
15712/25000 [=================>............] - ETA: 28s - loss: 7.6325 - accuracy: 0.5022
15744/25000 [=================>............] - ETA: 28s - loss: 7.6335 - accuracy: 0.5022
15776/25000 [=================>............] - ETA: 28s - loss: 7.6345 - accuracy: 0.5021
15808/25000 [=================>............] - ETA: 28s - loss: 7.6356 - accuracy: 0.5020
15840/25000 [==================>...........] - ETA: 28s - loss: 7.6376 - accuracy: 0.5019
15872/25000 [==================>...........] - ETA: 28s - loss: 7.6376 - accuracy: 0.5019
15904/25000 [==================>...........] - ETA: 28s - loss: 7.6387 - accuracy: 0.5018
15936/25000 [==================>...........] - ETA: 28s - loss: 7.6445 - accuracy: 0.5014
15968/25000 [==================>...........] - ETA: 28s - loss: 7.6465 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 28s - loss: 7.6436 - accuracy: 0.5015
16032/25000 [==================>...........] - ETA: 27s - loss: 7.6465 - accuracy: 0.5013
16064/25000 [==================>...........] - ETA: 27s - loss: 7.6408 - accuracy: 0.5017
16096/25000 [==================>...........] - ETA: 27s - loss: 7.6390 - accuracy: 0.5018
16128/25000 [==================>...........] - ETA: 27s - loss: 7.6390 - accuracy: 0.5018
16160/25000 [==================>...........] - ETA: 27s - loss: 7.6391 - accuracy: 0.5018
16192/25000 [==================>...........] - ETA: 27s - loss: 7.6429 - accuracy: 0.5015
16224/25000 [==================>...........] - ETA: 27s - loss: 7.6430 - accuracy: 0.5015
16256/25000 [==================>...........] - ETA: 27s - loss: 7.6440 - accuracy: 0.5015
16288/25000 [==================>...........] - ETA: 27s - loss: 7.6412 - accuracy: 0.5017
16320/25000 [==================>...........] - ETA: 27s - loss: 7.6403 - accuracy: 0.5017
16352/25000 [==================>...........] - ETA: 26s - loss: 7.6422 - accuracy: 0.5016
16384/25000 [==================>...........] - ETA: 26s - loss: 7.6451 - accuracy: 0.5014
16416/25000 [==================>...........] - ETA: 26s - loss: 7.6470 - accuracy: 0.5013
16448/25000 [==================>...........] - ETA: 26s - loss: 7.6470 - accuracy: 0.5013
16480/25000 [==================>...........] - ETA: 26s - loss: 7.6489 - accuracy: 0.5012
16512/25000 [==================>...........] - ETA: 26s - loss: 7.6480 - accuracy: 0.5012
16544/25000 [==================>...........] - ETA: 26s - loss: 7.6499 - accuracy: 0.5011
16576/25000 [==================>...........] - ETA: 26s - loss: 7.6509 - accuracy: 0.5010
16608/25000 [==================>...........] - ETA: 26s - loss: 7.6472 - accuracy: 0.5013
16640/25000 [==================>...........] - ETA: 26s - loss: 7.6500 - accuracy: 0.5011
16672/25000 [===================>..........] - ETA: 25s - loss: 7.6473 - accuracy: 0.5013
16704/25000 [===================>..........] - ETA: 25s - loss: 7.6483 - accuracy: 0.5012
16736/25000 [===================>..........] - ETA: 25s - loss: 7.6510 - accuracy: 0.5010
16768/25000 [===================>..........] - ETA: 25s - loss: 7.6511 - accuracy: 0.5010
16800/25000 [===================>..........] - ETA: 25s - loss: 7.6529 - accuracy: 0.5009
16832/25000 [===================>..........] - ETA: 25s - loss: 7.6530 - accuracy: 0.5009
16864/25000 [===================>..........] - ETA: 25s - loss: 7.6539 - accuracy: 0.5008
16896/25000 [===================>..........] - ETA: 25s - loss: 7.6548 - accuracy: 0.5008
16928/25000 [===================>..........] - ETA: 25s - loss: 7.6539 - accuracy: 0.5008
16960/25000 [===================>..........] - ETA: 25s - loss: 7.6558 - accuracy: 0.5007
16992/25000 [===================>..........] - ETA: 24s - loss: 7.6549 - accuracy: 0.5008
17024/25000 [===================>..........] - ETA: 24s - loss: 7.6531 - accuracy: 0.5009
17056/25000 [===================>..........] - ETA: 24s - loss: 7.6540 - accuracy: 0.5008
17088/25000 [===================>..........] - ETA: 24s - loss: 7.6532 - accuracy: 0.5009
17120/25000 [===================>..........] - ETA: 24s - loss: 7.6559 - accuracy: 0.5007
17152/25000 [===================>..........] - ETA: 24s - loss: 7.6568 - accuracy: 0.5006
17184/25000 [===================>..........] - ETA: 24s - loss: 7.6568 - accuracy: 0.5006
17216/25000 [===================>..........] - ETA: 24s - loss: 7.6568 - accuracy: 0.5006
17248/25000 [===================>..........] - ETA: 24s - loss: 7.6577 - accuracy: 0.5006
17280/25000 [===================>..........] - ETA: 24s - loss: 7.6560 - accuracy: 0.5007
17312/25000 [===================>..........] - ETA: 23s - loss: 7.6516 - accuracy: 0.5010
17344/25000 [===================>..........] - ETA: 23s - loss: 7.6454 - accuracy: 0.5014
17376/25000 [===================>..........] - ETA: 23s - loss: 7.6463 - accuracy: 0.5013
17408/25000 [===================>..........] - ETA: 23s - loss: 7.6472 - accuracy: 0.5013
17440/25000 [===================>..........] - ETA: 23s - loss: 7.6473 - accuracy: 0.5013
17472/25000 [===================>..........] - ETA: 23s - loss: 7.6429 - accuracy: 0.5015
17504/25000 [====================>.........] - ETA: 23s - loss: 7.6412 - accuracy: 0.5017
17536/25000 [====================>.........] - ETA: 23s - loss: 7.6404 - accuracy: 0.5017
17568/25000 [====================>.........] - ETA: 23s - loss: 7.6369 - accuracy: 0.5019
17600/25000 [====================>.........] - ETA: 23s - loss: 7.6353 - accuracy: 0.5020
17632/25000 [====================>.........] - ETA: 22s - loss: 7.6362 - accuracy: 0.5020
17664/25000 [====================>.........] - ETA: 22s - loss: 7.6397 - accuracy: 0.5018
17696/25000 [====================>.........] - ETA: 22s - loss: 7.6380 - accuracy: 0.5019
17728/25000 [====================>.........] - ETA: 22s - loss: 7.6398 - accuracy: 0.5017
17760/25000 [====================>.........] - ETA: 22s - loss: 7.6407 - accuracy: 0.5017
17792/25000 [====================>.........] - ETA: 22s - loss: 7.6347 - accuracy: 0.5021
17824/25000 [====================>.........] - ETA: 22s - loss: 7.6322 - accuracy: 0.5022
17856/25000 [====================>.........] - ETA: 22s - loss: 7.6331 - accuracy: 0.5022
17888/25000 [====================>.........] - ETA: 22s - loss: 7.6340 - accuracy: 0.5021
17920/25000 [====================>.........] - ETA: 22s - loss: 7.6350 - accuracy: 0.5021
17952/25000 [====================>.........] - ETA: 21s - loss: 7.6359 - accuracy: 0.5020
17984/25000 [====================>.........] - ETA: 21s - loss: 7.6376 - accuracy: 0.5019
18016/25000 [====================>.........] - ETA: 21s - loss: 7.6411 - accuracy: 0.5017
18048/25000 [====================>.........] - ETA: 21s - loss: 7.6394 - accuracy: 0.5018
18080/25000 [====================>.........] - ETA: 21s - loss: 7.6446 - accuracy: 0.5014
18112/25000 [====================>.........] - ETA: 21s - loss: 7.6480 - accuracy: 0.5012
18144/25000 [====================>.........] - ETA: 21s - loss: 7.6472 - accuracy: 0.5013
18176/25000 [====================>.........] - ETA: 21s - loss: 7.6438 - accuracy: 0.5015
18208/25000 [====================>.........] - ETA: 21s - loss: 7.6439 - accuracy: 0.5015
18240/25000 [====================>.........] - ETA: 21s - loss: 7.6414 - accuracy: 0.5016
18272/25000 [====================>.........] - ETA: 20s - loss: 7.6431 - accuracy: 0.5015
18304/25000 [====================>.........] - ETA: 20s - loss: 7.6415 - accuracy: 0.5016
18336/25000 [=====================>........] - ETA: 20s - loss: 7.6407 - accuracy: 0.5017
18368/25000 [=====================>........] - ETA: 20s - loss: 7.6424 - accuracy: 0.5016
18400/25000 [=====================>........] - ETA: 20s - loss: 7.6458 - accuracy: 0.5014
18432/25000 [=====================>........] - ETA: 20s - loss: 7.6425 - accuracy: 0.5016
18464/25000 [=====================>........] - ETA: 20s - loss: 7.6409 - accuracy: 0.5017
18496/25000 [=====================>........] - ETA: 20s - loss: 7.6476 - accuracy: 0.5012
18528/25000 [=====================>........] - ETA: 20s - loss: 7.6492 - accuracy: 0.5011
18560/25000 [=====================>........] - ETA: 20s - loss: 7.6451 - accuracy: 0.5014
18592/25000 [=====================>........] - ETA: 19s - loss: 7.6427 - accuracy: 0.5016
18624/25000 [=====================>........] - ETA: 19s - loss: 7.6411 - accuracy: 0.5017
18656/25000 [=====================>........] - ETA: 19s - loss: 7.6428 - accuracy: 0.5016
18688/25000 [=====================>........] - ETA: 19s - loss: 7.6412 - accuracy: 0.5017
18720/25000 [=====================>........] - ETA: 19s - loss: 7.6437 - accuracy: 0.5015
18752/25000 [=====================>........] - ETA: 19s - loss: 7.6421 - accuracy: 0.5016
18784/25000 [=====================>........] - ETA: 19s - loss: 7.6413 - accuracy: 0.5017
18816/25000 [=====================>........] - ETA: 19s - loss: 7.6389 - accuracy: 0.5018
18848/25000 [=====================>........] - ETA: 19s - loss: 7.6373 - accuracy: 0.5019
18880/25000 [=====================>........] - ETA: 19s - loss: 7.6333 - accuracy: 0.5022
18912/25000 [=====================>........] - ETA: 18s - loss: 7.6318 - accuracy: 0.5023
18944/25000 [=====================>........] - ETA: 18s - loss: 7.6342 - accuracy: 0.5021
18976/25000 [=====================>........] - ETA: 18s - loss: 7.6359 - accuracy: 0.5020
19008/25000 [=====================>........] - ETA: 18s - loss: 7.6384 - accuracy: 0.5018
19040/25000 [=====================>........] - ETA: 18s - loss: 7.6352 - accuracy: 0.5020
19072/25000 [=====================>........] - ETA: 18s - loss: 7.6312 - accuracy: 0.5023
19104/25000 [=====================>........] - ETA: 18s - loss: 7.6321 - accuracy: 0.5023
19136/25000 [=====================>........] - ETA: 18s - loss: 7.6306 - accuracy: 0.5024
19168/25000 [======================>.......] - ETA: 18s - loss: 7.6298 - accuracy: 0.5024
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6323 - accuracy: 0.5022
19232/25000 [======================>.......] - ETA: 17s - loss: 7.6315 - accuracy: 0.5023
19264/25000 [======================>.......] - ETA: 17s - loss: 7.6380 - accuracy: 0.5019
19296/25000 [======================>.......] - ETA: 17s - loss: 7.6380 - accuracy: 0.5019
19328/25000 [======================>.......] - ETA: 17s - loss: 7.6373 - accuracy: 0.5019
19360/25000 [======================>.......] - ETA: 17s - loss: 7.6365 - accuracy: 0.5020
19392/25000 [======================>.......] - ETA: 17s - loss: 7.6342 - accuracy: 0.5021
19424/25000 [======================>.......] - ETA: 17s - loss: 7.6358 - accuracy: 0.5020
19456/25000 [======================>.......] - ETA: 17s - loss: 7.6367 - accuracy: 0.5020
19488/25000 [======================>.......] - ETA: 17s - loss: 7.6351 - accuracy: 0.5021
19520/25000 [======================>.......] - ETA: 17s - loss: 7.6391 - accuracy: 0.5018
19552/25000 [======================>.......] - ETA: 16s - loss: 7.6407 - accuracy: 0.5017
19584/25000 [======================>.......] - ETA: 16s - loss: 7.6455 - accuracy: 0.5014
19616/25000 [======================>.......] - ETA: 16s - loss: 7.6463 - accuracy: 0.5013
19648/25000 [======================>.......] - ETA: 16s - loss: 7.6432 - accuracy: 0.5015
19680/25000 [======================>.......] - ETA: 16s - loss: 7.6456 - accuracy: 0.5014
19712/25000 [======================>.......] - ETA: 16s - loss: 7.6456 - accuracy: 0.5014
19744/25000 [======================>.......] - ETA: 16s - loss: 7.6511 - accuracy: 0.5010
19776/25000 [======================>.......] - ETA: 16s - loss: 7.6519 - accuracy: 0.5010
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6511 - accuracy: 0.5010
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6527 - accuracy: 0.5009
19872/25000 [======================>.......] - ETA: 15s - loss: 7.6558 - accuracy: 0.5007
19904/25000 [======================>.......] - ETA: 15s - loss: 7.6566 - accuracy: 0.5007
19936/25000 [======================>.......] - ETA: 15s - loss: 7.6589 - accuracy: 0.5005
19968/25000 [======================>.......] - ETA: 15s - loss: 7.6612 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 15s - loss: 7.6605 - accuracy: 0.5004
20032/25000 [=======================>......] - ETA: 15s - loss: 7.6567 - accuracy: 0.5006
20064/25000 [=======================>......] - ETA: 15s - loss: 7.6574 - accuracy: 0.5006
20096/25000 [=======================>......] - ETA: 15s - loss: 7.6552 - accuracy: 0.5007
20128/25000 [=======================>......] - ETA: 15s - loss: 7.6567 - accuracy: 0.5006
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6575 - accuracy: 0.5006
20192/25000 [=======================>......] - ETA: 14s - loss: 7.6590 - accuracy: 0.5005
20224/25000 [=======================>......] - ETA: 14s - loss: 7.6621 - accuracy: 0.5003
20256/25000 [=======================>......] - ETA: 14s - loss: 7.6583 - accuracy: 0.5005
20288/25000 [=======================>......] - ETA: 14s - loss: 7.6591 - accuracy: 0.5005
20320/25000 [=======================>......] - ETA: 14s - loss: 7.6576 - accuracy: 0.5006
20352/25000 [=======================>......] - ETA: 14s - loss: 7.6546 - accuracy: 0.5008
20384/25000 [=======================>......] - ETA: 14s - loss: 7.6583 - accuracy: 0.5005
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6584 - accuracy: 0.5005
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6554 - accuracy: 0.5007
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6561 - accuracy: 0.5007
20512/25000 [=======================>......] - ETA: 13s - loss: 7.6554 - accuracy: 0.5007
20544/25000 [=======================>......] - ETA: 13s - loss: 7.6592 - accuracy: 0.5005
20576/25000 [=======================>......] - ETA: 13s - loss: 7.6592 - accuracy: 0.5005
20608/25000 [=======================>......] - ETA: 13s - loss: 7.6599 - accuracy: 0.5004
20640/25000 [=======================>......] - ETA: 13s - loss: 7.6599 - accuracy: 0.5004
20672/25000 [=======================>......] - ETA: 13s - loss: 7.6570 - accuracy: 0.5006
20704/25000 [=======================>......] - ETA: 13s - loss: 7.6577 - accuracy: 0.5006
20736/25000 [=======================>......] - ETA: 13s - loss: 7.6577 - accuracy: 0.5006
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6570 - accuracy: 0.5006
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6578 - accuracy: 0.5006
20832/25000 [=======================>......] - ETA: 12s - loss: 7.6585 - accuracy: 0.5005
20864/25000 [========================>.....] - ETA: 12s - loss: 7.6593 - accuracy: 0.5005
20896/25000 [========================>.....] - ETA: 12s - loss: 7.6585 - accuracy: 0.5005
20928/25000 [========================>.....] - ETA: 12s - loss: 7.6556 - accuracy: 0.5007
20960/25000 [========================>.....] - ETA: 12s - loss: 7.6593 - accuracy: 0.5005
20992/25000 [========================>.....] - ETA: 12s - loss: 7.6600 - accuracy: 0.5004
21024/25000 [========================>.....] - ETA: 12s - loss: 7.6593 - accuracy: 0.5005
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6579 - accuracy: 0.5006
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6593 - accuracy: 0.5005
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6623 - accuracy: 0.5003
21152/25000 [========================>.....] - ETA: 11s - loss: 7.6608 - accuracy: 0.5004
21184/25000 [========================>.....] - ETA: 11s - loss: 7.6608 - accuracy: 0.5004
21216/25000 [========================>.....] - ETA: 11s - loss: 7.6616 - accuracy: 0.5003
21248/25000 [========================>.....] - ETA: 11s - loss: 7.6608 - accuracy: 0.5004
21280/25000 [========================>.....] - ETA: 11s - loss: 7.6623 - accuracy: 0.5003
21312/25000 [========================>.....] - ETA: 11s - loss: 7.6609 - accuracy: 0.5004
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6680 - accuracy: 0.4999
21472/25000 [========================>.....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
21504/25000 [========================>.....] - ETA: 10s - loss: 7.6680 - accuracy: 0.4999
21536/25000 [========================>.....] - ETA: 10s - loss: 7.6680 - accuracy: 0.4999
21568/25000 [========================>.....] - ETA: 10s - loss: 7.6702 - accuracy: 0.4998
21600/25000 [========================>.....] - ETA: 10s - loss: 7.6709 - accuracy: 0.4997
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6716 - accuracy: 0.4997
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6680 - accuracy: 0.4999
21792/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999 
21824/25000 [=========================>....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21856/25000 [=========================>....] - ETA: 9s - loss: 7.6631 - accuracy: 0.5002
21888/25000 [=========================>....] - ETA: 9s - loss: 7.6631 - accuracy: 0.5002
21920/25000 [=========================>....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
22112/25000 [=========================>....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
22144/25000 [=========================>....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
22176/25000 [=========================>....] - ETA: 8s - loss: 7.6639 - accuracy: 0.5002
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6625 - accuracy: 0.5003
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
22496/25000 [=========================>....] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6721 - accuracy: 0.4996
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6734 - accuracy: 0.4996
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6734 - accuracy: 0.4996
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6741 - accuracy: 0.4995
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6714 - accuracy: 0.4997
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6707 - accuracy: 0.4997
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6713 - accuracy: 0.4997
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6727 - accuracy: 0.4996
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6686 - accuracy: 0.4999
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6680 - accuracy: 0.4999
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6619 - accuracy: 0.5003
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6600 - accuracy: 0.5004
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6573 - accuracy: 0.5006
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6547 - accuracy: 0.5008
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6587 - accuracy: 0.5005
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6547 - accuracy: 0.5008
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6540 - accuracy: 0.5008
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6534 - accuracy: 0.5009
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6587 - accuracy: 0.5005
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6587 - accuracy: 0.5005
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6627 - accuracy: 0.5003
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6601 - accuracy: 0.5004
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5003
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24288/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24320/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24352/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24384/25000 [============================>.] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
24416/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24448/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 1s - loss: 7.6623 - accuracy: 0.5003
24704/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24768/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 93s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
