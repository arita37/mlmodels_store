
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:49<01:14, 24.96s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.33291991824194117, 'embedding_size_factor': 0.7953041989044733, 'layers.choice': 3, 'learning_rate': 0.004200837359177582, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.03937900688605826} and reward: 0.3488
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5N\x8fXBO\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9s!\xca\x95QrX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?q4\xe5\xb1,\xfc7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa4)|5r\xfc\x06u.' and reward: 0.3488
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5N\x8fXBO\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9s!\xca\x95QrX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?q4\xe5\xb1,\xfc7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa4)|5r\xfc\x06u.' and reward: 0.3488
 60%|██████    | 3/5 [01:41<01:06, 33.06s/it] 60%|██████    | 3/5 [01:41<01:07, 33.96s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.20714680413818315, 'embedding_size_factor': 1.3141721295337427, 'layers.choice': 0, 'learning_rate': 0.0002040218166401396, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.3676334776490086e-08} and reward: 0.353
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xca\x83\xc9V\x9fH\xf6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x06\xd9Z\xda\x97\xf7X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?*\xbd\xd6\x0fn\x7f\x80X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>M^\xa5\x01\xa0$\xa7u.' and reward: 0.353
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xca\x83\xc9V\x9fH\xf6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x06\xd9Z\xda\x97\xf7X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?*\xbd\xd6\x0fn\x7f\x80X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>M^\xa5\x01\xa0$\xa7u.' and reward: 0.353
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 152.3158519268036
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -34.63s of remaining time.
Ensemble size: 38
Ensemble weights: 
[0.65789474 0.07894737 0.26315789]
	0.3908	 = Validation accuracy score
	1.01s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 155.68s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f75a93d6ac8> 

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
 [-0.09415824  0.04383673 -0.02600401  0.09493946  0.14470984 -0.12818807]
 [ 0.16309322  0.20194745 -0.17679085  0.0919519   0.13034458 -0.0703012 ]
 [ 0.12427807 -0.14149302  0.05455852  0.05044635  0.29611686 -0.06520549]
 [ 0.02541232  0.11256387  0.28228176 -0.11665554  0.06786072 -0.31343684]
 [-0.47052741 -0.11882485 -0.51869148  0.4336766   0.27419096 -0.24492539]
 [ 0.31709617  0.29331687 -0.25538427  0.13167764 -0.32781312 -0.56178045]
 [ 0.1365803  -0.40898234 -0.10232418  0.24761017  0.11011024  0.10636975]
 [-0.05906157 -0.305879   -0.29678291 -0.35948834 -0.18362132  0.10806257]
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
{'loss': 0.4639723412692547, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 00:24:05.670238: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5800758227705956, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 00:24:06.779640: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1482752/17464789 [=>............................] - ETA: 0s
 4702208/17464789 [=======>......................] - ETA: 0s
 9732096/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-18 00:24:18.021944: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-18 00:24:18.026204: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-05-18 00:24:18.026334: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558b06ad7320 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 00:24:18.026345: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:40 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:49 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:12 - loss: 6.5486 - accuracy: 0.5729
  128/25000 [..............................] - ETA: 1:54 - loss: 6.9479 - accuracy: 0.5469
  160/25000 [..............................] - ETA: 1:42 - loss: 7.3791 - accuracy: 0.5188
  192/25000 [..............................] - ETA: 1:35 - loss: 7.5868 - accuracy: 0.5052
  224/25000 [..............................] - ETA: 1:30 - loss: 7.3928 - accuracy: 0.5179
  256/25000 [..............................] - ETA: 1:26 - loss: 7.3671 - accuracy: 0.5195
  288/25000 [..............................] - ETA: 1:23 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:21 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:19 - loss: 7.6231 - accuracy: 0.5028
  384/25000 [..............................] - ETA: 1:17 - loss: 7.7065 - accuracy: 0.4974
  416/25000 [..............................] - ETA: 1:16 - loss: 7.6298 - accuracy: 0.5024
  448/25000 [..............................] - ETA: 1:15 - loss: 7.6324 - accuracy: 0.5022
  480/25000 [..............................] - ETA: 1:14 - loss: 7.7944 - accuracy: 0.4917
  512/25000 [..............................] - ETA: 1:13 - loss: 7.8463 - accuracy: 0.4883
  544/25000 [..............................] - ETA: 1:12 - loss: 7.7794 - accuracy: 0.4926
  576/25000 [..............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
  608/25000 [..............................] - ETA: 1:11 - loss: 7.6414 - accuracy: 0.5016
  640/25000 [..............................] - ETA: 1:10 - loss: 7.7625 - accuracy: 0.4938
  672/25000 [..............................] - ETA: 1:10 - loss: 7.8492 - accuracy: 0.4881
  704/25000 [..............................] - ETA: 1:09 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:08 - loss: 7.8125 - accuracy: 0.4905
  768/25000 [..............................] - ETA: 1:08 - loss: 7.7265 - accuracy: 0.4961
  800/25000 [..............................] - ETA: 1:08 - loss: 7.8008 - accuracy: 0.4913
  832/25000 [..............................] - ETA: 1:07 - loss: 7.8325 - accuracy: 0.4892
  864/25000 [>.............................] - ETA: 1:07 - loss: 7.8796 - accuracy: 0.4861
  896/25000 [>.............................] - ETA: 1:07 - loss: 7.9575 - accuracy: 0.4810
  928/25000 [>.............................] - ETA: 1:06 - loss: 7.8979 - accuracy: 0.4849
  960/25000 [>.............................] - ETA: 1:06 - loss: 7.8263 - accuracy: 0.4896
  992/25000 [>.............................] - ETA: 1:06 - loss: 7.8366 - accuracy: 0.4889
 1024/25000 [>.............................] - ETA: 1:05 - loss: 7.8763 - accuracy: 0.4863
 1056/25000 [>.............................] - ETA: 1:05 - loss: 7.9715 - accuracy: 0.4801
 1088/25000 [>.............................] - ETA: 1:05 - loss: 7.8780 - accuracy: 0.4862
 1120/25000 [>.............................] - ETA: 1:05 - loss: 7.8994 - accuracy: 0.4848
 1152/25000 [>.............................] - ETA: 1:04 - loss: 7.8929 - accuracy: 0.4852
 1184/25000 [>.............................] - ETA: 1:04 - loss: 7.9386 - accuracy: 0.4823
 1216/25000 [>.............................] - ETA: 1:04 - loss: 7.9314 - accuracy: 0.4827
 1248/25000 [>.............................] - ETA: 1:04 - loss: 7.9983 - accuracy: 0.4784
 1280/25000 [>.............................] - ETA: 1:03 - loss: 8.0260 - accuracy: 0.4766
 1312/25000 [>.............................] - ETA: 1:03 - loss: 8.1458 - accuracy: 0.4688
 1344/25000 [>.............................] - ETA: 1:03 - loss: 8.1800 - accuracy: 0.4665
 1376/25000 [>.............................] - ETA: 1:03 - loss: 8.1904 - accuracy: 0.4658
 1408/25000 [>.............................] - ETA: 1:03 - loss: 8.2111 - accuracy: 0.4645
 1440/25000 [>.............................] - ETA: 1:02 - loss: 8.2310 - accuracy: 0.4632
 1472/25000 [>.............................] - ETA: 1:02 - loss: 8.2395 - accuracy: 0.4626
 1504/25000 [>.............................] - ETA: 1:02 - loss: 8.2273 - accuracy: 0.4634
 1536/25000 [>.............................] - ETA: 1:02 - loss: 8.2456 - accuracy: 0.4622
 1568/25000 [>.............................] - ETA: 1:02 - loss: 8.2534 - accuracy: 0.4617
 1600/25000 [>.............................] - ETA: 1:02 - loss: 8.2799 - accuracy: 0.4600
 1632/25000 [>.............................] - ETA: 1:02 - loss: 8.2679 - accuracy: 0.4608
 1664/25000 [>.............................] - ETA: 1:01 - loss: 8.2471 - accuracy: 0.4621
 1696/25000 [=>............................] - ETA: 1:01 - loss: 8.1819 - accuracy: 0.4664
 1728/25000 [=>............................] - ETA: 1:01 - loss: 8.1902 - accuracy: 0.4659
 1760/25000 [=>............................] - ETA: 1:01 - loss: 8.1893 - accuracy: 0.4659
 1792/25000 [=>............................] - ETA: 1:01 - loss: 8.1458 - accuracy: 0.4688
 1824/25000 [=>............................] - ETA: 1:01 - loss: 8.1374 - accuracy: 0.4693
 1856/25000 [=>............................] - ETA: 1:01 - loss: 8.1540 - accuracy: 0.4682
 1888/25000 [=>............................] - ETA: 1:01 - loss: 8.1377 - accuracy: 0.4693
 1920/25000 [=>............................] - ETA: 1:00 - loss: 8.1138 - accuracy: 0.4708
 1952/25000 [=>............................] - ETA: 1:00 - loss: 8.0672 - accuracy: 0.4739
 1984/25000 [=>............................] - ETA: 1:00 - loss: 8.0530 - accuracy: 0.4748
 2016/25000 [=>............................] - ETA: 1:00 - loss: 8.0241 - accuracy: 0.4767
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.9886 - accuracy: 0.4790
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.9910 - accuracy: 0.4788
 2112/25000 [=>............................] - ETA: 59s - loss: 7.9788 - accuracy: 0.4796 
 2144/25000 [=>............................] - ETA: 59s - loss: 7.9741 - accuracy: 0.4799
 2176/25000 [=>............................] - ETA: 59s - loss: 7.9837 - accuracy: 0.4793
 2208/25000 [=>............................] - ETA: 59s - loss: 7.9791 - accuracy: 0.4796
 2240/25000 [=>............................] - ETA: 59s - loss: 7.9678 - accuracy: 0.4804
 2272/25000 [=>............................] - ETA: 59s - loss: 7.9636 - accuracy: 0.4806
 2304/25000 [=>............................] - ETA: 59s - loss: 7.9461 - accuracy: 0.4818
 2336/25000 [=>............................] - ETA: 58s - loss: 7.9292 - accuracy: 0.4829
 2368/25000 [=>............................] - ETA: 58s - loss: 7.9515 - accuracy: 0.4814
 2400/25000 [=>............................] - ETA: 58s - loss: 7.9541 - accuracy: 0.4812
 2432/25000 [=>............................] - ETA: 58s - loss: 7.9629 - accuracy: 0.4807
 2464/25000 [=>............................] - ETA: 58s - loss: 7.9280 - accuracy: 0.4830
 2496/25000 [=>............................] - ETA: 58s - loss: 7.9431 - accuracy: 0.4820
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.9274 - accuracy: 0.4830
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.9421 - accuracy: 0.4820
 2592/25000 [==>...........................] - ETA: 57s - loss: 7.9565 - accuracy: 0.4811
 2624/25000 [==>...........................] - ETA: 57s - loss: 7.9413 - accuracy: 0.4821
 2656/25000 [==>...........................] - ETA: 57s - loss: 7.9380 - accuracy: 0.4823
 2688/25000 [==>...........................] - ETA: 57s - loss: 7.9461 - accuracy: 0.4818
 2720/25000 [==>...........................] - ETA: 57s - loss: 7.9372 - accuracy: 0.4824
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.9173 - accuracy: 0.4836
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.8869 - accuracy: 0.4856
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.8844 - accuracy: 0.4858
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.8820 - accuracy: 0.4860
 2880/25000 [==>...........................] - ETA: 56s - loss: 7.8796 - accuracy: 0.4861
 2912/25000 [==>...........................] - ETA: 56s - loss: 7.8878 - accuracy: 0.4856
 2944/25000 [==>...........................] - ETA: 56s - loss: 7.8906 - accuracy: 0.4854
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.8882 - accuracy: 0.4856
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.8858 - accuracy: 0.4857
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.8986 - accuracy: 0.4849
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.9062 - accuracy: 0.4844
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.9037 - accuracy: 0.4845
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.9209 - accuracy: 0.4834
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.9280 - accuracy: 0.4830
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.9302 - accuracy: 0.4828
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.9513 - accuracy: 0.4814
 3264/25000 [==>...........................] - ETA: 55s - loss: 7.9532 - accuracy: 0.4813
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.9411 - accuracy: 0.4821
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.9246 - accuracy: 0.4832
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.9085 - accuracy: 0.4842
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.9017 - accuracy: 0.4847
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.9174 - accuracy: 0.4836
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.9373 - accuracy: 0.4823
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.9436 - accuracy: 0.4819
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.9323 - accuracy: 0.4827
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.9299 - accuracy: 0.4828
 3584/25000 [===>..........................] - ETA: 54s - loss: 7.9276 - accuracy: 0.4830
 3616/25000 [===>..........................] - ETA: 54s - loss: 7.9338 - accuracy: 0.4826
 3648/25000 [===>..........................] - ETA: 54s - loss: 7.9356 - accuracy: 0.4825
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.9375 - accuracy: 0.4823
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.9145 - accuracy: 0.4838
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.9123 - accuracy: 0.4840
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.9103 - accuracy: 0.4841
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.9163 - accuracy: 0.4837
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.8982 - accuracy: 0.4849
 3872/25000 [===>..........................] - ETA: 53s - loss: 7.9240 - accuracy: 0.4832
 3904/25000 [===>..........................] - ETA: 53s - loss: 7.9141 - accuracy: 0.4839
 3936/25000 [===>..........................] - ETA: 53s - loss: 7.9237 - accuracy: 0.4832
 3968/25000 [===>..........................] - ETA: 53s - loss: 7.9139 - accuracy: 0.4839
 4000/25000 [===>..........................] - ETA: 53s - loss: 7.9005 - accuracy: 0.4848
 4032/25000 [===>..........................] - ETA: 53s - loss: 7.9062 - accuracy: 0.4844
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.9119 - accuracy: 0.4840
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.9137 - accuracy: 0.4839
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.8932 - accuracy: 0.4852
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.8693 - accuracy: 0.4868
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.8641 - accuracy: 0.4871
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.8699 - accuracy: 0.4867
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.8792 - accuracy: 0.4861
 4288/25000 [====>.........................] - ETA: 52s - loss: 7.8740 - accuracy: 0.4865
 4320/25000 [====>.........................] - ETA: 52s - loss: 7.8725 - accuracy: 0.4866
 4352/25000 [====>.........................] - ETA: 52s - loss: 7.8674 - accuracy: 0.4869
 4384/25000 [====>.........................] - ETA: 52s - loss: 7.8555 - accuracy: 0.4877
 4416/25000 [====>.........................] - ETA: 52s - loss: 7.8576 - accuracy: 0.4875
 4448/25000 [====>.........................] - ETA: 52s - loss: 7.8528 - accuracy: 0.4879
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.8686 - accuracy: 0.4868
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.8603 - accuracy: 0.4874
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.8758 - accuracy: 0.4864
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.8811 - accuracy: 0.4860
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.8829 - accuracy: 0.4859
 4640/25000 [====>.........................] - ETA: 51s - loss: 7.8913 - accuracy: 0.4853
 4672/25000 [====>.........................] - ETA: 51s - loss: 7.8832 - accuracy: 0.4859
 4704/25000 [====>.........................] - ETA: 51s - loss: 7.8687 - accuracy: 0.4868
 4736/25000 [====>.........................] - ETA: 51s - loss: 7.8738 - accuracy: 0.4865
 4768/25000 [====>.........................] - ETA: 51s - loss: 7.8467 - accuracy: 0.4883
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.8679 - accuracy: 0.4869
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.8665 - accuracy: 0.4870
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.8463 - accuracy: 0.4883
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.8326 - accuracy: 0.4892
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.8346 - accuracy: 0.4890
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.8397 - accuracy: 0.4887
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.8448 - accuracy: 0.4884
 5024/25000 [=====>........................] - ETA: 50s - loss: 7.8375 - accuracy: 0.4889
 5056/25000 [=====>........................] - ETA: 50s - loss: 7.8334 - accuracy: 0.4891
 5088/25000 [=====>........................] - ETA: 50s - loss: 7.8565 - accuracy: 0.4876
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.8523 - accuracy: 0.4879
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.8482 - accuracy: 0.4882
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.8352 - accuracy: 0.4890
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.8371 - accuracy: 0.4889
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.8448 - accuracy: 0.4884
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.8467 - accuracy: 0.4883
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.8514 - accuracy: 0.4880
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.8388 - accuracy: 0.4888
 5376/25000 [=====>........................] - ETA: 49s - loss: 7.8377 - accuracy: 0.4888
 5408/25000 [=====>........................] - ETA: 49s - loss: 7.8396 - accuracy: 0.4887
 5440/25000 [=====>........................] - ETA: 49s - loss: 7.8526 - accuracy: 0.4879
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.8572 - accuracy: 0.4876
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.8477 - accuracy: 0.4882
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.8494 - accuracy: 0.4881
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.8594 - accuracy: 0.4874
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.8528 - accuracy: 0.4879
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.8381 - accuracy: 0.4888
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.8426 - accuracy: 0.4885
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.8335 - accuracy: 0.4891
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.8433 - accuracy: 0.4885
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.8423 - accuracy: 0.4885
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.8413 - accuracy: 0.4886
 5824/25000 [=====>........................] - ETA: 48s - loss: 7.8351 - accuracy: 0.4890
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.8237 - accuracy: 0.4898
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.8307 - accuracy: 0.4893
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.8453 - accuracy: 0.4883
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.8366 - accuracy: 0.4889
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.8511 - accuracy: 0.4880
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.8603 - accuracy: 0.4874
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.8517 - accuracy: 0.4879
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.8457 - accuracy: 0.4883
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.8447 - accuracy: 0.4884
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.8363 - accuracy: 0.4889
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.8255 - accuracy: 0.4896
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.8247 - accuracy: 0.4897
 6240/25000 [======>.......................] - ETA: 47s - loss: 7.8239 - accuracy: 0.4897
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.8182 - accuracy: 0.4901
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.8272 - accuracy: 0.4895
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.8360 - accuracy: 0.4890
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.8376 - accuracy: 0.4889
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.8176 - accuracy: 0.4902
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.8192 - accuracy: 0.4900
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.8256 - accuracy: 0.4896
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.8248 - accuracy: 0.4897
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.8263 - accuracy: 0.4896
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.8302 - accuracy: 0.4893
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.8201 - accuracy: 0.4900
 6624/25000 [======>.......................] - ETA: 46s - loss: 7.8148 - accuracy: 0.4903
 6656/25000 [======>.......................] - ETA: 46s - loss: 7.8141 - accuracy: 0.4904
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.8202 - accuracy: 0.4900
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.8127 - accuracy: 0.4905
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.8097 - accuracy: 0.4907
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.8181 - accuracy: 0.4901
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.8106 - accuracy: 0.4906
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.8122 - accuracy: 0.4905
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.8115 - accuracy: 0.4906
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.8064 - accuracy: 0.4909
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.7925 - accuracy: 0.4918
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.7875 - accuracy: 0.4921
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.7891 - accuracy: 0.4920
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.7799 - accuracy: 0.4926
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.7902 - accuracy: 0.4919
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.7961 - accuracy: 0.4916
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.7977 - accuracy: 0.4915
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.7971 - accuracy: 0.4915
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.7965 - accuracy: 0.4915
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.7875 - accuracy: 0.4921
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.8017 - accuracy: 0.4912
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.8158 - accuracy: 0.4903
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.8236 - accuracy: 0.4898
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.8270 - accuracy: 0.4895
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.8243 - accuracy: 0.4897
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.8195 - accuracy: 0.4900
 7456/25000 [=======>......................] - ETA: 44s - loss: 7.8126 - accuracy: 0.4905
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.8079 - accuracy: 0.4908
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.8073 - accuracy: 0.4908
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.8047 - accuracy: 0.4910
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.8041 - accuracy: 0.4910
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.8075 - accuracy: 0.4908
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.8030 - accuracy: 0.4911
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.8004 - accuracy: 0.4913
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.8018 - accuracy: 0.4912
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.7953 - accuracy: 0.4916
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.7987 - accuracy: 0.4914
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.8002 - accuracy: 0.4913
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.7996 - accuracy: 0.4913
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.8010 - accuracy: 0.4912
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.8024 - accuracy: 0.4911
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.7961 - accuracy: 0.4916
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.7994 - accuracy: 0.4913
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.7950 - accuracy: 0.4916
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.7964 - accuracy: 0.4915
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.7921 - accuracy: 0.4918
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.7916 - accuracy: 0.4918
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.7855 - accuracy: 0.4922
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.7869 - accuracy: 0.4922
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.7958 - accuracy: 0.4916
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.7971 - accuracy: 0.4915
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.7948 - accuracy: 0.4916
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.7980 - accuracy: 0.4914
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.8012 - accuracy: 0.4912
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.7915 - accuracy: 0.4919
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.7855 - accuracy: 0.4922
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.7832 - accuracy: 0.4924
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.7792 - accuracy: 0.4927
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.7769 - accuracy: 0.4928
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.7729 - accuracy: 0.4931
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.7725 - accuracy: 0.4931
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.7757 - accuracy: 0.4929
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.7771 - accuracy: 0.4928
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.7855 - accuracy: 0.4922
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.7886 - accuracy: 0.4920
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7987 - accuracy: 0.4914
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.7965 - accuracy: 0.4915
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.7943 - accuracy: 0.4917
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.7921 - accuracy: 0.4918
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.7847 - accuracy: 0.4923
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.7946 - accuracy: 0.4917
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.7924 - accuracy: 0.4918
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.7800 - accuracy: 0.4926
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.7779 - accuracy: 0.4927
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.7792 - accuracy: 0.4927
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.7754 - accuracy: 0.4929
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.7682 - accuracy: 0.4934
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.7594 - accuracy: 0.4939
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.7591 - accuracy: 0.4940
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.7588 - accuracy: 0.4940
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.7568 - accuracy: 0.4941
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.7615 - accuracy: 0.4938
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.7595 - accuracy: 0.4939
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.7525 - accuracy: 0.4944
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.7506 - accuracy: 0.4945
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.7503 - accuracy: 0.4945
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.7484 - accuracy: 0.4947
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7465 - accuracy: 0.4948
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7495 - accuracy: 0.4946
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7540 - accuracy: 0.4943
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7586 - accuracy: 0.4940
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7583 - accuracy: 0.4940
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7548 - accuracy: 0.4943
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.7640 - accuracy: 0.4936
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.7637 - accuracy: 0.4937
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.7634 - accuracy: 0.4937
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.7631 - accuracy: 0.4937
 9728/25000 [==========>...................] - ETA: 38s - loss: 7.7612 - accuracy: 0.4938
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.7609 - accuracy: 0.4939
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.7543 - accuracy: 0.4943
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.7493 - accuracy: 0.4946
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7475 - accuracy: 0.4947
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7457 - accuracy: 0.4948
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.7377 - accuracy: 0.4954
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.7467 - accuracy: 0.4948
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.7419 - accuracy: 0.4951
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7447 - accuracy: 0.4949
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7444 - accuracy: 0.4949
10080/25000 [===========>..................] - ETA: 37s - loss: 7.7457 - accuracy: 0.4948
10112/25000 [===========>..................] - ETA: 37s - loss: 7.7470 - accuracy: 0.4948
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7467 - accuracy: 0.4948
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7435 - accuracy: 0.4950
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7357 - accuracy: 0.4955
10240/25000 [===========>..................] - ETA: 37s - loss: 7.7370 - accuracy: 0.4954
10272/25000 [===========>..................] - ETA: 37s - loss: 7.7398 - accuracy: 0.4952
10304/25000 [===========>..................] - ETA: 37s - loss: 7.7410 - accuracy: 0.4951
10336/25000 [===========>..................] - ETA: 37s - loss: 7.7512 - accuracy: 0.4945
10368/25000 [===========>..................] - ETA: 37s - loss: 7.7494 - accuracy: 0.4946
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7477 - accuracy: 0.4947
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7460 - accuracy: 0.4948
10464/25000 [===========>..................] - ETA: 36s - loss: 7.7516 - accuracy: 0.4945
10496/25000 [===========>..................] - ETA: 36s - loss: 7.7543 - accuracy: 0.4943
10528/25000 [===========>..................] - ETA: 36s - loss: 7.7482 - accuracy: 0.4947
10560/25000 [===========>..................] - ETA: 36s - loss: 7.7421 - accuracy: 0.4951
10592/25000 [===========>..................] - ETA: 36s - loss: 7.7390 - accuracy: 0.4953
10624/25000 [===========>..................] - ETA: 36s - loss: 7.7373 - accuracy: 0.4954
10656/25000 [===========>..................] - ETA: 36s - loss: 7.7371 - accuracy: 0.4954
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7384 - accuracy: 0.4953
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7296 - accuracy: 0.4959
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7336 - accuracy: 0.4956
10784/25000 [===========>..................] - ETA: 36s - loss: 7.7334 - accuracy: 0.4956
10816/25000 [===========>..................] - ETA: 36s - loss: 7.7347 - accuracy: 0.4956
10848/25000 [============>.................] - ETA: 35s - loss: 7.7373 - accuracy: 0.4954
10880/25000 [============>.................] - ETA: 35s - loss: 7.7371 - accuracy: 0.4954
10912/25000 [============>.................] - ETA: 35s - loss: 7.7355 - accuracy: 0.4955
10944/25000 [============>.................] - ETA: 35s - loss: 7.7353 - accuracy: 0.4955
10976/25000 [============>.................] - ETA: 35s - loss: 7.7337 - accuracy: 0.4956
11008/25000 [============>.................] - ETA: 35s - loss: 7.7265 - accuracy: 0.4961
11040/25000 [============>.................] - ETA: 35s - loss: 7.7208 - accuracy: 0.4965
11072/25000 [============>.................] - ETA: 35s - loss: 7.7234 - accuracy: 0.4963
11104/25000 [============>.................] - ETA: 35s - loss: 7.7260 - accuracy: 0.4961
11136/25000 [============>.................] - ETA: 35s - loss: 7.7217 - accuracy: 0.4964
11168/25000 [============>.................] - ETA: 35s - loss: 7.7243 - accuracy: 0.4962
11200/25000 [============>.................] - ETA: 34s - loss: 7.7269 - accuracy: 0.4961
11232/25000 [============>.................] - ETA: 34s - loss: 7.7294 - accuracy: 0.4959
11264/25000 [============>.................] - ETA: 34s - loss: 7.7320 - accuracy: 0.4957
11296/25000 [============>.................] - ETA: 34s - loss: 7.7291 - accuracy: 0.4959
11328/25000 [============>.................] - ETA: 34s - loss: 7.7289 - accuracy: 0.4959
11360/25000 [============>.................] - ETA: 34s - loss: 7.7233 - accuracy: 0.4963
11392/25000 [============>.................] - ETA: 34s - loss: 7.7191 - accuracy: 0.4966
11424/25000 [============>.................] - ETA: 34s - loss: 7.7176 - accuracy: 0.4967
11456/25000 [============>.................] - ETA: 34s - loss: 7.7202 - accuracy: 0.4965
11488/25000 [============>.................] - ETA: 34s - loss: 7.7187 - accuracy: 0.4966
11520/25000 [============>.................] - ETA: 34s - loss: 7.7199 - accuracy: 0.4965
11552/25000 [============>.................] - ETA: 34s - loss: 7.7224 - accuracy: 0.4964
11584/25000 [============>.................] - ETA: 34s - loss: 7.7275 - accuracy: 0.4960
11616/25000 [============>.................] - ETA: 33s - loss: 7.7273 - accuracy: 0.4960
11648/25000 [============>.................] - ETA: 33s - loss: 7.7285 - accuracy: 0.4960
11680/25000 [=============>................] - ETA: 33s - loss: 7.7283 - accuracy: 0.4960
11712/25000 [=============>................] - ETA: 33s - loss: 7.7268 - accuracy: 0.4961
11744/25000 [=============>................] - ETA: 33s - loss: 7.7254 - accuracy: 0.4962
11776/25000 [=============>................] - ETA: 33s - loss: 7.7226 - accuracy: 0.4963
11808/25000 [=============>................] - ETA: 33s - loss: 7.7173 - accuracy: 0.4967
11840/25000 [=============>................] - ETA: 33s - loss: 7.7249 - accuracy: 0.4962
11872/25000 [=============>................] - ETA: 33s - loss: 7.7234 - accuracy: 0.4963
11904/25000 [=============>................] - ETA: 33s - loss: 7.7284 - accuracy: 0.4960
11936/25000 [=============>................] - ETA: 33s - loss: 7.7257 - accuracy: 0.4961
11968/25000 [=============>................] - ETA: 33s - loss: 7.7281 - accuracy: 0.4960
12000/25000 [=============>................] - ETA: 32s - loss: 7.7267 - accuracy: 0.4961
12032/25000 [=============>................] - ETA: 32s - loss: 7.7265 - accuracy: 0.4961
12064/25000 [=============>................] - ETA: 32s - loss: 7.7238 - accuracy: 0.4963
12096/25000 [=============>................] - ETA: 32s - loss: 7.7211 - accuracy: 0.4964
12128/25000 [=============>................] - ETA: 32s - loss: 7.7172 - accuracy: 0.4967
12160/25000 [=============>................] - ETA: 32s - loss: 7.7171 - accuracy: 0.4967
12192/25000 [=============>................] - ETA: 32s - loss: 7.7232 - accuracy: 0.4963
12224/25000 [=============>................] - ETA: 32s - loss: 7.7218 - accuracy: 0.4964
12256/25000 [=============>................] - ETA: 32s - loss: 7.7154 - accuracy: 0.4968
12288/25000 [=============>................] - ETA: 32s - loss: 7.7165 - accuracy: 0.4967
12320/25000 [=============>................] - ETA: 32s - loss: 7.7152 - accuracy: 0.4968
12352/25000 [=============>................] - ETA: 32s - loss: 7.7113 - accuracy: 0.4971
12384/25000 [=============>................] - ETA: 31s - loss: 7.7050 - accuracy: 0.4975
12416/25000 [=============>................] - ETA: 31s - loss: 7.7049 - accuracy: 0.4975
12448/25000 [=============>................] - ETA: 31s - loss: 7.7023 - accuracy: 0.4977
12480/25000 [=============>................] - ETA: 31s - loss: 7.7035 - accuracy: 0.4976
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7009 - accuracy: 0.4978
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6947 - accuracy: 0.4982
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6971 - accuracy: 0.4980
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7007 - accuracy: 0.4978
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6994 - accuracy: 0.4979
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7017 - accuracy: 0.4977
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6944 - accuracy: 0.4982
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6967 - accuracy: 0.4980
12768/25000 [==============>...............] - ETA: 30s - loss: 7.6918 - accuracy: 0.4984
12800/25000 [==============>...............] - ETA: 30s - loss: 7.6966 - accuracy: 0.4980
12832/25000 [==============>...............] - ETA: 30s - loss: 7.6881 - accuracy: 0.4986
12864/25000 [==============>...............] - ETA: 30s - loss: 7.6857 - accuracy: 0.4988
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6904 - accuracy: 0.4984
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6892 - accuracy: 0.4985
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6879 - accuracy: 0.4986
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6867 - accuracy: 0.4987
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6796 - accuracy: 0.4992
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6831 - accuracy: 0.4989
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6818 - accuracy: 0.4990
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6783 - accuracy: 0.4992
13152/25000 [==============>...............] - ETA: 29s - loss: 7.6759 - accuracy: 0.4994
13184/25000 [==============>...............] - ETA: 29s - loss: 7.6736 - accuracy: 0.4995
13216/25000 [==============>...............] - ETA: 29s - loss: 7.6724 - accuracy: 0.4996
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6724 - accuracy: 0.4996
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6724 - accuracy: 0.4996
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6712 - accuracy: 0.4997
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6712 - accuracy: 0.4997
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6769 - accuracy: 0.4993
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6723 - accuracy: 0.4996
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6768 - accuracy: 0.4993
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6825 - accuracy: 0.4990
13568/25000 [===============>..............] - ETA: 28s - loss: 7.6836 - accuracy: 0.4989
13600/25000 [===============>..............] - ETA: 28s - loss: 7.6892 - accuracy: 0.4985
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6880 - accuracy: 0.4986
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6913 - accuracy: 0.4984
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6968 - accuracy: 0.4980
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6934 - accuracy: 0.4983
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6900 - accuracy: 0.4985
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6911 - accuracy: 0.4984
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6921 - accuracy: 0.4983
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6910 - accuracy: 0.4984
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6920 - accuracy: 0.4983
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6953 - accuracy: 0.4981
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6919 - accuracy: 0.4984
13984/25000 [===============>..............] - ETA: 27s - loss: 7.6918 - accuracy: 0.4984
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6929 - accuracy: 0.4983
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6939 - accuracy: 0.4982
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6971 - accuracy: 0.4980
14112/25000 [===============>..............] - ETA: 27s - loss: 7.7025 - accuracy: 0.4977
14144/25000 [===============>..............] - ETA: 27s - loss: 7.7046 - accuracy: 0.4975
14176/25000 [================>.............] - ETA: 27s - loss: 7.7045 - accuracy: 0.4975
14208/25000 [================>.............] - ETA: 27s - loss: 7.7076 - accuracy: 0.4973
14240/25000 [================>.............] - ETA: 27s - loss: 7.7065 - accuracy: 0.4974
14272/25000 [================>.............] - ETA: 27s - loss: 7.7064 - accuracy: 0.4974
14304/25000 [================>.............] - ETA: 26s - loss: 7.7031 - accuracy: 0.4976
14336/25000 [================>.............] - ETA: 26s - loss: 7.7062 - accuracy: 0.4974
14368/25000 [================>.............] - ETA: 26s - loss: 7.7061 - accuracy: 0.4974
14400/25000 [================>.............] - ETA: 26s - loss: 7.7018 - accuracy: 0.4977
14432/25000 [================>.............] - ETA: 26s - loss: 7.7006 - accuracy: 0.4978
14464/25000 [================>.............] - ETA: 26s - loss: 7.7048 - accuracy: 0.4975
14496/25000 [================>.............] - ETA: 26s - loss: 7.7110 - accuracy: 0.4971
14528/25000 [================>.............] - ETA: 26s - loss: 7.7088 - accuracy: 0.4972
14560/25000 [================>.............] - ETA: 26s - loss: 7.7098 - accuracy: 0.4972
14592/25000 [================>.............] - ETA: 26s - loss: 7.7108 - accuracy: 0.4971
14624/25000 [================>.............] - ETA: 26s - loss: 7.7075 - accuracy: 0.4973
14656/25000 [================>.............] - ETA: 26s - loss: 7.7074 - accuracy: 0.4973
14688/25000 [================>.............] - ETA: 25s - loss: 7.7073 - accuracy: 0.4973
14720/25000 [================>.............] - ETA: 25s - loss: 7.7072 - accuracy: 0.4974
14752/25000 [================>.............] - ETA: 25s - loss: 7.7082 - accuracy: 0.4973
14784/25000 [================>.............] - ETA: 25s - loss: 7.7029 - accuracy: 0.4976
14816/25000 [================>.............] - ETA: 25s - loss: 7.7070 - accuracy: 0.4974
14848/25000 [================>.............] - ETA: 25s - loss: 7.7110 - accuracy: 0.4971
14880/25000 [================>.............] - ETA: 25s - loss: 7.7037 - accuracy: 0.4976
14912/25000 [================>.............] - ETA: 25s - loss: 7.7047 - accuracy: 0.4975
14944/25000 [================>.............] - ETA: 25s - loss: 7.7046 - accuracy: 0.4975
14976/25000 [================>.............] - ETA: 25s - loss: 7.7014 - accuracy: 0.4977
15008/25000 [=================>............] - ETA: 25s - loss: 7.7075 - accuracy: 0.4973
15040/25000 [=================>............] - ETA: 25s - loss: 7.7105 - accuracy: 0.4971
15072/25000 [=================>............] - ETA: 24s - loss: 7.7114 - accuracy: 0.4971
15104/25000 [=================>............] - ETA: 24s - loss: 7.7113 - accuracy: 0.4971
15136/25000 [=================>............] - ETA: 24s - loss: 7.7102 - accuracy: 0.4972
15168/25000 [=================>............] - ETA: 24s - loss: 7.7151 - accuracy: 0.4968
15200/25000 [=================>............] - ETA: 24s - loss: 7.7140 - accuracy: 0.4969
15232/25000 [=================>............] - ETA: 24s - loss: 7.7149 - accuracy: 0.4968
15264/25000 [=================>............] - ETA: 24s - loss: 7.7148 - accuracy: 0.4969
15296/25000 [=================>............] - ETA: 24s - loss: 7.7127 - accuracy: 0.4970
15328/25000 [=================>............] - ETA: 24s - loss: 7.7156 - accuracy: 0.4968
15360/25000 [=================>............] - ETA: 24s - loss: 7.7105 - accuracy: 0.4971
15392/25000 [=================>............] - ETA: 24s - loss: 7.7075 - accuracy: 0.4973
15424/25000 [=================>............] - ETA: 24s - loss: 7.7064 - accuracy: 0.4974
15456/25000 [=================>............] - ETA: 24s - loss: 7.7013 - accuracy: 0.4977
15488/25000 [=================>............] - ETA: 23s - loss: 7.6983 - accuracy: 0.4979
15520/25000 [=================>............] - ETA: 23s - loss: 7.6992 - accuracy: 0.4979
15552/25000 [=================>............] - ETA: 23s - loss: 7.6982 - accuracy: 0.4979
15584/25000 [=================>............] - ETA: 23s - loss: 7.6991 - accuracy: 0.4979
15616/25000 [=================>............] - ETA: 23s - loss: 7.6951 - accuracy: 0.4981
15648/25000 [=================>............] - ETA: 23s - loss: 7.6960 - accuracy: 0.4981
15680/25000 [=================>............] - ETA: 23s - loss: 7.6960 - accuracy: 0.4981
15712/25000 [=================>............] - ETA: 23s - loss: 7.6930 - accuracy: 0.4983
15744/25000 [=================>............] - ETA: 23s - loss: 7.6958 - accuracy: 0.4981
15776/25000 [=================>............] - ETA: 23s - loss: 7.6948 - accuracy: 0.4982
15808/25000 [=================>............] - ETA: 23s - loss: 7.6957 - accuracy: 0.4981
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6976 - accuracy: 0.4980
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6966 - accuracy: 0.4980
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6955 - accuracy: 0.4981
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6945 - accuracy: 0.4982
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6964 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6973 - accuracy: 0.4980
16032/25000 [==================>...........] - ETA: 22s - loss: 7.7001 - accuracy: 0.4978
16064/25000 [==================>...........] - ETA: 22s - loss: 7.7029 - accuracy: 0.4976
16096/25000 [==================>...........] - ETA: 22s - loss: 7.7047 - accuracy: 0.4975
16128/25000 [==================>...........] - ETA: 22s - loss: 7.7046 - accuracy: 0.4975
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6989 - accuracy: 0.4979
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6950 - accuracy: 0.4981
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6940 - accuracy: 0.4982
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6911 - accuracy: 0.4984
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6920 - accuracy: 0.4983
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6939 - accuracy: 0.4982
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6901 - accuracy: 0.4985
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6938 - accuracy: 0.4982
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6937 - accuracy: 0.4982
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6918 - accuracy: 0.4984
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6945 - accuracy: 0.4982
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6973 - accuracy: 0.4980
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6972 - accuracy: 0.4980
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6962 - accuracy: 0.4981
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6962 - accuracy: 0.4981
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6933 - accuracy: 0.4983
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6951 - accuracy: 0.4981
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6960 - accuracy: 0.4981
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6941 - accuracy: 0.4982
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6877 - accuracy: 0.4986
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6867 - accuracy: 0.4987
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6839 - accuracy: 0.4989
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6866 - accuracy: 0.4987
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6857 - accuracy: 0.4988
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6847 - accuracy: 0.4988
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6874 - accuracy: 0.4986
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6838 - accuracy: 0.4989
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6855 - accuracy: 0.4988
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6864 - accuracy: 0.4987
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6855 - accuracy: 0.4988
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6872 - accuracy: 0.4987
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6854 - accuracy: 0.4988
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6862 - accuracy: 0.4987
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6862 - accuracy: 0.4987
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6862 - accuracy: 0.4987
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6861 - accuracy: 0.4987
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6861 - accuracy: 0.4987
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6834 - accuracy: 0.4989
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6825 - accuracy: 0.4990
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6834 - accuracy: 0.4989
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6851 - accuracy: 0.4988
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6789 - accuracy: 0.4992
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6806 - accuracy: 0.4991
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6832 - accuracy: 0.4989
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6806 - accuracy: 0.4991
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6771 - accuracy: 0.4993
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6779 - accuracy: 0.4993
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6770 - accuracy: 0.4993
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6718 - accuracy: 0.4997
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6735 - accuracy: 0.4995
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6718 - accuracy: 0.4997
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6752 - accuracy: 0.4994
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6769 - accuracy: 0.4993
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6786 - accuracy: 0.4992
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6803 - accuracy: 0.4991
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6803 - accuracy: 0.4991
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6803 - accuracy: 0.4991
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6820 - accuracy: 0.4990
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6819 - accuracy: 0.4990
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6828 - accuracy: 0.4989
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6827 - accuracy: 0.4989
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6836 - accuracy: 0.4989
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6810 - accuracy: 0.4991
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6869 - accuracy: 0.4987
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6868 - accuracy: 0.4987
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6851 - accuracy: 0.4988
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6876 - accuracy: 0.4986
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6892 - accuracy: 0.4985
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6884 - accuracy: 0.4986
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6875 - accuracy: 0.4986
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6883 - accuracy: 0.4986
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6916 - accuracy: 0.4984
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6907 - accuracy: 0.4984
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6956 - accuracy: 0.4981
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6981 - accuracy: 0.4979
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6964 - accuracy: 0.4981
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6963 - accuracy: 0.4981
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6971 - accuracy: 0.4980
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6987 - accuracy: 0.4979
18688/25000 [=====================>........] - ETA: 15s - loss: 7.7003 - accuracy: 0.4978
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6961 - accuracy: 0.4981
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6928 - accuracy: 0.4983
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6919 - accuracy: 0.4983
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6919 - accuracy: 0.4984
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6886 - accuracy: 0.4986
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6869 - accuracy: 0.4987
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6869 - accuracy: 0.4987
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6860 - accuracy: 0.4987
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6868 - accuracy: 0.4987
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6868 - accuracy: 0.4987
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6892 - accuracy: 0.4985
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6859 - accuracy: 0.4987
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6851 - accuracy: 0.4988
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6826 - accuracy: 0.4990
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6826 - accuracy: 0.4990
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6770 - accuracy: 0.4993
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6778 - accuracy: 0.4993
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6809 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6801 - accuracy: 0.4991
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6793 - accuracy: 0.4992
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6793 - accuracy: 0.4992
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6753 - accuracy: 0.4994
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6753 - accuracy: 0.4994
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6713 - accuracy: 0.4997
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6713 - accuracy: 0.4997
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6705 - accuracy: 0.4997
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6682 - accuracy: 0.4999
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6658 - accuracy: 0.5001
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6643 - accuracy: 0.5002
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6604 - accuracy: 0.5004
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6589 - accuracy: 0.5005
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6612 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6635 - accuracy: 0.5002
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6643 - accuracy: 0.5002
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6620 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6620 - accuracy: 0.5003
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6605 - accuracy: 0.5004
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6590 - accuracy: 0.5005
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6575 - accuracy: 0.5006
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6567 - accuracy: 0.5006
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6552 - accuracy: 0.5007
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6568 - accuracy: 0.5006
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6530 - accuracy: 0.5009
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6538 - accuracy: 0.5008
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6508 - accuracy: 0.5010
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6508 - accuracy: 0.5010
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6508 - accuracy: 0.5010
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6501 - accuracy: 0.5011
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6494 - accuracy: 0.5011
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6524 - accuracy: 0.5009
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6539 - accuracy: 0.5008
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6532 - accuracy: 0.5009
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6540 - accuracy: 0.5008
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6510 - accuracy: 0.5010
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6510 - accuracy: 0.5010
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6481 - accuracy: 0.5012
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6496 - accuracy: 0.5011
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6474 - accuracy: 0.5013
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6489 - accuracy: 0.5012
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6475 - accuracy: 0.5013
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6460 - accuracy: 0.5013
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6468 - accuracy: 0.5013
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6431 - accuracy: 0.5015
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6461 - accuracy: 0.5013
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6447 - accuracy: 0.5014
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6447 - accuracy: 0.5014 
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6426 - accuracy: 0.5016
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6448 - accuracy: 0.5014
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6441 - accuracy: 0.5015
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6456 - accuracy: 0.5014
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6456 - accuracy: 0.5014
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6456 - accuracy: 0.5014
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6449 - accuracy: 0.5014
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6471 - accuracy: 0.5013
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6443 - accuracy: 0.5015
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6479 - accuracy: 0.5012
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6501 - accuracy: 0.5011
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6480 - accuracy: 0.5012
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6466 - accuracy: 0.5013
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6437 - accuracy: 0.5015
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6416 - accuracy: 0.5016
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6445 - accuracy: 0.5014
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6424 - accuracy: 0.5016
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6439 - accuracy: 0.5015
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6453 - accuracy: 0.5014
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6468 - accuracy: 0.5013
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6461 - accuracy: 0.5013
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6447 - accuracy: 0.5014
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6447 - accuracy: 0.5014
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6434 - accuracy: 0.5015
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6413 - accuracy: 0.5017
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6434 - accuracy: 0.5015
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6386 - accuracy: 0.5018
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6386 - accuracy: 0.5018
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6365 - accuracy: 0.5020
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6366 - accuracy: 0.5020
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6380 - accuracy: 0.5019
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6388 - accuracy: 0.5018
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6381 - accuracy: 0.5019
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6395 - accuracy: 0.5018
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6430 - accuracy: 0.5015
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6410 - accuracy: 0.5017
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6397 - accuracy: 0.5018
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6390 - accuracy: 0.5018
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6370 - accuracy: 0.5019
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6356 - accuracy: 0.5020
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6377 - accuracy: 0.5019
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6385 - accuracy: 0.5018
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6406 - accuracy: 0.5017
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6440 - accuracy: 0.5015
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6461 - accuracy: 0.5013
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6475 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6503 - accuracy: 0.5011
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6530 - accuracy: 0.5009
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6530 - accuracy: 0.5009
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6490 - accuracy: 0.5011
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6470 - accuracy: 0.5013
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6484 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6477 - accuracy: 0.5012
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6539 - accuracy: 0.5008
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6539 - accuracy: 0.5008
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6552 - accuracy: 0.5007
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6566 - accuracy: 0.5007
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6539 - accuracy: 0.5008
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6553 - accuracy: 0.5007
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6566 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6586 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6593 - accuracy: 0.5005
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6593 - accuracy: 0.5005
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6580 - accuracy: 0.5006
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6580 - accuracy: 0.5006
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6580 - accuracy: 0.5006
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6600 - accuracy: 0.5004
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6607 - accuracy: 0.5004
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6587 - accuracy: 0.5005
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6555 - accuracy: 0.5007
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6548 - accuracy: 0.5008
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6548 - accuracy: 0.5008
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6535 - accuracy: 0.5009
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6549 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6686 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24288/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24320/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24352/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24480/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24512/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24672/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24704/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
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
