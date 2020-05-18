
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4d4199a543322deeed64b48d9193e96f47abe311', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '4d4199a543322deeed64b48d9193e96f47abe311', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4d4199a543322deeed64b48d9193e96f47abe311

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4d4199a543322deeed64b48d9193e96f47abe311

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/4d4199a543322deeed64b48d9193e96f47abe311

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
 40%|████      | 2/5 [00:58<01:27, 29.20s/it] 40%|████      | 2/5 [00:58<01:27, 29.20s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4545146089631684, 'embedding_size_factor': 0.6830460895769705, 'layers.choice': 3, 'learning_rate': 0.0014828778372500408, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.355433713731568e-12} and reward: 0.3728
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x16\xc4qCD0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\xdb\x83y\x0c\x99\xceX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?XK\xa3\xf4(-\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x84\xb7\xf7\x172\xc6\xedu.' and reward: 0.3728
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x16\xc4qCD0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\xdb\x83y\x0c\x99\xceX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?XK\xa3\xf4(-\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x84\xb7\xf7\x172\xc6\xedu.' and reward: 0.3728
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 122.63301301002502
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -4.55s of remaining time.
Ensemble size: 27
Ensemble weights: 
[0.62962963 0.37037037]
	0.389	 = Validation accuracy score
	1.03s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 125.61s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f5b041e3ac8> 

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
 [-1.17918484e-01  9.43967327e-03  6.01332821e-03  8.33964497e-02
   1.05064832e-01 -7.98152909e-02]
 [ 1.30577743e-01  2.00551763e-01  3.78879383e-02  8.53311047e-02
  -3.92706133e-02 -9.14068893e-02]
 [-1.32247299e-01  4.53528941e-01  9.89302844e-02 -5.92675135e-02
   2.85300195e-01 -9.18888301e-02]
 [-1.22976758e-01  6.02386929e-02  9.85922068e-02 -1.04806677e-01
   1.40319526e-01 -5.51129021e-02]
 [ 2.08844557e-01  1.43241346e-01 -2.24485546e-01 -8.29551462e-03
   1.72840208e-01  2.66161591e-01]
 [ 1.06993064e-01  2.11782381e-01 -1.66577309e-01  8.24487060e-02
  -1.09188724e-04 -2.21077830e-01]
 [ 4.41156834e-01  4.85093817e-02 -3.36807132e-01 -7.34770596e-02
   6.12796128e-01  6.41066059e-02]
 [-2.96868652e-01  6.45997822e-01  7.41948307e-01  5.61928868e-01
   3.08667481e-01  1.73657164e-01]
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
{'loss': 0.6777572929859161, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 08:28:04.264670: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.3873522952198982, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 08:28:05.577538: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 4s
  958464/17464789 [>.............................] - ETA: 0s
 4063232/17464789 [=====>........................] - ETA: 0s
 8708096/17464789 [=============>................] - ETA: 0s
12312576/17464789 [====================>.........] - ETA: 0s
16031744/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-18 08:28:18.919641: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 08:28:18.924415: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-18 08:28:18.924620: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f7aed8b140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 08:28:18.924638: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:06 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 3:19 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:42 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:23 - loss: 6.5885 - accuracy: 0.5703
  160/25000 [..............................] - ETA: 2:12 - loss: 6.9958 - accuracy: 0.5437
  192/25000 [..............................] - ETA: 2:05 - loss: 6.9479 - accuracy: 0.5469
  224/25000 [..............................] - ETA: 1:59 - loss: 6.6398 - accuracy: 0.5670
  256/25000 [..............................] - ETA: 1:55 - loss: 6.7682 - accuracy: 0.5586
  288/25000 [..............................] - ETA: 1:52 - loss: 7.0277 - accuracy: 0.5417
  320/25000 [..............................] - ETA: 1:50 - loss: 6.9958 - accuracy: 0.5437
  352/25000 [..............................] - ETA: 1:48 - loss: 7.0568 - accuracy: 0.5398
  384/25000 [..............................] - ETA: 1:46 - loss: 6.9878 - accuracy: 0.5443
  416/25000 [..............................] - ETA: 1:45 - loss: 7.0032 - accuracy: 0.5433
  448/25000 [..............................] - ETA: 1:44 - loss: 7.0506 - accuracy: 0.5402
  480/25000 [..............................] - ETA: 1:43 - loss: 6.9958 - accuracy: 0.5437
  512/25000 [..............................] - ETA: 1:42 - loss: 6.9479 - accuracy: 0.5469
  544/25000 [..............................] - ETA: 1:40 - loss: 6.9620 - accuracy: 0.5460
  576/25000 [..............................] - ETA: 1:40 - loss: 7.0277 - accuracy: 0.5417
  608/25000 [..............................] - ETA: 1:39 - loss: 7.1370 - accuracy: 0.5345
  640/25000 [..............................] - ETA: 1:38 - loss: 7.2593 - accuracy: 0.5266
  672/25000 [..............................] - ETA: 1:38 - loss: 7.2103 - accuracy: 0.5298
  704/25000 [..............................] - ETA: 1:38 - loss: 7.1875 - accuracy: 0.5312
  736/25000 [..............................] - ETA: 1:37 - loss: 7.1875 - accuracy: 0.5312
  768/25000 [..............................] - ETA: 1:37 - loss: 7.2673 - accuracy: 0.5260
  800/25000 [..............................] - ETA: 1:36 - loss: 7.3216 - accuracy: 0.5225
  832/25000 [..............................] - ETA: 1:36 - loss: 7.3349 - accuracy: 0.5216
  864/25000 [>.............................] - ETA: 1:35 - loss: 7.3117 - accuracy: 0.5231
  896/25000 [>.............................] - ETA: 1:35 - loss: 7.3415 - accuracy: 0.5212
  928/25000 [>.............................] - ETA: 1:34 - loss: 7.3692 - accuracy: 0.5194
  960/25000 [>.............................] - ETA: 1:34 - loss: 7.3152 - accuracy: 0.5229
  992/25000 [>.............................] - ETA: 1:33 - loss: 7.3729 - accuracy: 0.5192
 1024/25000 [>.............................] - ETA: 1:33 - loss: 7.3671 - accuracy: 0.5195
 1056/25000 [>.............................] - ETA: 1:32 - loss: 7.3617 - accuracy: 0.5199
 1088/25000 [>.............................] - ETA: 1:32 - loss: 7.3425 - accuracy: 0.5211
 1120/25000 [>.............................] - ETA: 1:32 - loss: 7.3381 - accuracy: 0.5214
 1152/25000 [>.............................] - ETA: 1:32 - loss: 7.2939 - accuracy: 0.5243
 1184/25000 [>.............................] - ETA: 1:32 - loss: 7.2522 - accuracy: 0.5270
 1216/25000 [>.............................] - ETA: 1:31 - loss: 7.2757 - accuracy: 0.5255
 1248/25000 [>.............................] - ETA: 1:31 - loss: 7.2612 - accuracy: 0.5264
 1280/25000 [>.............................] - ETA: 1:31 - loss: 7.2833 - accuracy: 0.5250
 1312/25000 [>.............................] - ETA: 1:31 - loss: 7.2810 - accuracy: 0.5252
 1344/25000 [>.............................] - ETA: 1:30 - loss: 7.2673 - accuracy: 0.5260
 1376/25000 [>.............................] - ETA: 1:30 - loss: 7.2543 - accuracy: 0.5269
 1408/25000 [>.............................] - ETA: 1:30 - loss: 7.2419 - accuracy: 0.5277
 1440/25000 [>.............................] - ETA: 1:30 - loss: 7.2194 - accuracy: 0.5292
 1472/25000 [>.............................] - ETA: 1:30 - loss: 7.3020 - accuracy: 0.5238
 1504/25000 [>.............................] - ETA: 1:29 - loss: 7.3608 - accuracy: 0.5199
 1536/25000 [>.............................] - ETA: 1:29 - loss: 7.3971 - accuracy: 0.5176
 1568/25000 [>.............................] - ETA: 1:29 - loss: 7.4026 - accuracy: 0.5172
 1600/25000 [>.............................] - ETA: 1:29 - loss: 7.4175 - accuracy: 0.5163
 1632/25000 [>.............................] - ETA: 1:29 - loss: 7.4035 - accuracy: 0.5172
 1664/25000 [>.............................] - ETA: 1:29 - loss: 7.3902 - accuracy: 0.5180
 1696/25000 [=>............................] - ETA: 1:29 - loss: 7.3773 - accuracy: 0.5189
 1728/25000 [=>............................] - ETA: 1:28 - loss: 7.4093 - accuracy: 0.5168
 1760/25000 [=>............................] - ETA: 1:28 - loss: 7.4314 - accuracy: 0.5153
 1792/25000 [=>............................] - ETA: 1:28 - loss: 7.4442 - accuracy: 0.5145
 1824/25000 [=>............................] - ETA: 1:27 - loss: 7.3976 - accuracy: 0.5175
 1856/25000 [=>............................] - ETA: 1:27 - loss: 7.4105 - accuracy: 0.5167
 1888/25000 [=>............................] - ETA: 1:27 - loss: 7.3824 - accuracy: 0.5185
 1920/25000 [=>............................] - ETA: 1:27 - loss: 7.3791 - accuracy: 0.5188
 1952/25000 [=>............................] - ETA: 1:27 - loss: 7.3995 - accuracy: 0.5174
 1984/25000 [=>............................] - ETA: 1:26 - loss: 7.4425 - accuracy: 0.5146
 2016/25000 [=>............................] - ETA: 1:26 - loss: 7.4689 - accuracy: 0.5129
 2048/25000 [=>............................] - ETA: 1:26 - loss: 7.4645 - accuracy: 0.5132
 2080/25000 [=>............................] - ETA: 1:26 - loss: 7.4676 - accuracy: 0.5130
 2112/25000 [=>............................] - ETA: 1:25 - loss: 7.4996 - accuracy: 0.5109
 2144/25000 [=>............................] - ETA: 1:25 - loss: 7.4664 - accuracy: 0.5131
 2176/25000 [=>............................] - ETA: 1:25 - loss: 7.4975 - accuracy: 0.5110
 2208/25000 [=>............................] - ETA: 1:25 - loss: 7.4930 - accuracy: 0.5113
 2240/25000 [=>............................] - ETA: 1:25 - loss: 7.5229 - accuracy: 0.5094
 2272/25000 [=>............................] - ETA: 1:25 - loss: 7.5249 - accuracy: 0.5092
 2304/25000 [=>............................] - ETA: 1:25 - loss: 7.5269 - accuracy: 0.5091
 2336/25000 [=>............................] - ETA: 1:24 - loss: 7.5550 - accuracy: 0.5073
 2368/25000 [=>............................] - ETA: 1:24 - loss: 7.5565 - accuracy: 0.5072
 2400/25000 [=>............................] - ETA: 1:24 - loss: 7.5452 - accuracy: 0.5079
 2432/25000 [=>............................] - ETA: 1:24 - loss: 7.5342 - accuracy: 0.5086
 2464/25000 [=>............................] - ETA: 1:24 - loss: 7.5297 - accuracy: 0.5089
 2496/25000 [=>............................] - ETA: 1:24 - loss: 7.5130 - accuracy: 0.5100
 2528/25000 [==>...........................] - ETA: 1:24 - loss: 7.5089 - accuracy: 0.5103
 2560/25000 [==>...........................] - ETA: 1:24 - loss: 7.5049 - accuracy: 0.5105
 2592/25000 [==>...........................] - ETA: 1:23 - loss: 7.4714 - accuracy: 0.5127
 2624/25000 [==>...........................] - ETA: 1:23 - loss: 7.4913 - accuracy: 0.5114
 2656/25000 [==>...........................] - ETA: 1:23 - loss: 7.5050 - accuracy: 0.5105
 2688/25000 [==>...........................] - ETA: 1:23 - loss: 7.4955 - accuracy: 0.5112
 2720/25000 [==>...........................] - ETA: 1:23 - loss: 7.4919 - accuracy: 0.5114
 2752/25000 [==>...........................] - ETA: 1:23 - loss: 7.5273 - accuracy: 0.5091
 2784/25000 [==>...........................] - ETA: 1:23 - loss: 7.5455 - accuracy: 0.5079
 2816/25000 [==>...........................] - ETA: 1:22 - loss: 7.5359 - accuracy: 0.5085
 2848/25000 [==>...........................] - ETA: 1:22 - loss: 7.5105 - accuracy: 0.5102
 2880/25000 [==>...........................] - ETA: 1:22 - loss: 7.5229 - accuracy: 0.5094
 2912/25000 [==>...........................] - ETA: 1:22 - loss: 7.5297 - accuracy: 0.5089
 2944/25000 [==>...........................] - ETA: 1:22 - loss: 7.5416 - accuracy: 0.5082
 2976/25000 [==>...........................] - ETA: 1:22 - loss: 7.5378 - accuracy: 0.5084
 3008/25000 [==>...........................] - ETA: 1:22 - loss: 7.5188 - accuracy: 0.5096
 3040/25000 [==>...........................] - ETA: 1:21 - loss: 7.5203 - accuracy: 0.5095
 3072/25000 [==>...........................] - ETA: 1:21 - loss: 7.4969 - accuracy: 0.5111
 3104/25000 [==>...........................] - ETA: 1:21 - loss: 7.4888 - accuracy: 0.5116
 3136/25000 [==>...........................] - ETA: 1:21 - loss: 7.4955 - accuracy: 0.5112
 3168/25000 [==>...........................] - ETA: 1:21 - loss: 7.5021 - accuracy: 0.5107
 3200/25000 [==>...........................] - ETA: 1:21 - loss: 7.5085 - accuracy: 0.5103
 3232/25000 [==>...........................] - ETA: 1:21 - loss: 7.5053 - accuracy: 0.5105
 3264/25000 [==>...........................] - ETA: 1:21 - loss: 7.5069 - accuracy: 0.5104
 3296/25000 [==>...........................] - ETA: 1:20 - loss: 7.5038 - accuracy: 0.5106
 3328/25000 [==>...........................] - ETA: 1:20 - loss: 7.4915 - accuracy: 0.5114
 3360/25000 [===>..........................] - ETA: 1:20 - loss: 7.4841 - accuracy: 0.5119
 3392/25000 [===>..........................] - ETA: 1:20 - loss: 7.4994 - accuracy: 0.5109
 3424/25000 [===>..........................] - ETA: 1:20 - loss: 7.5099 - accuracy: 0.5102
 3456/25000 [===>..........................] - ETA: 1:20 - loss: 7.5069 - accuracy: 0.5104
 3488/25000 [===>..........................] - ETA: 1:19 - loss: 7.5303 - accuracy: 0.5089
 3520/25000 [===>..........................] - ETA: 1:19 - loss: 7.5272 - accuracy: 0.5091
 3552/25000 [===>..........................] - ETA: 1:19 - loss: 7.5414 - accuracy: 0.5082
 3584/25000 [===>..........................] - ETA: 1:19 - loss: 7.5212 - accuracy: 0.5095
 3616/25000 [===>..........................] - ETA: 1:19 - loss: 7.5394 - accuracy: 0.5083
 3648/25000 [===>..........................] - ETA: 1:19 - loss: 7.5321 - accuracy: 0.5088
 3680/25000 [===>..........................] - ETA: 1:19 - loss: 7.5291 - accuracy: 0.5090
 3712/25000 [===>..........................] - ETA: 1:19 - loss: 7.5344 - accuracy: 0.5086
 3744/25000 [===>..........................] - ETA: 1:19 - loss: 7.5315 - accuracy: 0.5088
 3776/25000 [===>..........................] - ETA: 1:19 - loss: 7.5326 - accuracy: 0.5087
 3808/25000 [===>..........................] - ETA: 1:19 - loss: 7.5257 - accuracy: 0.5092
 3840/25000 [===>..........................] - ETA: 1:19 - loss: 7.5309 - accuracy: 0.5089
 3872/25000 [===>..........................] - ETA: 1:18 - loss: 7.5359 - accuracy: 0.5085
 3904/25000 [===>..........................] - ETA: 1:18 - loss: 7.5409 - accuracy: 0.5082
 3936/25000 [===>..........................] - ETA: 1:18 - loss: 7.5303 - accuracy: 0.5089
 3968/25000 [===>..........................] - ETA: 1:18 - loss: 7.5507 - accuracy: 0.5076
 4000/25000 [===>..........................] - ETA: 1:18 - loss: 7.5670 - accuracy: 0.5065
 4032/25000 [===>..........................] - ETA: 1:18 - loss: 7.5639 - accuracy: 0.5067
 4064/25000 [===>..........................] - ETA: 1:18 - loss: 7.5534 - accuracy: 0.5074
 4096/25000 [===>..........................] - ETA: 1:18 - loss: 7.5431 - accuracy: 0.5081
 4128/25000 [===>..........................] - ETA: 1:18 - loss: 7.5478 - accuracy: 0.5078
 4160/25000 [===>..........................] - ETA: 1:18 - loss: 7.5524 - accuracy: 0.5075
 4192/25000 [====>.........................] - ETA: 1:17 - loss: 7.5532 - accuracy: 0.5074
 4224/25000 [====>.........................] - ETA: 1:17 - loss: 7.5505 - accuracy: 0.5076
 4256/25000 [====>.........................] - ETA: 1:17 - loss: 7.5441 - accuracy: 0.5080
 4288/25000 [====>.........................] - ETA: 1:17 - loss: 7.5522 - accuracy: 0.5075
 4320/25000 [====>.........................] - ETA: 1:17 - loss: 7.5495 - accuracy: 0.5076
 4352/25000 [====>.........................] - ETA: 1:17 - loss: 7.5363 - accuracy: 0.5085
 4384/25000 [====>.........................] - ETA: 1:17 - loss: 7.5232 - accuracy: 0.5094
 4416/25000 [====>.........................] - ETA: 1:17 - loss: 7.5208 - accuracy: 0.5095
 4448/25000 [====>.........................] - ETA: 1:16 - loss: 7.5184 - accuracy: 0.5097
 4480/25000 [====>.........................] - ETA: 1:16 - loss: 7.5058 - accuracy: 0.5105
 4512/25000 [====>.........................] - ETA: 1:16 - loss: 7.5035 - accuracy: 0.5106
 4544/25000 [====>.........................] - ETA: 1:16 - loss: 7.4979 - accuracy: 0.5110
 4576/25000 [====>.........................] - ETA: 1:16 - loss: 7.5024 - accuracy: 0.5107
 4608/25000 [====>.........................] - ETA: 1:16 - loss: 7.5069 - accuracy: 0.5104
 4640/25000 [====>.........................] - ETA: 1:16 - loss: 7.5014 - accuracy: 0.5108
 4672/25000 [====>.........................] - ETA: 1:16 - loss: 7.4960 - accuracy: 0.5111
 4704/25000 [====>.........................] - ETA: 1:15 - loss: 7.4906 - accuracy: 0.5115
 4736/25000 [====>.........................] - ETA: 1:15 - loss: 7.4983 - accuracy: 0.5110
 4768/25000 [====>.........................] - ETA: 1:15 - loss: 7.4897 - accuracy: 0.5115
 4800/25000 [====>.........................] - ETA: 1:15 - loss: 7.5101 - accuracy: 0.5102
 4832/25000 [====>.........................] - ETA: 1:15 - loss: 7.5143 - accuracy: 0.5099
 4864/25000 [====>.........................] - ETA: 1:15 - loss: 7.5058 - accuracy: 0.5105
 4896/25000 [====>.........................] - ETA: 1:15 - loss: 7.5194 - accuracy: 0.5096
 4928/25000 [====>.........................] - ETA: 1:15 - loss: 7.5204 - accuracy: 0.5095
 4960/25000 [====>.........................] - ETA: 1:15 - loss: 7.5337 - accuracy: 0.5087
 4992/25000 [====>.........................] - ETA: 1:15 - loss: 7.5376 - accuracy: 0.5084
 5024/25000 [=====>........................] - ETA: 1:14 - loss: 7.5354 - accuracy: 0.5086
 5056/25000 [=====>........................] - ETA: 1:14 - loss: 7.5453 - accuracy: 0.5079
 5088/25000 [=====>........................] - ETA: 1:14 - loss: 7.5400 - accuracy: 0.5083
 5120/25000 [=====>........................] - ETA: 1:14 - loss: 7.5408 - accuracy: 0.5082
 5152/25000 [=====>........................] - ETA: 1:14 - loss: 7.5505 - accuracy: 0.5076
 5184/25000 [=====>........................] - ETA: 1:14 - loss: 7.5483 - accuracy: 0.5077
 5216/25000 [=====>........................] - ETA: 1:14 - loss: 7.5461 - accuracy: 0.5079
 5248/25000 [=====>........................] - ETA: 1:14 - loss: 7.5468 - accuracy: 0.5078
 5280/25000 [=====>........................] - ETA: 1:13 - loss: 7.5505 - accuracy: 0.5076
 5312/25000 [=====>........................] - ETA: 1:13 - loss: 7.5569 - accuracy: 0.5072
 5344/25000 [=====>........................] - ETA: 1:13 - loss: 7.5633 - accuracy: 0.5067
 5376/25000 [=====>........................] - ETA: 1:13 - loss: 7.5582 - accuracy: 0.5071
 5408/25000 [=====>........................] - ETA: 1:13 - loss: 7.5532 - accuracy: 0.5074
 5440/25000 [=====>........................] - ETA: 1:13 - loss: 7.5595 - accuracy: 0.5070
 5472/25000 [=====>........................] - ETA: 1:13 - loss: 7.5713 - accuracy: 0.5062
 5504/25000 [=====>........................] - ETA: 1:13 - loss: 7.5635 - accuracy: 0.5067
 5536/25000 [=====>........................] - ETA: 1:13 - loss: 7.5614 - accuracy: 0.5069
 5568/25000 [=====>........................] - ETA: 1:12 - loss: 7.5620 - accuracy: 0.5068
 5600/25000 [=====>........................] - ETA: 1:12 - loss: 7.5735 - accuracy: 0.5061
 5632/25000 [=====>........................] - ETA: 1:12 - loss: 7.5768 - accuracy: 0.5059
 5664/25000 [=====>........................] - ETA: 1:12 - loss: 7.5827 - accuracy: 0.5055
 5696/25000 [=====>........................] - ETA: 1:12 - loss: 7.5724 - accuracy: 0.5061
 5728/25000 [=====>........................] - ETA: 1:12 - loss: 7.5836 - accuracy: 0.5054
 5760/25000 [=====>........................] - ETA: 1:12 - loss: 7.5761 - accuracy: 0.5059
 5792/25000 [=====>........................] - ETA: 1:12 - loss: 7.5793 - accuracy: 0.5057
 5824/25000 [=====>........................] - ETA: 1:11 - loss: 7.5797 - accuracy: 0.5057
 5856/25000 [======>.......................] - ETA: 1:11 - loss: 7.5750 - accuracy: 0.5060
 5888/25000 [======>.......................] - ETA: 1:11 - loss: 7.5703 - accuracy: 0.5063
 5920/25000 [======>.......................] - ETA: 1:11 - loss: 7.5630 - accuracy: 0.5068
 5952/25000 [======>.......................] - ETA: 1:11 - loss: 7.5687 - accuracy: 0.5064
 5984/25000 [======>.......................] - ETA: 1:11 - loss: 7.5769 - accuracy: 0.5058
 6016/25000 [======>.......................] - ETA: 1:11 - loss: 7.5800 - accuracy: 0.5057
 6048/25000 [======>.......................] - ETA: 1:11 - loss: 7.5728 - accuracy: 0.5061
 6080/25000 [======>.......................] - ETA: 1:10 - loss: 7.5708 - accuracy: 0.5063
 6112/25000 [======>.......................] - ETA: 1:10 - loss: 7.5688 - accuracy: 0.5064
 6144/25000 [======>.......................] - ETA: 1:10 - loss: 7.5618 - accuracy: 0.5068
 6176/25000 [======>.......................] - ETA: 1:10 - loss: 7.5748 - accuracy: 0.5060
 6208/25000 [======>.......................] - ETA: 1:10 - loss: 7.5703 - accuracy: 0.5063
 6240/25000 [======>.......................] - ETA: 1:10 - loss: 7.5683 - accuracy: 0.5064
 6272/25000 [======>.......................] - ETA: 1:10 - loss: 7.5615 - accuracy: 0.5069
 6304/25000 [======>.......................] - ETA: 1:10 - loss: 7.5474 - accuracy: 0.5078
 6336/25000 [======>.......................] - ETA: 1:10 - loss: 7.5456 - accuracy: 0.5079
 6368/25000 [======>.......................] - ETA: 1:09 - loss: 7.5462 - accuracy: 0.5079
 6400/25000 [======>.......................] - ETA: 1:09 - loss: 7.5468 - accuracy: 0.5078
 6432/25000 [======>.......................] - ETA: 1:09 - loss: 7.5403 - accuracy: 0.5082
 6464/25000 [======>.......................] - ETA: 1:09 - loss: 7.5338 - accuracy: 0.5087
 6496/25000 [======>.......................] - ETA: 1:09 - loss: 7.5321 - accuracy: 0.5088
 6528/25000 [======>.......................] - ETA: 1:09 - loss: 7.5374 - accuracy: 0.5084
 6560/25000 [======>.......................] - ETA: 1:09 - loss: 7.5381 - accuracy: 0.5084
 6592/25000 [======>.......................] - ETA: 1:09 - loss: 7.5410 - accuracy: 0.5082
 6624/25000 [======>.......................] - ETA: 1:08 - loss: 7.5300 - accuracy: 0.5089
 6656/25000 [======>.......................] - ETA: 1:08 - loss: 7.5307 - accuracy: 0.5089
 6688/25000 [=======>......................] - ETA: 1:08 - loss: 7.5314 - accuracy: 0.5088
 6720/25000 [=======>......................] - ETA: 1:08 - loss: 7.5274 - accuracy: 0.5091
 6752/25000 [=======>......................] - ETA: 1:08 - loss: 7.5258 - accuracy: 0.5092
 6784/25000 [=======>......................] - ETA: 1:08 - loss: 7.5265 - accuracy: 0.5091
 6816/25000 [=======>......................] - ETA: 1:08 - loss: 7.5316 - accuracy: 0.5088
 6848/25000 [=======>......................] - ETA: 1:08 - loss: 7.5345 - accuracy: 0.5086
 6880/25000 [=======>......................] - ETA: 1:07 - loss: 7.5374 - accuracy: 0.5084
 6912/25000 [=======>......................] - ETA: 1:07 - loss: 7.5313 - accuracy: 0.5088
 6944/25000 [=======>......................] - ETA: 1:07 - loss: 7.5253 - accuracy: 0.5092
 6976/25000 [=======>......................] - ETA: 1:07 - loss: 7.5216 - accuracy: 0.5095
 7008/25000 [=======>......................] - ETA: 1:07 - loss: 7.5222 - accuracy: 0.5094
 7040/25000 [=======>......................] - ETA: 1:07 - loss: 7.5359 - accuracy: 0.5085
 7072/25000 [=======>......................] - ETA: 1:07 - loss: 7.5344 - accuracy: 0.5086
 7104/25000 [=======>......................] - ETA: 1:07 - loss: 7.5393 - accuracy: 0.5083
 7136/25000 [=======>......................] - ETA: 1:06 - loss: 7.5377 - accuracy: 0.5084
 7168/25000 [=======>......................] - ETA: 1:06 - loss: 7.5468 - accuracy: 0.5078
 7200/25000 [=======>......................] - ETA: 1:06 - loss: 7.5580 - accuracy: 0.5071
 7232/25000 [=======>......................] - ETA: 1:06 - loss: 7.5585 - accuracy: 0.5071
 7264/25000 [=======>......................] - ETA: 1:06 - loss: 7.5674 - accuracy: 0.5065
 7296/25000 [=======>......................] - ETA: 1:06 - loss: 7.5720 - accuracy: 0.5062
 7328/25000 [=======>......................] - ETA: 1:06 - loss: 7.5725 - accuracy: 0.5061
 7360/25000 [=======>......................] - ETA: 1:06 - loss: 7.5645 - accuracy: 0.5067
 7392/25000 [=======>......................] - ETA: 1:06 - loss: 7.5671 - accuracy: 0.5065
 7424/25000 [=======>......................] - ETA: 1:05 - loss: 7.5654 - accuracy: 0.5066
 7456/25000 [=======>......................] - ETA: 1:05 - loss: 7.5700 - accuracy: 0.5063
 7488/25000 [=======>......................] - ETA: 1:05 - loss: 7.5724 - accuracy: 0.5061
 7520/25000 [========>.....................] - ETA: 1:05 - loss: 7.5830 - accuracy: 0.5055
 7552/25000 [========>.....................] - ETA: 1:05 - loss: 7.5813 - accuracy: 0.5056
 7584/25000 [========>.....................] - ETA: 1:05 - loss: 7.5837 - accuracy: 0.5054
 7616/25000 [========>.....................] - ETA: 1:05 - loss: 7.5921 - accuracy: 0.5049
 7648/25000 [========>.....................] - ETA: 1:05 - loss: 7.5924 - accuracy: 0.5048
 7680/25000 [========>.....................] - ETA: 1:04 - loss: 7.5927 - accuracy: 0.5048
 7712/25000 [========>.....................] - ETA: 1:04 - loss: 7.5831 - accuracy: 0.5054
 7744/25000 [========>.....................] - ETA: 1:04 - loss: 7.5894 - accuracy: 0.5050
 7776/25000 [========>.....................] - ETA: 1:04 - loss: 7.5917 - accuracy: 0.5049
 7808/25000 [========>.....................] - ETA: 1:04 - loss: 7.6038 - accuracy: 0.5041
 7840/25000 [========>.....................] - ETA: 1:04 - loss: 7.6001 - accuracy: 0.5043
 7872/25000 [========>.....................] - ETA: 1:04 - loss: 7.5946 - accuracy: 0.5047
 7904/25000 [========>.....................] - ETA: 1:04 - loss: 7.5987 - accuracy: 0.5044
 7936/25000 [========>.....................] - ETA: 1:04 - loss: 7.5971 - accuracy: 0.5045
 7968/25000 [========>.....................] - ETA: 1:03 - loss: 7.5916 - accuracy: 0.5049
 8000/25000 [========>.....................] - ETA: 1:03 - loss: 7.5861 - accuracy: 0.5052
 8032/25000 [========>.....................] - ETA: 1:03 - loss: 7.5960 - accuracy: 0.5046
 8064/25000 [========>.....................] - ETA: 1:03 - loss: 7.5963 - accuracy: 0.5046
 8096/25000 [========>.....................] - ETA: 1:03 - loss: 7.6022 - accuracy: 0.5042
 8128/25000 [========>.....................] - ETA: 1:03 - loss: 7.6025 - accuracy: 0.5042
 8160/25000 [========>.....................] - ETA: 1:03 - loss: 7.6027 - accuracy: 0.5042
 8192/25000 [========>.....................] - ETA: 1:03 - loss: 7.5955 - accuracy: 0.5046
 8224/25000 [========>.....................] - ETA: 1:03 - loss: 7.5995 - accuracy: 0.5044
 8256/25000 [========>.....................] - ETA: 1:02 - loss: 7.6016 - accuracy: 0.5042
 8288/25000 [========>.....................] - ETA: 1:02 - loss: 7.6019 - accuracy: 0.5042
 8320/25000 [========>.....................] - ETA: 1:02 - loss: 7.6058 - accuracy: 0.5040
 8352/25000 [=========>....................] - ETA: 1:02 - loss: 7.6079 - accuracy: 0.5038
 8384/25000 [=========>....................] - ETA: 1:02 - loss: 7.6136 - accuracy: 0.5035
 8416/25000 [=========>....................] - ETA: 1:02 - loss: 7.6138 - accuracy: 0.5034
 8448/25000 [=========>....................] - ETA: 1:02 - loss: 7.6104 - accuracy: 0.5037
 8480/25000 [=========>....................] - ETA: 1:02 - loss: 7.6033 - accuracy: 0.5041
 8512/25000 [=========>....................] - ETA: 1:01 - loss: 7.6054 - accuracy: 0.5040
 8544/25000 [=========>....................] - ETA: 1:01 - loss: 7.6164 - accuracy: 0.5033
 8576/25000 [=========>....................] - ETA: 1:01 - loss: 7.6166 - accuracy: 0.5033
 8608/25000 [=========>....................] - ETA: 1:01 - loss: 7.6132 - accuracy: 0.5035
 8640/25000 [=========>....................] - ETA: 1:01 - loss: 7.6134 - accuracy: 0.5035
 8672/25000 [=========>....................] - ETA: 1:01 - loss: 7.6224 - accuracy: 0.5029
 8704/25000 [=========>....................] - ETA: 1:01 - loss: 7.6173 - accuracy: 0.5032
 8736/25000 [=========>....................] - ETA: 1:01 - loss: 7.6069 - accuracy: 0.5039
 8768/25000 [=========>....................] - ETA: 1:00 - loss: 7.6054 - accuracy: 0.5040
 8800/25000 [=========>....................] - ETA: 1:00 - loss: 7.6091 - accuracy: 0.5038
 8832/25000 [=========>....................] - ETA: 1:00 - loss: 7.6163 - accuracy: 0.5033
 8864/25000 [=========>....................] - ETA: 1:00 - loss: 7.6147 - accuracy: 0.5034
 8896/25000 [=========>....................] - ETA: 1:00 - loss: 7.6184 - accuracy: 0.5031
 8928/25000 [=========>....................] - ETA: 1:00 - loss: 7.6151 - accuracy: 0.5034
 8960/25000 [=========>....................] - ETA: 1:00 - loss: 7.6187 - accuracy: 0.5031
 8992/25000 [=========>....................] - ETA: 1:00 - loss: 7.6138 - accuracy: 0.5034
 9024/25000 [=========>....................] - ETA: 59s - loss: 7.6173 - accuracy: 0.5032 
 9056/25000 [=========>....................] - ETA: 59s - loss: 7.6158 - accuracy: 0.5033
 9088/25000 [=========>....................] - ETA: 59s - loss: 7.6109 - accuracy: 0.5036
 9120/25000 [=========>....................] - ETA: 59s - loss: 7.6128 - accuracy: 0.5035
 9152/25000 [=========>....................] - ETA: 59s - loss: 7.6130 - accuracy: 0.5035
 9184/25000 [==========>...................] - ETA: 59s - loss: 7.6149 - accuracy: 0.5034
 9216/25000 [==========>...................] - ETA: 59s - loss: 7.6167 - accuracy: 0.5033
 9248/25000 [==========>...................] - ETA: 59s - loss: 7.6252 - accuracy: 0.5027
 9280/25000 [==========>...................] - ETA: 59s - loss: 7.6319 - accuracy: 0.5023
 9312/25000 [==========>...................] - ETA: 58s - loss: 7.6370 - accuracy: 0.5019
 9344/25000 [==========>...................] - ETA: 58s - loss: 7.6240 - accuracy: 0.5028
 9376/25000 [==========>...................] - ETA: 58s - loss: 7.6241 - accuracy: 0.5028
 9408/25000 [==========>...................] - ETA: 58s - loss: 7.6194 - accuracy: 0.5031
 9440/25000 [==========>...................] - ETA: 58s - loss: 7.6098 - accuracy: 0.5037
 9472/25000 [==========>...................] - ETA: 58s - loss: 7.6100 - accuracy: 0.5037
 9504/25000 [==========>...................] - ETA: 58s - loss: 7.6150 - accuracy: 0.5034
 9536/25000 [==========>...................] - ETA: 58s - loss: 7.6119 - accuracy: 0.5036
 9568/25000 [==========>...................] - ETA: 57s - loss: 7.6153 - accuracy: 0.5033
 9600/25000 [==========>...................] - ETA: 57s - loss: 7.6139 - accuracy: 0.5034
 9632/25000 [==========>...................] - ETA: 57s - loss: 7.6141 - accuracy: 0.5034
 9664/25000 [==========>...................] - ETA: 57s - loss: 7.6127 - accuracy: 0.5035
 9696/25000 [==========>...................] - ETA: 57s - loss: 7.6129 - accuracy: 0.5035
 9728/25000 [==========>...................] - ETA: 57s - loss: 7.6115 - accuracy: 0.5036
 9760/25000 [==========>...................] - ETA: 57s - loss: 7.6116 - accuracy: 0.5036
 9792/25000 [==========>...................] - ETA: 57s - loss: 7.6102 - accuracy: 0.5037
 9824/25000 [==========>...................] - ETA: 56s - loss: 7.6104 - accuracy: 0.5037
 9856/25000 [==========>...................] - ETA: 56s - loss: 7.6122 - accuracy: 0.5036
 9888/25000 [==========>...................] - ETA: 56s - loss: 7.6092 - accuracy: 0.5037
 9920/25000 [==========>...................] - ETA: 56s - loss: 7.6002 - accuracy: 0.5043
 9952/25000 [==========>...................] - ETA: 56s - loss: 7.6019 - accuracy: 0.5042
 9984/25000 [==========>...................] - ETA: 56s - loss: 7.6006 - accuracy: 0.5043
10016/25000 [===========>..................] - ETA: 56s - loss: 7.5962 - accuracy: 0.5046
10048/25000 [===========>..................] - ETA: 56s - loss: 7.5918 - accuracy: 0.5049
10080/25000 [===========>..................] - ETA: 55s - loss: 7.5890 - accuracy: 0.5051
10112/25000 [===========>..................] - ETA: 55s - loss: 7.5969 - accuracy: 0.5045
10144/25000 [===========>..................] - ETA: 55s - loss: 7.5971 - accuracy: 0.5045
10176/25000 [===========>..................] - ETA: 55s - loss: 7.5928 - accuracy: 0.5048
10208/25000 [===========>..................] - ETA: 55s - loss: 7.5870 - accuracy: 0.5052
10240/25000 [===========>..................] - ETA: 55s - loss: 7.5858 - accuracy: 0.5053
10272/25000 [===========>..................] - ETA: 55s - loss: 7.5845 - accuracy: 0.5054
10304/25000 [===========>..................] - ETA: 55s - loss: 7.5892 - accuracy: 0.5050
10336/25000 [===========>..................] - ETA: 55s - loss: 7.5939 - accuracy: 0.5047
10368/25000 [===========>..................] - ETA: 54s - loss: 7.5942 - accuracy: 0.5047
10400/25000 [===========>..................] - ETA: 54s - loss: 7.5959 - accuracy: 0.5046
10432/25000 [===========>..................] - ETA: 54s - loss: 7.5975 - accuracy: 0.5045
10464/25000 [===========>..................] - ETA: 54s - loss: 7.6021 - accuracy: 0.5042
10496/25000 [===========>..................] - ETA: 54s - loss: 7.6009 - accuracy: 0.5043
10528/25000 [===========>..................] - ETA: 54s - loss: 7.6011 - accuracy: 0.5043
10560/25000 [===========>..................] - ETA: 54s - loss: 7.6027 - accuracy: 0.5042
10592/25000 [===========>..................] - ETA: 53s - loss: 7.5986 - accuracy: 0.5044
10624/25000 [===========>..................] - ETA: 53s - loss: 7.5988 - accuracy: 0.5044
10656/25000 [===========>..................] - ETA: 53s - loss: 7.5976 - accuracy: 0.5045
10688/25000 [===========>..................] - ETA: 53s - loss: 7.5935 - accuracy: 0.5048
10720/25000 [===========>..................] - ETA: 53s - loss: 7.5937 - accuracy: 0.5048
10752/25000 [===========>..................] - ETA: 53s - loss: 7.5925 - accuracy: 0.5048
10784/25000 [===========>..................] - ETA: 53s - loss: 7.5870 - accuracy: 0.5052
10816/25000 [===========>..................] - ETA: 53s - loss: 7.5816 - accuracy: 0.5055
10848/25000 [============>.................] - ETA: 53s - loss: 7.5832 - accuracy: 0.5054
10880/25000 [============>.................] - ETA: 52s - loss: 7.5807 - accuracy: 0.5056
10912/25000 [============>.................] - ETA: 52s - loss: 7.5823 - accuracy: 0.5055
10944/25000 [============>.................] - ETA: 52s - loss: 7.5868 - accuracy: 0.5052
10976/25000 [============>.................] - ETA: 52s - loss: 7.5926 - accuracy: 0.5048
11008/25000 [============>.................] - ETA: 52s - loss: 7.5928 - accuracy: 0.5048
11040/25000 [============>.................] - ETA: 52s - loss: 7.5972 - accuracy: 0.5045
11072/25000 [============>.................] - ETA: 52s - loss: 7.5988 - accuracy: 0.5044
11104/25000 [============>.................] - ETA: 51s - loss: 7.5990 - accuracy: 0.5044
11136/25000 [============>.................] - ETA: 51s - loss: 7.5964 - accuracy: 0.5046
11168/25000 [============>.................] - ETA: 51s - loss: 7.5939 - accuracy: 0.5047
11200/25000 [============>.................] - ETA: 51s - loss: 7.5886 - accuracy: 0.5051
11232/25000 [============>.................] - ETA: 51s - loss: 7.5902 - accuracy: 0.5050
11264/25000 [============>.................] - ETA: 51s - loss: 7.5849 - accuracy: 0.5053
11296/25000 [============>.................] - ETA: 51s - loss: 7.5879 - accuracy: 0.5051
11328/25000 [============>.................] - ETA: 51s - loss: 7.5881 - accuracy: 0.5051
11360/25000 [============>.................] - ETA: 51s - loss: 7.5829 - accuracy: 0.5055
11392/25000 [============>.................] - ETA: 50s - loss: 7.5791 - accuracy: 0.5057
11424/25000 [============>.................] - ETA: 50s - loss: 7.5740 - accuracy: 0.5060
11456/25000 [============>.................] - ETA: 50s - loss: 7.5743 - accuracy: 0.5060
11488/25000 [============>.................] - ETA: 50s - loss: 7.5759 - accuracy: 0.5059
11520/25000 [============>.................] - ETA: 50s - loss: 7.5708 - accuracy: 0.5063
11552/25000 [============>.................] - ETA: 50s - loss: 7.5750 - accuracy: 0.5060
11584/25000 [============>.................] - ETA: 50s - loss: 7.5713 - accuracy: 0.5062
11616/25000 [============>.................] - ETA: 50s - loss: 7.5716 - accuracy: 0.5062
11648/25000 [============>.................] - ETA: 49s - loss: 7.5745 - accuracy: 0.5060
11680/25000 [=============>................] - ETA: 49s - loss: 7.5721 - accuracy: 0.5062
11712/25000 [=============>................] - ETA: 49s - loss: 7.5789 - accuracy: 0.5057
11744/25000 [=============>................] - ETA: 49s - loss: 7.5831 - accuracy: 0.5054
11776/25000 [=============>................] - ETA: 49s - loss: 7.5859 - accuracy: 0.5053
11808/25000 [=============>................] - ETA: 49s - loss: 7.5913 - accuracy: 0.5049
11840/25000 [=============>................] - ETA: 49s - loss: 7.5928 - accuracy: 0.5048
11872/25000 [=============>................] - ETA: 49s - loss: 7.5956 - accuracy: 0.5046
11904/25000 [=============>................] - ETA: 48s - loss: 7.5945 - accuracy: 0.5047
11936/25000 [=============>................] - ETA: 48s - loss: 7.5883 - accuracy: 0.5051
11968/25000 [=============>................] - ETA: 48s - loss: 7.5885 - accuracy: 0.5051
12000/25000 [=============>................] - ETA: 48s - loss: 7.5925 - accuracy: 0.5048
12032/25000 [=============>................] - ETA: 48s - loss: 7.5876 - accuracy: 0.5052
12064/25000 [=============>................] - ETA: 48s - loss: 7.5878 - accuracy: 0.5051
12096/25000 [=============>................] - ETA: 48s - loss: 7.5893 - accuracy: 0.5050
12128/25000 [=============>................] - ETA: 48s - loss: 7.5933 - accuracy: 0.5048
12160/25000 [=============>................] - ETA: 47s - loss: 7.5897 - accuracy: 0.5050
12192/25000 [=============>................] - ETA: 47s - loss: 7.5912 - accuracy: 0.5049
12224/25000 [=============>................] - ETA: 47s - loss: 7.5951 - accuracy: 0.5047
12256/25000 [=============>................] - ETA: 47s - loss: 7.5941 - accuracy: 0.5047
12288/25000 [=============>................] - ETA: 47s - loss: 7.5868 - accuracy: 0.5052
12320/25000 [=============>................] - ETA: 47s - loss: 7.5857 - accuracy: 0.5053
12352/25000 [=============>................] - ETA: 47s - loss: 7.5884 - accuracy: 0.5051
12384/25000 [=============>................] - ETA: 47s - loss: 7.5886 - accuracy: 0.5051
12416/25000 [=============>................] - ETA: 46s - loss: 7.5888 - accuracy: 0.5051
12448/25000 [=============>................] - ETA: 46s - loss: 7.5915 - accuracy: 0.5049
12480/25000 [=============>................] - ETA: 46s - loss: 7.5892 - accuracy: 0.5050
12512/25000 [==============>...............] - ETA: 46s - loss: 7.5906 - accuracy: 0.5050
12544/25000 [==============>...............] - ETA: 46s - loss: 7.5908 - accuracy: 0.5049
12576/25000 [==============>...............] - ETA: 46s - loss: 7.5874 - accuracy: 0.5052
12608/25000 [==============>...............] - ETA: 46s - loss: 7.5900 - accuracy: 0.5050
12640/25000 [==============>...............] - ETA: 46s - loss: 7.5841 - accuracy: 0.5054
12672/25000 [==============>...............] - ETA: 46s - loss: 7.5843 - accuracy: 0.5054
12704/25000 [==============>...............] - ETA: 45s - loss: 7.5821 - accuracy: 0.5055
12736/25000 [==============>...............] - ETA: 45s - loss: 7.5860 - accuracy: 0.5053
12768/25000 [==============>...............] - ETA: 45s - loss: 7.5886 - accuracy: 0.5051
12800/25000 [==============>...............] - ETA: 45s - loss: 7.5947 - accuracy: 0.5047
12832/25000 [==============>...............] - ETA: 45s - loss: 7.5913 - accuracy: 0.5049
12864/25000 [==============>...............] - ETA: 45s - loss: 7.5951 - accuracy: 0.5047
12896/25000 [==============>...............] - ETA: 45s - loss: 7.5905 - accuracy: 0.5050
12928/25000 [==============>...............] - ETA: 45s - loss: 7.5966 - accuracy: 0.5046
12960/25000 [==============>...............] - ETA: 44s - loss: 7.5992 - accuracy: 0.5044
12992/25000 [==============>...............] - ETA: 44s - loss: 7.5993 - accuracy: 0.5044
13024/25000 [==============>...............] - ETA: 44s - loss: 7.5960 - accuracy: 0.5046
13056/25000 [==============>...............] - ETA: 44s - loss: 7.5938 - accuracy: 0.5047
13088/25000 [==============>...............] - ETA: 44s - loss: 7.5975 - accuracy: 0.5045
13120/25000 [==============>...............] - ETA: 44s - loss: 7.5977 - accuracy: 0.5045
13152/25000 [==============>...............] - ETA: 44s - loss: 7.5943 - accuracy: 0.5047
13184/25000 [==============>...............] - ETA: 44s - loss: 7.5945 - accuracy: 0.5047
13216/25000 [==============>...............] - ETA: 43s - loss: 7.5982 - accuracy: 0.5045
13248/25000 [==============>...............] - ETA: 43s - loss: 7.6018 - accuracy: 0.5042
13280/25000 [==============>...............] - ETA: 43s - loss: 7.5973 - accuracy: 0.5045
13312/25000 [==============>...............] - ETA: 43s - loss: 7.5987 - accuracy: 0.5044
13344/25000 [===============>..............] - ETA: 43s - loss: 7.5988 - accuracy: 0.5044
13376/25000 [===============>..............] - ETA: 43s - loss: 7.5967 - accuracy: 0.5046
13408/25000 [===============>..............] - ETA: 43s - loss: 7.6003 - accuracy: 0.5043
13440/25000 [===============>..............] - ETA: 43s - loss: 7.6016 - accuracy: 0.5042
13472/25000 [===============>..............] - ETA: 42s - loss: 7.6017 - accuracy: 0.5042
13504/25000 [===============>..............] - ETA: 42s - loss: 7.6030 - accuracy: 0.5041
13536/25000 [===============>..............] - ETA: 42s - loss: 7.5975 - accuracy: 0.5045
13568/25000 [===============>..............] - ETA: 42s - loss: 7.5932 - accuracy: 0.5048
13600/25000 [===============>..............] - ETA: 42s - loss: 7.5911 - accuracy: 0.5049
13632/25000 [===============>..............] - ETA: 42s - loss: 7.5901 - accuracy: 0.5050
13664/25000 [===============>..............] - ETA: 42s - loss: 7.5869 - accuracy: 0.5052
13696/25000 [===============>..............] - ETA: 42s - loss: 7.5894 - accuracy: 0.5050
13728/25000 [===============>..............] - ETA: 41s - loss: 7.5884 - accuracy: 0.5051
13760/25000 [===============>..............] - ETA: 41s - loss: 7.5875 - accuracy: 0.5052
13792/25000 [===============>..............] - ETA: 41s - loss: 7.5832 - accuracy: 0.5054
13824/25000 [===============>..............] - ETA: 41s - loss: 7.5868 - accuracy: 0.5052
13856/25000 [===============>..............] - ETA: 41s - loss: 7.5958 - accuracy: 0.5046
13888/25000 [===============>..............] - ETA: 41s - loss: 7.6015 - accuracy: 0.5042
13920/25000 [===============>..............] - ETA: 41s - loss: 7.5983 - accuracy: 0.5045
13952/25000 [===============>..............] - ETA: 41s - loss: 7.5974 - accuracy: 0.5045
13984/25000 [===============>..............] - ETA: 41s - loss: 7.5997 - accuracy: 0.5044
14016/25000 [===============>..............] - ETA: 40s - loss: 7.6021 - accuracy: 0.5042
14048/25000 [===============>..............] - ETA: 40s - loss: 7.6088 - accuracy: 0.5038
14080/25000 [===============>..............] - ETA: 40s - loss: 7.6078 - accuracy: 0.5038
14112/25000 [===============>..............] - ETA: 40s - loss: 7.6047 - accuracy: 0.5040
14144/25000 [===============>..............] - ETA: 40s - loss: 7.6037 - accuracy: 0.5041
14176/25000 [================>.............] - ETA: 40s - loss: 7.6028 - accuracy: 0.5042
14208/25000 [================>.............] - ETA: 40s - loss: 7.5997 - accuracy: 0.5044
14240/25000 [================>.............] - ETA: 40s - loss: 7.5977 - accuracy: 0.5045
14272/25000 [================>.............] - ETA: 39s - loss: 7.5979 - accuracy: 0.5045
14304/25000 [================>.............] - ETA: 39s - loss: 7.6034 - accuracy: 0.5041
14336/25000 [================>.............] - ETA: 39s - loss: 7.6046 - accuracy: 0.5040
14368/25000 [================>.............] - ETA: 39s - loss: 7.6079 - accuracy: 0.5038
14400/25000 [================>.............] - ETA: 39s - loss: 7.6059 - accuracy: 0.5040
14432/25000 [================>.............] - ETA: 39s - loss: 7.6071 - accuracy: 0.5039
14464/25000 [================>.............] - ETA: 39s - loss: 7.6083 - accuracy: 0.5038
14496/25000 [================>.............] - ETA: 39s - loss: 7.6095 - accuracy: 0.5037
14528/25000 [================>.............] - ETA: 38s - loss: 7.6107 - accuracy: 0.5036
14560/25000 [================>.............] - ETA: 38s - loss: 7.6150 - accuracy: 0.5034
14592/25000 [================>.............] - ETA: 38s - loss: 7.6204 - accuracy: 0.5030
14624/25000 [================>.............] - ETA: 38s - loss: 7.6215 - accuracy: 0.5029
14656/25000 [================>.............] - ETA: 38s - loss: 7.6206 - accuracy: 0.5030
14688/25000 [================>.............] - ETA: 38s - loss: 7.6280 - accuracy: 0.5025
14720/25000 [================>.............] - ETA: 38s - loss: 7.6333 - accuracy: 0.5022
14752/25000 [================>.............] - ETA: 38s - loss: 7.6302 - accuracy: 0.5024
14784/25000 [================>.............] - ETA: 37s - loss: 7.6262 - accuracy: 0.5026
14816/25000 [================>.............] - ETA: 37s - loss: 7.6273 - accuracy: 0.5026
14848/25000 [================>.............] - ETA: 37s - loss: 7.6284 - accuracy: 0.5025
14880/25000 [================>.............] - ETA: 37s - loss: 7.6254 - accuracy: 0.5027
14912/25000 [================>.............] - ETA: 37s - loss: 7.6245 - accuracy: 0.5027
14944/25000 [================>.............] - ETA: 37s - loss: 7.6266 - accuracy: 0.5026
14976/25000 [================>.............] - ETA: 37s - loss: 7.6246 - accuracy: 0.5027
15008/25000 [=================>............] - ETA: 37s - loss: 7.6268 - accuracy: 0.5026
15040/25000 [=================>............] - ETA: 37s - loss: 7.6320 - accuracy: 0.5023
15072/25000 [=================>............] - ETA: 36s - loss: 7.6371 - accuracy: 0.5019
15104/25000 [=================>............] - ETA: 36s - loss: 7.6311 - accuracy: 0.5023
15136/25000 [=================>............] - ETA: 36s - loss: 7.6352 - accuracy: 0.5020
15168/25000 [=================>............] - ETA: 36s - loss: 7.6322 - accuracy: 0.5022
15200/25000 [=================>............] - ETA: 36s - loss: 7.6293 - accuracy: 0.5024
15232/25000 [=================>............] - ETA: 36s - loss: 7.6314 - accuracy: 0.5023
15264/25000 [=================>............] - ETA: 36s - loss: 7.6365 - accuracy: 0.5020
15296/25000 [=================>............] - ETA: 36s - loss: 7.6335 - accuracy: 0.5022
15328/25000 [=================>............] - ETA: 35s - loss: 7.6366 - accuracy: 0.5020
15360/25000 [=================>............] - ETA: 35s - loss: 7.6357 - accuracy: 0.5020
15392/25000 [=================>............] - ETA: 35s - loss: 7.6397 - accuracy: 0.5018
15424/25000 [=================>............] - ETA: 35s - loss: 7.6398 - accuracy: 0.5018
15456/25000 [=================>............] - ETA: 35s - loss: 7.6398 - accuracy: 0.5017
15488/25000 [=================>............] - ETA: 35s - loss: 7.6379 - accuracy: 0.5019
15520/25000 [=================>............] - ETA: 35s - loss: 7.6409 - accuracy: 0.5017
15552/25000 [=================>............] - ETA: 35s - loss: 7.6479 - accuracy: 0.5012
15584/25000 [=================>............] - ETA: 35s - loss: 7.6450 - accuracy: 0.5014
15616/25000 [=================>............] - ETA: 34s - loss: 7.6470 - accuracy: 0.5013
15648/25000 [=================>............] - ETA: 34s - loss: 7.6460 - accuracy: 0.5013
15680/25000 [=================>............] - ETA: 34s - loss: 7.6451 - accuracy: 0.5014
15712/25000 [=================>............] - ETA: 34s - loss: 7.6451 - accuracy: 0.5014
15744/25000 [=================>............] - ETA: 34s - loss: 7.6423 - accuracy: 0.5016
15776/25000 [=================>............] - ETA: 34s - loss: 7.6394 - accuracy: 0.5018
15808/25000 [=================>............] - ETA: 34s - loss: 7.6404 - accuracy: 0.5017
15840/25000 [==================>...........] - ETA: 34s - loss: 7.6444 - accuracy: 0.5015
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6444 - accuracy: 0.5014
15904/25000 [==================>...........] - ETA: 33s - loss: 7.6444 - accuracy: 0.5014
15936/25000 [==================>...........] - ETA: 33s - loss: 7.6474 - accuracy: 0.5013
15968/25000 [==================>...........] - ETA: 33s - loss: 7.6445 - accuracy: 0.5014
16000/25000 [==================>...........] - ETA: 33s - loss: 7.6465 - accuracy: 0.5013
16032/25000 [==================>...........] - ETA: 33s - loss: 7.6456 - accuracy: 0.5014
16064/25000 [==================>...........] - ETA: 33s - loss: 7.6456 - accuracy: 0.5014
16096/25000 [==================>...........] - ETA: 33s - loss: 7.6457 - accuracy: 0.5014
16128/25000 [==================>...........] - ETA: 32s - loss: 7.6438 - accuracy: 0.5015
16160/25000 [==================>...........] - ETA: 32s - loss: 7.6457 - accuracy: 0.5014
16192/25000 [==================>...........] - ETA: 32s - loss: 7.6448 - accuracy: 0.5014
16224/25000 [==================>...........] - ETA: 32s - loss: 7.6458 - accuracy: 0.5014
16256/25000 [==================>...........] - ETA: 32s - loss: 7.6430 - accuracy: 0.5015
16288/25000 [==================>...........] - ETA: 32s - loss: 7.6403 - accuracy: 0.5017
16320/25000 [==================>...........] - ETA: 32s - loss: 7.6356 - accuracy: 0.5020
16352/25000 [==================>...........] - ETA: 32s - loss: 7.6319 - accuracy: 0.5023
16384/25000 [==================>...........] - ETA: 32s - loss: 7.6292 - accuracy: 0.5024
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6283 - accuracy: 0.5025
16448/25000 [==================>...........] - ETA: 31s - loss: 7.6303 - accuracy: 0.5024
16480/25000 [==================>...........] - ETA: 31s - loss: 7.6303 - accuracy: 0.5024
16512/25000 [==================>...........] - ETA: 31s - loss: 7.6360 - accuracy: 0.5020
16544/25000 [==================>...........] - ETA: 31s - loss: 7.6351 - accuracy: 0.5021
16576/25000 [==================>...........] - ETA: 31s - loss: 7.6370 - accuracy: 0.5019
16608/25000 [==================>...........] - ETA: 31s - loss: 7.6343 - accuracy: 0.5021
16640/25000 [==================>...........] - ETA: 31s - loss: 7.6362 - accuracy: 0.5020
16672/25000 [===================>..........] - ETA: 30s - loss: 7.6399 - accuracy: 0.5017
16704/25000 [===================>..........] - ETA: 30s - loss: 7.6409 - accuracy: 0.5017
16736/25000 [===================>..........] - ETA: 30s - loss: 7.6364 - accuracy: 0.5020
16768/25000 [===================>..........] - ETA: 30s - loss: 7.6383 - accuracy: 0.5018
16800/25000 [===================>..........] - ETA: 30s - loss: 7.6347 - accuracy: 0.5021
16832/25000 [===================>..........] - ETA: 30s - loss: 7.6320 - accuracy: 0.5023
16864/25000 [===================>..........] - ETA: 30s - loss: 7.6321 - accuracy: 0.5023
16896/25000 [===================>..........] - ETA: 30s - loss: 7.6312 - accuracy: 0.5023
16928/25000 [===================>..........] - ETA: 30s - loss: 7.6277 - accuracy: 0.5025
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6314 - accuracy: 0.5023
16992/25000 [===================>..........] - ETA: 29s - loss: 7.6350 - accuracy: 0.5021
17024/25000 [===================>..........] - ETA: 29s - loss: 7.6306 - accuracy: 0.5023
17056/25000 [===================>..........] - ETA: 29s - loss: 7.6316 - accuracy: 0.5023
17088/25000 [===================>..........] - ETA: 29s - loss: 7.6325 - accuracy: 0.5022
17120/25000 [===================>..........] - ETA: 29s - loss: 7.6281 - accuracy: 0.5025
17152/25000 [===================>..........] - ETA: 29s - loss: 7.6309 - accuracy: 0.5023
17184/25000 [===================>..........] - ETA: 29s - loss: 7.6372 - accuracy: 0.5019
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6381 - accuracy: 0.5019
17248/25000 [===================>..........] - ETA: 28s - loss: 7.6391 - accuracy: 0.5018
17280/25000 [===================>..........] - ETA: 28s - loss: 7.6391 - accuracy: 0.5018
17312/25000 [===================>..........] - ETA: 28s - loss: 7.6409 - accuracy: 0.5017
17344/25000 [===================>..........] - ETA: 28s - loss: 7.6436 - accuracy: 0.5015
17376/25000 [===================>..........] - ETA: 28s - loss: 7.6437 - accuracy: 0.5015
17408/25000 [===================>..........] - ETA: 28s - loss: 7.6464 - accuracy: 0.5013
17440/25000 [===================>..........] - ETA: 28s - loss: 7.6464 - accuracy: 0.5013
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6464 - accuracy: 0.5013
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6500 - accuracy: 0.5011
17536/25000 [====================>.........] - ETA: 27s - loss: 7.6483 - accuracy: 0.5012
17568/25000 [====================>.........] - ETA: 27s - loss: 7.6500 - accuracy: 0.5011
17600/25000 [====================>.........] - ETA: 27s - loss: 7.6483 - accuracy: 0.5012
17632/25000 [====================>.........] - ETA: 27s - loss: 7.6501 - accuracy: 0.5011
17664/25000 [====================>.........] - ETA: 27s - loss: 7.6519 - accuracy: 0.5010
17696/25000 [====================>.........] - ETA: 27s - loss: 7.6536 - accuracy: 0.5008
17728/25000 [====================>.........] - ETA: 27s - loss: 7.6554 - accuracy: 0.5007
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6588 - accuracy: 0.5005
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6632 - accuracy: 0.5002
17824/25000 [====================>.........] - ETA: 26s - loss: 7.6589 - accuracy: 0.5005
17856/25000 [====================>.........] - ETA: 26s - loss: 7.6572 - accuracy: 0.5006
17888/25000 [====================>.........] - ETA: 26s - loss: 7.6546 - accuracy: 0.5008
17920/25000 [====================>.........] - ETA: 26s - loss: 7.6538 - accuracy: 0.5008
17952/25000 [====================>.........] - ETA: 26s - loss: 7.6555 - accuracy: 0.5007
17984/25000 [====================>.........] - ETA: 26s - loss: 7.6615 - accuracy: 0.5003
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6641 - accuracy: 0.5002
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6658 - accuracy: 0.5001
18080/25000 [====================>.........] - ETA: 25s - loss: 7.6675 - accuracy: 0.4999
18112/25000 [====================>.........] - ETA: 25s - loss: 7.6649 - accuracy: 0.5001
18144/25000 [====================>.........] - ETA: 25s - loss: 7.6641 - accuracy: 0.5002
18176/25000 [====================>.........] - ETA: 25s - loss: 7.6632 - accuracy: 0.5002
18208/25000 [====================>.........] - ETA: 25s - loss: 7.6641 - accuracy: 0.5002
18240/25000 [====================>.........] - ETA: 25s - loss: 7.6683 - accuracy: 0.4999
18272/25000 [====================>.........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6658 - accuracy: 0.5001
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18368/25000 [=====================>........] - ETA: 24s - loss: 7.6708 - accuracy: 0.4997
18400/25000 [=====================>........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18432/25000 [=====================>........] - ETA: 24s - loss: 7.6699 - accuracy: 0.4998
18464/25000 [=====================>........] - ETA: 24s - loss: 7.6708 - accuracy: 0.4997
18496/25000 [=====================>........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18528/25000 [=====================>........] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6633 - accuracy: 0.5002
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6674 - accuracy: 0.4999
18624/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18656/25000 [=====================>........] - ETA: 23s - loss: 7.6699 - accuracy: 0.4998
18688/25000 [=====================>........] - ETA: 23s - loss: 7.6699 - accuracy: 0.4998
18720/25000 [=====================>........] - ETA: 23s - loss: 7.6691 - accuracy: 0.4998
18752/25000 [=====================>........] - ETA: 23s - loss: 7.6748 - accuracy: 0.4995
18784/25000 [=====================>........] - ETA: 23s - loss: 7.6748 - accuracy: 0.4995
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6756 - accuracy: 0.4994
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6756 - accuracy: 0.4994
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6780 - accuracy: 0.4993
18912/25000 [=====================>........] - ETA: 22s - loss: 7.6755 - accuracy: 0.4994
18944/25000 [=====================>........] - ETA: 22s - loss: 7.6755 - accuracy: 0.4994
18976/25000 [=====================>........] - ETA: 22s - loss: 7.6739 - accuracy: 0.4995
19008/25000 [=====================>........] - ETA: 22s - loss: 7.6747 - accuracy: 0.4995
19040/25000 [=====================>........] - ETA: 22s - loss: 7.6747 - accuracy: 0.4995
19072/25000 [=====================>........] - ETA: 22s - loss: 7.6755 - accuracy: 0.4994
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6746 - accuracy: 0.4995
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6762 - accuracy: 0.4994
19168/25000 [======================>.......] - ETA: 21s - loss: 7.6762 - accuracy: 0.4994
19200/25000 [======================>.......] - ETA: 21s - loss: 7.6722 - accuracy: 0.4996
19232/25000 [======================>.......] - ETA: 21s - loss: 7.6786 - accuracy: 0.4992
19264/25000 [======================>.......] - ETA: 21s - loss: 7.6770 - accuracy: 0.4993
19296/25000 [======================>.......] - ETA: 21s - loss: 7.6762 - accuracy: 0.4994
19328/25000 [======================>.......] - ETA: 21s - loss: 7.6753 - accuracy: 0.4994
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6722 - accuracy: 0.4996
19424/25000 [======================>.......] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
19456/25000 [======================>.......] - ETA: 20s - loss: 7.6745 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
19520/25000 [======================>.......] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
19552/25000 [======================>.......] - ETA: 20s - loss: 7.6721 - accuracy: 0.4996
19584/25000 [======================>.......] - ETA: 20s - loss: 7.6713 - accuracy: 0.4997
19616/25000 [======================>.......] - ETA: 20s - loss: 7.6713 - accuracy: 0.4997
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6736 - accuracy: 0.4995
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6736 - accuracy: 0.4995
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6705 - accuracy: 0.4997
19744/25000 [======================>.......] - ETA: 19s - loss: 7.6697 - accuracy: 0.4998
19776/25000 [======================>.......] - ETA: 19s - loss: 7.6697 - accuracy: 0.4998
19808/25000 [======================>.......] - ETA: 19s - loss: 7.6658 - accuracy: 0.5001
19840/25000 [======================>.......] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
19872/25000 [======================>.......] - ETA: 19s - loss: 7.6643 - accuracy: 0.5002
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6620 - accuracy: 0.5003
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5002
19968/25000 [======================>.......] - ETA: 18s - loss: 7.6651 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5002
20032/25000 [=======================>......] - ETA: 18s - loss: 7.6659 - accuracy: 0.5000
20064/25000 [=======================>......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5001
20096/25000 [=======================>......] - ETA: 18s - loss: 7.6651 - accuracy: 0.5001
20128/25000 [=======================>......] - ETA: 18s - loss: 7.6636 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6643 - accuracy: 0.5001
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6659 - accuracy: 0.5000
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6659 - accuracy: 0.5000
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6636 - accuracy: 0.5002
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6674 - accuracy: 0.5000
20320/25000 [=======================>......] - ETA: 17s - loss: 7.6659 - accuracy: 0.5000
20352/25000 [=======================>......] - ETA: 17s - loss: 7.6659 - accuracy: 0.5000
20384/25000 [=======================>......] - ETA: 17s - loss: 7.6636 - accuracy: 0.5002
20416/25000 [=======================>......] - ETA: 17s - loss: 7.6644 - accuracy: 0.5001
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6629 - accuracy: 0.5002
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6636 - accuracy: 0.5002
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6659 - accuracy: 0.5000
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6651 - accuracy: 0.5001
20576/25000 [=======================>......] - ETA: 16s - loss: 7.6629 - accuracy: 0.5002
20608/25000 [=======================>......] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
20640/25000 [=======================>......] - ETA: 16s - loss: 7.6681 - accuracy: 0.4999
20672/25000 [=======================>......] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6718 - accuracy: 0.4997
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6755 - accuracy: 0.4994
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6725 - accuracy: 0.4996
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6725 - accuracy: 0.4996
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6732 - accuracy: 0.4996
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6762 - accuracy: 0.4994
20896/25000 [========================>.....] - ETA: 15s - loss: 7.6747 - accuracy: 0.4995
20928/25000 [========================>.....] - ETA: 15s - loss: 7.6783 - accuracy: 0.4992
20960/25000 [========================>.....] - ETA: 15s - loss: 7.6769 - accuracy: 0.4993
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6761 - accuracy: 0.4994
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6732 - accuracy: 0.4996
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6724 - accuracy: 0.4996
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6746 - accuracy: 0.4995
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6710 - accuracy: 0.4997
21152/25000 [========================>.....] - ETA: 14s - loss: 7.6695 - accuracy: 0.4998
21184/25000 [========================>.....] - ETA: 14s - loss: 7.6673 - accuracy: 0.5000
21216/25000 [========================>.....] - ETA: 14s - loss: 7.6695 - accuracy: 0.4998
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6666 - accuracy: 0.5000
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6666 - accuracy: 0.5000
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6645 - accuracy: 0.5001
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6616 - accuracy: 0.5003
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6609 - accuracy: 0.5004
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6645 - accuracy: 0.5001
21440/25000 [========================>.....] - ETA: 13s - loss: 7.6623 - accuracy: 0.5003
21472/25000 [========================>.....] - ETA: 13s - loss: 7.6638 - accuracy: 0.5002
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6602 - accuracy: 0.5004
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6595 - accuracy: 0.5005
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6624 - accuracy: 0.5003
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6631 - accuracy: 0.5002
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6645 - accuracy: 0.5001
21696/25000 [=========================>....] - ETA: 12s - loss: 7.6610 - accuracy: 0.5004
21728/25000 [=========================>....] - ETA: 12s - loss: 7.6610 - accuracy: 0.5004
21760/25000 [=========================>....] - ETA: 12s - loss: 7.6582 - accuracy: 0.5006
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6561 - accuracy: 0.5007
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6582 - accuracy: 0.5005
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6554 - accuracy: 0.5007
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6540 - accuracy: 0.5008
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6540 - accuracy: 0.5008
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6547 - accuracy: 0.5008
21984/25000 [=========================>....] - ETA: 11s - loss: 7.6548 - accuracy: 0.5008
22016/25000 [=========================>....] - ETA: 11s - loss: 7.6541 - accuracy: 0.5008
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6583 - accuracy: 0.5005
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6590 - accuracy: 0.5005
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6604 - accuracy: 0.5004
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6611 - accuracy: 0.5004
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6604 - accuracy: 0.5004
22272/25000 [=========================>....] - ETA: 10s - loss: 7.6611 - accuracy: 0.5004
22304/25000 [=========================>....] - ETA: 10s - loss: 7.6591 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6543 - accuracy: 0.5008 
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6557 - accuracy: 0.5007
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6605 - accuracy: 0.5004
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6612 - accuracy: 0.5004
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6598 - accuracy: 0.5004
22528/25000 [==========================>...] - ETA: 9s - loss: 7.6585 - accuracy: 0.5005
22560/25000 [==========================>...] - ETA: 9s - loss: 7.6578 - accuracy: 0.5006
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6571 - accuracy: 0.5006
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6626 - accuracy: 0.5003
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6592 - accuracy: 0.5005
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6612 - accuracy: 0.5004
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6606 - accuracy: 0.5004
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6633 - accuracy: 0.5002
22816/25000 [==========================>...] - ETA: 8s - loss: 7.6599 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6599 - accuracy: 0.5004
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6566 - accuracy: 0.5007
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6546 - accuracy: 0.5008
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6533 - accuracy: 0.5009
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6533 - accuracy: 0.5009
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6520 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6533 - accuracy: 0.5009
23072/25000 [==========================>...] - ETA: 7s - loss: 7.6520 - accuracy: 0.5010
23104/25000 [==========================>...] - ETA: 7s - loss: 7.6540 - accuracy: 0.5008
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6547 - accuracy: 0.5008
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6554 - accuracy: 0.5007
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6567 - accuracy: 0.5006
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6600 - accuracy: 0.5004
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6574 - accuracy: 0.5006
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6581 - accuracy: 0.5006
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6600 - accuracy: 0.5004
23360/25000 [===========================>..] - ETA: 6s - loss: 7.6594 - accuracy: 0.5005
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6568 - accuracy: 0.5006
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6568 - accuracy: 0.5006
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6536 - accuracy: 0.5009
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6536 - accuracy: 0.5009
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6556 - accuracy: 0.5007
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6556 - accuracy: 0.5007
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6582 - accuracy: 0.5006
23648/25000 [===========================>..] - ETA: 5s - loss: 7.6575 - accuracy: 0.5006
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6582 - accuracy: 0.5005
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6582 - accuracy: 0.5005
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6582 - accuracy: 0.5005
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6570 - accuracy: 0.5006
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6589 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23904/25000 [===========================>..] - ETA: 4s - loss: 7.6576 - accuracy: 0.5006
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6545 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6551 - accuracy: 0.5008
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6558 - accuracy: 0.5007
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6596 - accuracy: 0.5005
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6603 - accuracy: 0.5004
24192/25000 [============================>.] - ETA: 3s - loss: 7.6596 - accuracy: 0.5005
24224/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24256/25000 [============================>.] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24320/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24640/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24704/25000 [============================>.] - ETA: 1s - loss: 7.6617 - accuracy: 0.5003
24736/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24864/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24896/25000 [============================>.] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
24928/25000 [============================>.] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
24960/25000 [============================>.] - ETA: 0s - loss: 7.6721 - accuracy: 0.4996
24992/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 112s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
