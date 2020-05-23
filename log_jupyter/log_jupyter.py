
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7423a9c1aea8d708841a3941e104542978e088ce', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7423a9c1aea8d708841a3941e104542978e088ce

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py']





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//timeseries_m5_deepar.ipynb 

UsageError: Line magic function `%%capture` not found.





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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:18, 26.25s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3392883519037857, 'embedding_size_factor': 0.9164787937666588, 'layers.choice': 1, 'learning_rate': 0.0027360548384073465, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 5.618569133230322e-06} and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xb6\xe6}\xd5\xca\xbcX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedS\xcbU\xd6\x92\x0cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?fi\xecA\xa1\xf9#X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd7\x90\xe4\x85\xfb\x15\xf7u.' and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xb6\xe6}\xd5\xca\xbcX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedS\xcbU\xd6\x92\x0cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?fi\xecA\xa1\xf9#X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd7\x90\xe4\x85\xfb\x15\xf7u.' and reward: 0.3786
 60%|██████    | 3/5 [02:15<01:26, 43.35s/it] 60%|██████    | 3/5 [02:15<01:30, 45.25s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.29572256967197763, 'embedding_size_factor': 0.6222719768211343, 'layers.choice': 3, 'learning_rate': 0.005656765538786007, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 4.393492463841121e-12} and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xed\x1e[[\x88\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xe9\xa6\xeb\xb5?\xd5X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?w+\x8co\xd7\xf6\xedX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x93R\xa1\xfcG\xefLu.' and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xed\x1e[[\x88\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xe9\xa6\xeb\xb5?\xd5X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?w+\x8co\xd7\xf6\xedX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x93R\xa1\xfcG\xefLu.' and reward: 0.3786
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 192.6267650127411
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -75.31s of remaining time.
Ensemble size: 41
Ensemble weights: 
[0.3902439  0.31707317 0.29268293]
	0.3912	 = Validation accuracy score
	1.14s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 196.49s ...
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 456, in test_cli
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7ff14b779a20> 

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
 [ 0.12865688 -0.02719284  0.01120276  0.07666487 -0.00568259  0.06468985]
 [-0.06801169 -0.02697234  0.03036209  0.03572424  0.14431199 -0.15084252]
 [ 0.11307896 -0.14527486 -0.03070651  0.1087347   0.15764377 -0.07950984]
 [ 0.09333796 -0.01931681 -0.15377401  0.06203671  0.32119668  0.01529914]
 [ 0.28047168 -0.10531732  0.23955213  0.22079243  0.18865415 -0.29224017]
 [-0.15427829 -0.0570286   0.30114296  0.06040179  0.70710433  0.05172309]
 [ 0.433476   -0.24084838  0.0785238   0.0009179   0.45547253 -0.1428349 ]
 [-0.11872073  0.06796487 -0.2925736  -0.1243846   0.23252639  0.07229596]
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
{'loss': 0.4251679666340351, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-23 00:36:44.499528: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 458, in test_cli
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
{'loss': 0.40070103108882904, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-23 00:36:45.697805: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 460, in test_cli
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
 1515520/17464789 [=>............................] - ETA: 0s
 6504448/17464789 [==========>...................] - ETA: 0s
17096704/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 00:36:57.834942: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 00:36:57.838875: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-23 00:36:57.839045: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56121fd4a200 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 00:36:57.839060: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:55 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 2:59 - loss: 7.4270 - accuracy: 0.5156
   96/25000 [..............................] - ETA: 2:20 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 2:02 - loss: 6.4687 - accuracy: 0.5781
  160/25000 [..............................] - ETA: 1:50 - loss: 6.8041 - accuracy: 0.5562
  192/25000 [..............................] - ETA: 1:43 - loss: 7.0277 - accuracy: 0.5417
  224/25000 [..............................] - ETA: 1:37 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:35 - loss: 7.2474 - accuracy: 0.5273
  288/25000 [..............................] - ETA: 1:32 - loss: 7.2939 - accuracy: 0.5243
  320/25000 [..............................] - ETA: 1:29 - loss: 7.2354 - accuracy: 0.5281
  352/25000 [..............................] - ETA: 1:28 - loss: 7.3181 - accuracy: 0.5227
  384/25000 [..............................] - ETA: 1:26 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:24 - loss: 7.4086 - accuracy: 0.5168
  448/25000 [..............................] - ETA: 1:23 - loss: 7.3928 - accuracy: 0.5179
  480/25000 [..............................] - ETA: 1:21 - loss: 7.6027 - accuracy: 0.5042
  512/25000 [..............................] - ETA: 1:20 - loss: 7.5169 - accuracy: 0.5098
  544/25000 [..............................] - ETA: 1:19 - loss: 7.3848 - accuracy: 0.5184
  576/25000 [..............................] - ETA: 1:19 - loss: 7.4270 - accuracy: 0.5156
  608/25000 [..............................] - ETA: 1:18 - loss: 7.4396 - accuracy: 0.5148
  640/25000 [..............................] - ETA: 1:17 - loss: 7.4750 - accuracy: 0.5125
  672/25000 [..............................] - ETA: 1:17 - loss: 7.4384 - accuracy: 0.5149
  704/25000 [..............................] - ETA: 1:16 - loss: 7.4488 - accuracy: 0.5142
  736/25000 [..............................] - ETA: 1:15 - loss: 7.4375 - accuracy: 0.5149
  768/25000 [..............................] - ETA: 1:15 - loss: 7.4270 - accuracy: 0.5156
  800/25000 [..............................] - ETA: 1:14 - loss: 7.4750 - accuracy: 0.5125
  832/25000 [..............................] - ETA: 1:14 - loss: 7.4086 - accuracy: 0.5168
  864/25000 [>.............................] - ETA: 1:14 - loss: 7.3294 - accuracy: 0.5220
  896/25000 [>.............................] - ETA: 1:13 - loss: 7.3244 - accuracy: 0.5223
  928/25000 [>.............................] - ETA: 1:13 - loss: 7.4188 - accuracy: 0.5162
  960/25000 [>.............................] - ETA: 1:13 - loss: 7.4590 - accuracy: 0.5135
  992/25000 [>.............................] - ETA: 1:12 - loss: 7.5275 - accuracy: 0.5091
 1024/25000 [>.............................] - ETA: 1:12 - loss: 7.5019 - accuracy: 0.5107
 1056/25000 [>.............................] - ETA: 1:12 - loss: 7.5505 - accuracy: 0.5076
 1088/25000 [>.............................] - ETA: 1:12 - loss: 7.5680 - accuracy: 0.5064
 1120/25000 [>.............................] - ETA: 1:12 - loss: 7.5571 - accuracy: 0.5071
 1152/25000 [>.............................] - ETA: 1:11 - loss: 7.6001 - accuracy: 0.5043
 1184/25000 [>.............................] - ETA: 1:11 - loss: 7.5630 - accuracy: 0.5068
 1216/25000 [>.............................] - ETA: 1:11 - loss: 7.5279 - accuracy: 0.5090
 1248/25000 [>.............................] - ETA: 1:10 - loss: 7.5315 - accuracy: 0.5088
 1280/25000 [>.............................] - ETA: 1:10 - loss: 7.4869 - accuracy: 0.5117
 1312/25000 [>.............................] - ETA: 1:10 - loss: 7.4563 - accuracy: 0.5137
 1344/25000 [>.............................] - ETA: 1:10 - loss: 7.5183 - accuracy: 0.5097
 1376/25000 [>.............................] - ETA: 1:10 - loss: 7.5552 - accuracy: 0.5073
 1408/25000 [>.............................] - ETA: 1:09 - loss: 7.5577 - accuracy: 0.5071
 1440/25000 [>.............................] - ETA: 1:09 - loss: 7.5921 - accuracy: 0.5049
 1472/25000 [>.............................] - ETA: 1:09 - loss: 7.6250 - accuracy: 0.5027
 1504/25000 [>.............................] - ETA: 1:09 - loss: 7.6360 - accuracy: 0.5020
 1536/25000 [>.............................] - ETA: 1:09 - loss: 7.6067 - accuracy: 0.5039
 1568/25000 [>.............................] - ETA: 1:08 - loss: 7.5688 - accuracy: 0.5064
 1600/25000 [>.............................] - ETA: 1:08 - loss: 7.5133 - accuracy: 0.5100
 1632/25000 [>.............................] - ETA: 1:08 - loss: 7.5351 - accuracy: 0.5086
 1664/25000 [>.............................] - ETA: 1:08 - loss: 7.5100 - accuracy: 0.5102
 1696/25000 [=>............................] - ETA: 1:08 - loss: 7.5491 - accuracy: 0.5077
 1728/25000 [=>............................] - ETA: 1:08 - loss: 7.5158 - accuracy: 0.5098
 1760/25000 [=>............................] - ETA: 1:08 - loss: 7.5447 - accuracy: 0.5080
 1792/25000 [=>............................] - ETA: 1:08 - loss: 7.5040 - accuracy: 0.5106
 1824/25000 [=>............................] - ETA: 1:08 - loss: 7.5069 - accuracy: 0.5104
 1856/25000 [=>............................] - ETA: 1:07 - loss: 7.4931 - accuracy: 0.5113
 1888/25000 [=>............................] - ETA: 1:07 - loss: 7.4879 - accuracy: 0.5117
 1920/25000 [=>............................] - ETA: 1:07 - loss: 7.5069 - accuracy: 0.5104
 1952/25000 [=>............................] - ETA: 1:07 - loss: 7.5331 - accuracy: 0.5087
 1984/25000 [=>............................] - ETA: 1:07 - loss: 7.5507 - accuracy: 0.5076
 2016/25000 [=>............................] - ETA: 1:07 - loss: 7.5373 - accuracy: 0.5084
 2048/25000 [=>............................] - ETA: 1:06 - loss: 7.5319 - accuracy: 0.5088
 2080/25000 [=>............................] - ETA: 1:06 - loss: 7.5339 - accuracy: 0.5087
 2112/25000 [=>............................] - ETA: 1:06 - loss: 7.5142 - accuracy: 0.5099
 2144/25000 [=>............................] - ETA: 1:06 - loss: 7.5164 - accuracy: 0.5098
 2176/25000 [=>............................] - ETA: 1:06 - loss: 7.4834 - accuracy: 0.5119
 2208/25000 [=>............................] - ETA: 1:05 - loss: 7.5069 - accuracy: 0.5104
 2240/25000 [=>............................] - ETA: 1:05 - loss: 7.4613 - accuracy: 0.5134
 2272/25000 [=>............................] - ETA: 1:05 - loss: 7.4979 - accuracy: 0.5110
 2304/25000 [=>............................] - ETA: 1:05 - loss: 7.4936 - accuracy: 0.5113
 2336/25000 [=>............................] - ETA: 1:05 - loss: 7.4697 - accuracy: 0.5128
 2368/25000 [=>............................] - ETA: 1:05 - loss: 7.4918 - accuracy: 0.5114
 2400/25000 [=>............................] - ETA: 1:05 - loss: 7.5133 - accuracy: 0.5100
 2432/25000 [=>............................] - ETA: 1:05 - loss: 7.5468 - accuracy: 0.5078
 2464/25000 [=>............................] - ETA: 1:04 - loss: 7.5919 - accuracy: 0.5049
 2496/25000 [=>............................] - ETA: 1:04 - loss: 7.5929 - accuracy: 0.5048
 2528/25000 [==>...........................] - ETA: 1:04 - loss: 7.5817 - accuracy: 0.5055
 2560/25000 [==>...........................] - ETA: 1:04 - loss: 7.5708 - accuracy: 0.5063
 2592/25000 [==>...........................] - ETA: 1:04 - loss: 7.5661 - accuracy: 0.5066
 2624/25000 [==>...........................] - ETA: 1:04 - loss: 7.5907 - accuracy: 0.5050
 2656/25000 [==>...........................] - ETA: 1:04 - loss: 7.6320 - accuracy: 0.5023
 2688/25000 [==>...........................] - ETA: 1:04 - loss: 7.6438 - accuracy: 0.5015
 2720/25000 [==>...........................] - ETA: 1:04 - loss: 7.6441 - accuracy: 0.5015
 2752/25000 [==>...........................] - ETA: 1:03 - loss: 7.6443 - accuracy: 0.5015
 2784/25000 [==>...........................] - ETA: 1:03 - loss: 7.6721 - accuracy: 0.4996
 2816/25000 [==>...........................] - ETA: 1:03 - loss: 7.6503 - accuracy: 0.5011
 2848/25000 [==>...........................] - ETA: 1:03 - loss: 7.6612 - accuracy: 0.5004
 2880/25000 [==>...........................] - ETA: 1:03 - loss: 7.6613 - accuracy: 0.5003
 2912/25000 [==>...........................] - ETA: 1:03 - loss: 7.6719 - accuracy: 0.4997
 2944/25000 [==>...........................] - ETA: 1:03 - loss: 7.6875 - accuracy: 0.4986
 2976/25000 [==>...........................] - ETA: 1:02 - loss: 7.6718 - accuracy: 0.4997
 3008/25000 [==>...........................] - ETA: 1:02 - loss: 7.6666 - accuracy: 0.5000
 3040/25000 [==>...........................] - ETA: 1:02 - loss: 7.6464 - accuracy: 0.5013
 3072/25000 [==>...........................] - ETA: 1:02 - loss: 7.6666 - accuracy: 0.5000
 3104/25000 [==>...........................] - ETA: 1:02 - loss: 7.6814 - accuracy: 0.4990
 3136/25000 [==>...........................] - ETA: 1:02 - loss: 7.6862 - accuracy: 0.4987
 3168/25000 [==>...........................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 3200/25000 [==>...........................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 3232/25000 [==>...........................] - ETA: 1:01 - loss: 7.6714 - accuracy: 0.4997
 3264/25000 [==>...........................] - ETA: 1:01 - loss: 7.6713 - accuracy: 0.4997
 3296/25000 [==>...........................] - ETA: 1:01 - loss: 7.6620 - accuracy: 0.5003
 3328/25000 [==>...........................] - ETA: 1:01 - loss: 7.6344 - accuracy: 0.5021
 3360/25000 [===>..........................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 3392/25000 [===>..........................] - ETA: 1:01 - loss: 7.6621 - accuracy: 0.5003
 3424/25000 [===>..........................] - ETA: 1:01 - loss: 7.6711 - accuracy: 0.4997
 3456/25000 [===>..........................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 3488/25000 [===>..........................] - ETA: 1:00 - loss: 7.6578 - accuracy: 0.5006
 3520/25000 [===>..........................] - ETA: 1:00 - loss: 7.6797 - accuracy: 0.4991
 3552/25000 [===>..........................] - ETA: 1:00 - loss: 7.6839 - accuracy: 0.4989
 3584/25000 [===>..........................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 3616/25000 [===>..........................] - ETA: 1:00 - loss: 7.6581 - accuracy: 0.5006
 3648/25000 [===>..........................] - ETA: 1:00 - loss: 7.6540 - accuracy: 0.5008
 3680/25000 [===>..........................] - ETA: 1:00 - loss: 7.6708 - accuracy: 0.4997
 3712/25000 [===>..........................] - ETA: 1:00 - loss: 7.6708 - accuracy: 0.4997
 3744/25000 [===>..........................] - ETA: 59s - loss: 7.6461 - accuracy: 0.5013 
 3776/25000 [===>..........................] - ETA: 59s - loss: 7.6341 - accuracy: 0.5021
 3808/25000 [===>..........................] - ETA: 59s - loss: 7.6344 - accuracy: 0.5021
 3840/25000 [===>..........................] - ETA: 59s - loss: 7.6267 - accuracy: 0.5026
 3872/25000 [===>..........................] - ETA: 59s - loss: 7.6151 - accuracy: 0.5034
 3904/25000 [===>..........................] - ETA: 59s - loss: 7.5959 - accuracy: 0.5046
 3936/25000 [===>..........................] - ETA: 59s - loss: 7.5965 - accuracy: 0.5046
 3968/25000 [===>..........................] - ETA: 59s - loss: 7.6125 - accuracy: 0.5035
 4000/25000 [===>..........................] - ETA: 59s - loss: 7.6130 - accuracy: 0.5035
 4032/25000 [===>..........................] - ETA: 58s - loss: 7.6210 - accuracy: 0.5030
 4064/25000 [===>..........................] - ETA: 58s - loss: 7.6213 - accuracy: 0.5030
 4096/25000 [===>..........................] - ETA: 58s - loss: 7.6217 - accuracy: 0.5029
 4128/25000 [===>..........................] - ETA: 58s - loss: 7.6220 - accuracy: 0.5029
 4160/25000 [===>..........................] - ETA: 58s - loss: 7.6334 - accuracy: 0.5022
 4192/25000 [====>.........................] - ETA: 58s - loss: 7.6374 - accuracy: 0.5019
 4224/25000 [====>.........................] - ETA: 58s - loss: 7.6376 - accuracy: 0.5019
 4256/25000 [====>.........................] - ETA: 58s - loss: 7.6306 - accuracy: 0.5023
 4288/25000 [====>.........................] - ETA: 58s - loss: 7.6344 - accuracy: 0.5021
 4320/25000 [====>.........................] - ETA: 57s - loss: 7.6347 - accuracy: 0.5021
 4352/25000 [====>.........................] - ETA: 57s - loss: 7.6490 - accuracy: 0.5011
 4384/25000 [====>.........................] - ETA: 57s - loss: 7.6526 - accuracy: 0.5009
 4416/25000 [====>.........................] - ETA: 57s - loss: 7.6701 - accuracy: 0.4998
 4448/25000 [====>.........................] - ETA: 57s - loss: 7.6632 - accuracy: 0.5002
 4480/25000 [====>.........................] - ETA: 57s - loss: 7.6632 - accuracy: 0.5002
 4512/25000 [====>.........................] - ETA: 57s - loss: 7.6700 - accuracy: 0.4998
 4544/25000 [====>.........................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 4576/25000 [====>.........................] - ETA: 56s - loss: 7.6599 - accuracy: 0.5004
 4608/25000 [====>.........................] - ETA: 56s - loss: 7.6633 - accuracy: 0.5002
 4640/25000 [====>.........................] - ETA: 56s - loss: 7.6633 - accuracy: 0.5002
 4672/25000 [====>.........................] - ETA: 56s - loss: 7.6699 - accuracy: 0.4998
 4704/25000 [====>.........................] - ETA: 56s - loss: 7.6764 - accuracy: 0.4994
 4736/25000 [====>.........................] - ETA: 56s - loss: 7.6731 - accuracy: 0.4996
 4768/25000 [====>.........................] - ETA: 56s - loss: 7.6698 - accuracy: 0.4998
 4800/25000 [====>.........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 4832/25000 [====>.........................] - ETA: 56s - loss: 7.6793 - accuracy: 0.4992
 4864/25000 [====>.........................] - ETA: 56s - loss: 7.6729 - accuracy: 0.4996
 4896/25000 [====>.........................] - ETA: 55s - loss: 7.6635 - accuracy: 0.5002
 4928/25000 [====>.........................] - ETA: 55s - loss: 7.6542 - accuracy: 0.5008
 4960/25000 [====>.........................] - ETA: 55s - loss: 7.6697 - accuracy: 0.4998
 4992/25000 [====>.........................] - ETA: 55s - loss: 7.6635 - accuracy: 0.5002
 5024/25000 [=====>........................] - ETA: 55s - loss: 7.6422 - accuracy: 0.5016
 5056/25000 [=====>........................] - ETA: 55s - loss: 7.6393 - accuracy: 0.5018
 5088/25000 [=====>........................] - ETA: 55s - loss: 7.6425 - accuracy: 0.5016
 5120/25000 [=====>........................] - ETA: 55s - loss: 7.6516 - accuracy: 0.5010
 5152/25000 [=====>........................] - ETA: 55s - loss: 7.6398 - accuracy: 0.5017
 5184/25000 [=====>........................] - ETA: 55s - loss: 7.6370 - accuracy: 0.5019
 5216/25000 [=====>........................] - ETA: 55s - loss: 7.6225 - accuracy: 0.5029
 5248/25000 [=====>........................] - ETA: 54s - loss: 7.6345 - accuracy: 0.5021
 5280/25000 [=====>........................] - ETA: 54s - loss: 7.6405 - accuracy: 0.5017
 5312/25000 [=====>........................] - ETA: 54s - loss: 7.6378 - accuracy: 0.5019
 5344/25000 [=====>........................] - ETA: 54s - loss: 7.6379 - accuracy: 0.5019
 5376/25000 [=====>........................] - ETA: 54s - loss: 7.6324 - accuracy: 0.5022
 5408/25000 [=====>........................] - ETA: 54s - loss: 7.6354 - accuracy: 0.5020
 5440/25000 [=====>........................] - ETA: 54s - loss: 7.6384 - accuracy: 0.5018
 5472/25000 [=====>........................] - ETA: 54s - loss: 7.6330 - accuracy: 0.5022
 5504/25000 [=====>........................] - ETA: 54s - loss: 7.6360 - accuracy: 0.5020
 5536/25000 [=====>........................] - ETA: 54s - loss: 7.6362 - accuracy: 0.5020
 5568/25000 [=====>........................] - ETA: 53s - loss: 7.6281 - accuracy: 0.5025
 5600/25000 [=====>........................] - ETA: 53s - loss: 7.6283 - accuracy: 0.5025
 5632/25000 [=====>........................] - ETA: 53s - loss: 7.6312 - accuracy: 0.5023
 5664/25000 [=====>........................] - ETA: 53s - loss: 7.6423 - accuracy: 0.5016
 5696/25000 [=====>........................] - ETA: 53s - loss: 7.6424 - accuracy: 0.5016
 5728/25000 [=====>........................] - ETA: 53s - loss: 7.6586 - accuracy: 0.5005
 5760/25000 [=====>........................] - ETA: 53s - loss: 7.6586 - accuracy: 0.5005
 5792/25000 [=====>........................] - ETA: 53s - loss: 7.6587 - accuracy: 0.5005
 5824/25000 [=====>........................] - ETA: 53s - loss: 7.6640 - accuracy: 0.5002
 5856/25000 [======>.......................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 5888/25000 [======>.......................] - ETA: 52s - loss: 7.6588 - accuracy: 0.5005
 5920/25000 [======>.......................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
 5952/25000 [======>.......................] - ETA: 52s - loss: 7.6615 - accuracy: 0.5003
 5984/25000 [======>.......................] - ETA: 52s - loss: 7.6692 - accuracy: 0.4998
 6016/25000 [======>.......................] - ETA: 52s - loss: 7.6641 - accuracy: 0.5002
 6048/25000 [======>.......................] - ETA: 52s - loss: 7.6641 - accuracy: 0.5002
 6080/25000 [======>.......................] - ETA: 52s - loss: 7.6691 - accuracy: 0.4998
 6112/25000 [======>.......................] - ETA: 52s - loss: 7.6842 - accuracy: 0.4989
 6144/25000 [======>.......................] - ETA: 52s - loss: 7.6891 - accuracy: 0.4985
 6176/25000 [======>.......................] - ETA: 52s - loss: 7.6914 - accuracy: 0.4984
 6208/25000 [======>.......................] - ETA: 52s - loss: 7.6814 - accuracy: 0.4990
 6240/25000 [======>.......................] - ETA: 51s - loss: 7.6740 - accuracy: 0.4995
 6272/25000 [======>.......................] - ETA: 51s - loss: 7.6740 - accuracy: 0.4995
 6304/25000 [======>.......................] - ETA: 51s - loss: 7.6739 - accuracy: 0.4995
 6336/25000 [======>.......................] - ETA: 51s - loss: 7.6811 - accuracy: 0.4991
 6368/25000 [======>.......................] - ETA: 51s - loss: 7.6738 - accuracy: 0.4995
 6400/25000 [======>.......................] - ETA: 51s - loss: 7.6762 - accuracy: 0.4994
 6432/25000 [======>.......................] - ETA: 51s - loss: 7.6714 - accuracy: 0.4997
 6464/25000 [======>.......................] - ETA: 51s - loss: 7.6832 - accuracy: 0.4989
 6496/25000 [======>.......................] - ETA: 51s - loss: 7.6713 - accuracy: 0.4997
 6528/25000 [======>.......................] - ETA: 51s - loss: 7.6643 - accuracy: 0.5002
 6560/25000 [======>.......................] - ETA: 51s - loss: 7.6619 - accuracy: 0.5003
 6592/25000 [======>.......................] - ETA: 50s - loss: 7.6620 - accuracy: 0.5003
 6624/25000 [======>.......................] - ETA: 50s - loss: 7.6597 - accuracy: 0.5005
 6656/25000 [======>.......................] - ETA: 50s - loss: 7.6620 - accuracy: 0.5003
 6688/25000 [=======>......................] - ETA: 50s - loss: 7.6620 - accuracy: 0.5003
 6720/25000 [=======>......................] - ETA: 50s - loss: 7.6506 - accuracy: 0.5010
 6752/25000 [=======>......................] - ETA: 50s - loss: 7.6553 - accuracy: 0.5007
 6784/25000 [=======>......................] - ETA: 50s - loss: 7.6531 - accuracy: 0.5009
 6816/25000 [=======>......................] - ETA: 50s - loss: 7.6576 - accuracy: 0.5006
 6848/25000 [=======>......................] - ETA: 50s - loss: 7.6577 - accuracy: 0.5006
 6880/25000 [=======>......................] - ETA: 50s - loss: 7.6555 - accuracy: 0.5007
 6912/25000 [=======>......................] - ETA: 50s - loss: 7.6511 - accuracy: 0.5010
 6944/25000 [=======>......................] - ETA: 49s - loss: 7.6512 - accuracy: 0.5010
 6976/25000 [=======>......................] - ETA: 49s - loss: 7.6556 - accuracy: 0.5007
 7008/25000 [=======>......................] - ETA: 49s - loss: 7.6557 - accuracy: 0.5007
 7040/25000 [=======>......................] - ETA: 49s - loss: 7.6579 - accuracy: 0.5006
 7072/25000 [=======>......................] - ETA: 49s - loss: 7.6579 - accuracy: 0.5006
 7104/25000 [=======>......................] - ETA: 49s - loss: 7.6645 - accuracy: 0.5001
 7136/25000 [=======>......................] - ETA: 49s - loss: 7.6580 - accuracy: 0.5006
 7168/25000 [=======>......................] - ETA: 49s - loss: 7.6602 - accuracy: 0.5004
 7200/25000 [=======>......................] - ETA: 49s - loss: 7.6709 - accuracy: 0.4997
 7232/25000 [=======>......................] - ETA: 49s - loss: 7.6709 - accuracy: 0.4997
 7264/25000 [=======>......................] - ETA: 49s - loss: 7.6687 - accuracy: 0.4999
 7296/25000 [=======>......................] - ETA: 48s - loss: 7.6708 - accuracy: 0.4997
 7328/25000 [=======>......................] - ETA: 48s - loss: 7.6729 - accuracy: 0.4996
 7360/25000 [=======>......................] - ETA: 48s - loss: 7.6750 - accuracy: 0.4995
 7392/25000 [=======>......................] - ETA: 48s - loss: 7.6811 - accuracy: 0.4991
 7424/25000 [=======>......................] - ETA: 48s - loss: 7.6831 - accuracy: 0.4989
 7456/25000 [=======>......................] - ETA: 48s - loss: 7.6790 - accuracy: 0.4992
 7488/25000 [=======>......................] - ETA: 48s - loss: 7.6748 - accuracy: 0.4995
 7520/25000 [========>.....................] - ETA: 48s - loss: 7.6768 - accuracy: 0.4993
 7552/25000 [========>.....................] - ETA: 48s - loss: 7.6829 - accuracy: 0.4989
 7584/25000 [========>.....................] - ETA: 48s - loss: 7.6828 - accuracy: 0.4989
 7616/25000 [========>.....................] - ETA: 48s - loss: 7.6827 - accuracy: 0.4989
 7648/25000 [========>.....................] - ETA: 48s - loss: 7.6827 - accuracy: 0.4990
 7680/25000 [========>.....................] - ETA: 47s - loss: 7.6826 - accuracy: 0.4990
 7712/25000 [========>.....................] - ETA: 47s - loss: 7.6785 - accuracy: 0.4992
 7744/25000 [========>.....................] - ETA: 47s - loss: 7.6686 - accuracy: 0.4999
 7776/25000 [========>.....................] - ETA: 47s - loss: 7.6607 - accuracy: 0.5004
 7808/25000 [========>.....................] - ETA: 47s - loss: 7.6627 - accuracy: 0.5003
 7840/25000 [========>.....................] - ETA: 47s - loss: 7.6686 - accuracy: 0.4999
 7872/25000 [========>.....................] - ETA: 47s - loss: 7.6744 - accuracy: 0.4995
 7904/25000 [========>.....................] - ETA: 47s - loss: 7.6763 - accuracy: 0.4994
 7936/25000 [========>.....................] - ETA: 47s - loss: 7.6782 - accuracy: 0.4992
 7968/25000 [========>.....................] - ETA: 47s - loss: 7.6878 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 46s - loss: 7.6935 - accuracy: 0.4983
 8032/25000 [========>.....................] - ETA: 46s - loss: 7.6972 - accuracy: 0.4980
 8064/25000 [========>.....................] - ETA: 46s - loss: 7.7046 - accuracy: 0.4975
 8096/25000 [========>.....................] - ETA: 46s - loss: 7.7026 - accuracy: 0.4977
 8128/25000 [========>.....................] - ETA: 46s - loss: 7.7081 - accuracy: 0.4973
 8160/25000 [========>.....................] - ETA: 46s - loss: 7.7174 - accuracy: 0.4967
 8192/25000 [========>.....................] - ETA: 46s - loss: 7.7265 - accuracy: 0.4961
 8224/25000 [========>.....................] - ETA: 46s - loss: 7.7300 - accuracy: 0.4959
 8256/25000 [========>.....................] - ETA: 46s - loss: 7.7279 - accuracy: 0.4960
 8288/25000 [========>.....................] - ETA: 46s - loss: 7.7277 - accuracy: 0.4960
 8320/25000 [========>.....................] - ETA: 45s - loss: 7.7311 - accuracy: 0.4958
 8352/25000 [=========>....................] - ETA: 45s - loss: 7.7309 - accuracy: 0.4958
 8384/25000 [=========>....................] - ETA: 45s - loss: 7.7306 - accuracy: 0.4958
 8416/25000 [=========>....................] - ETA: 45s - loss: 7.7286 - accuracy: 0.4960
 8448/25000 [=========>....................] - ETA: 45s - loss: 7.7265 - accuracy: 0.4961
 8480/25000 [=========>....................] - ETA: 45s - loss: 7.7209 - accuracy: 0.4965
 8512/25000 [=========>....................] - ETA: 45s - loss: 7.7225 - accuracy: 0.4964
 8544/25000 [=========>....................] - ETA: 45s - loss: 7.7205 - accuracy: 0.4965
 8576/25000 [=========>....................] - ETA: 45s - loss: 7.7238 - accuracy: 0.4963
 8608/25000 [=========>....................] - ETA: 45s - loss: 7.7236 - accuracy: 0.4963
 8640/25000 [=========>....................] - ETA: 44s - loss: 7.7234 - accuracy: 0.4963
 8672/25000 [=========>....................] - ETA: 44s - loss: 7.7267 - accuracy: 0.4961
 8704/25000 [=========>....................] - ETA: 44s - loss: 7.7142 - accuracy: 0.4969
 8736/25000 [=========>....................] - ETA: 44s - loss: 7.7228 - accuracy: 0.4963
 8768/25000 [=========>....................] - ETA: 44s - loss: 7.7278 - accuracy: 0.4960
 8800/25000 [=========>....................] - ETA: 44s - loss: 7.7293 - accuracy: 0.4959
 8832/25000 [=========>....................] - ETA: 44s - loss: 7.7309 - accuracy: 0.4958
 8864/25000 [=========>....................] - ETA: 44s - loss: 7.7358 - accuracy: 0.4955
 8896/25000 [=========>....................] - ETA: 44s - loss: 7.7373 - accuracy: 0.4954
 8928/25000 [=========>....................] - ETA: 44s - loss: 7.7405 - accuracy: 0.4952
 8960/25000 [=========>....................] - ETA: 44s - loss: 7.7351 - accuracy: 0.4955
 8992/25000 [=========>....................] - ETA: 43s - loss: 7.7348 - accuracy: 0.4956
 9024/25000 [=========>....................] - ETA: 43s - loss: 7.7312 - accuracy: 0.4958
 9056/25000 [=========>....................] - ETA: 43s - loss: 7.7276 - accuracy: 0.4960
 9088/25000 [=========>....................] - ETA: 43s - loss: 7.7240 - accuracy: 0.4963
 9120/25000 [=========>....................] - ETA: 43s - loss: 7.7171 - accuracy: 0.4967
 9152/25000 [=========>....................] - ETA: 43s - loss: 7.7253 - accuracy: 0.4962
 9184/25000 [==========>...................] - ETA: 43s - loss: 7.7284 - accuracy: 0.4960
 9216/25000 [==========>...................] - ETA: 43s - loss: 7.7265 - accuracy: 0.4961
 9248/25000 [==========>...................] - ETA: 43s - loss: 7.7263 - accuracy: 0.4961
 9280/25000 [==========>...................] - ETA: 43s - loss: 7.7244 - accuracy: 0.4962
 9312/25000 [==========>...................] - ETA: 43s - loss: 7.7308 - accuracy: 0.4958
 9344/25000 [==========>...................] - ETA: 42s - loss: 7.7224 - accuracy: 0.4964
 9376/25000 [==========>...................] - ETA: 42s - loss: 7.7222 - accuracy: 0.4964
 9408/25000 [==========>...................] - ETA: 42s - loss: 7.7204 - accuracy: 0.4965
 9440/25000 [==========>...................] - ETA: 42s - loss: 7.7186 - accuracy: 0.4966
 9472/25000 [==========>...................] - ETA: 42s - loss: 7.7152 - accuracy: 0.4968
 9504/25000 [==========>...................] - ETA: 42s - loss: 7.7134 - accuracy: 0.4969
 9536/25000 [==========>...................] - ETA: 42s - loss: 7.7116 - accuracy: 0.4971
 9568/25000 [==========>...................] - ETA: 42s - loss: 7.7083 - accuracy: 0.4973
 9600/25000 [==========>...................] - ETA: 42s - loss: 7.6986 - accuracy: 0.4979
 9632/25000 [==========>...................] - ETA: 42s - loss: 7.6985 - accuracy: 0.4979
 9664/25000 [==========>...................] - ETA: 42s - loss: 7.6952 - accuracy: 0.4981
 9696/25000 [==========>...................] - ETA: 42s - loss: 7.6903 - accuracy: 0.4985
 9728/25000 [==========>...................] - ETA: 41s - loss: 7.6887 - accuracy: 0.4986
 9760/25000 [==========>...................] - ETA: 41s - loss: 7.6965 - accuracy: 0.4981
 9792/25000 [==========>...................] - ETA: 41s - loss: 7.6932 - accuracy: 0.4983
 9824/25000 [==========>...................] - ETA: 41s - loss: 7.6932 - accuracy: 0.4983
 9856/25000 [==========>...................] - ETA: 41s - loss: 7.7008 - accuracy: 0.4978
 9888/25000 [==========>...................] - ETA: 41s - loss: 7.6992 - accuracy: 0.4979
 9920/25000 [==========>...................] - ETA: 41s - loss: 7.7037 - accuracy: 0.4976
 9952/25000 [==========>...................] - ETA: 41s - loss: 7.7021 - accuracy: 0.4977
 9984/25000 [==========>...................] - ETA: 41s - loss: 7.7065 - accuracy: 0.4974
10016/25000 [===========>..................] - ETA: 41s - loss: 7.7034 - accuracy: 0.4976
10048/25000 [===========>..................] - ETA: 41s - loss: 7.7124 - accuracy: 0.4970
10080/25000 [===========>..................] - ETA: 40s - loss: 7.7123 - accuracy: 0.4970
10112/25000 [===========>..................] - ETA: 40s - loss: 7.7106 - accuracy: 0.4971
10144/25000 [===========>..................] - ETA: 40s - loss: 7.7135 - accuracy: 0.4969
10176/25000 [===========>..................] - ETA: 40s - loss: 7.7163 - accuracy: 0.4968
10208/25000 [===========>..................] - ETA: 40s - loss: 7.7177 - accuracy: 0.4967
10240/25000 [===========>..................] - ETA: 40s - loss: 7.7235 - accuracy: 0.4963
10272/25000 [===========>..................] - ETA: 40s - loss: 7.7308 - accuracy: 0.4958
10304/25000 [===========>..................] - ETA: 40s - loss: 7.7276 - accuracy: 0.4960
10336/25000 [===========>..................] - ETA: 40s - loss: 7.7260 - accuracy: 0.4961
10368/25000 [===========>..................] - ETA: 40s - loss: 7.7287 - accuracy: 0.4959
10400/25000 [===========>..................] - ETA: 40s - loss: 7.7344 - accuracy: 0.4956
10432/25000 [===========>..................] - ETA: 39s - loss: 7.7357 - accuracy: 0.4955
10464/25000 [===========>..................] - ETA: 39s - loss: 7.7340 - accuracy: 0.4956
10496/25000 [===========>..................] - ETA: 39s - loss: 7.7294 - accuracy: 0.4959
10528/25000 [===========>..................] - ETA: 39s - loss: 7.7249 - accuracy: 0.4962
10560/25000 [===========>..................] - ETA: 39s - loss: 7.7305 - accuracy: 0.4958
10592/25000 [===========>..................] - ETA: 39s - loss: 7.7260 - accuracy: 0.4961
10624/25000 [===========>..................] - ETA: 39s - loss: 7.7244 - accuracy: 0.4962
10656/25000 [===========>..................] - ETA: 39s - loss: 7.7213 - accuracy: 0.4964
10688/25000 [===========>..................] - ETA: 39s - loss: 7.7226 - accuracy: 0.4964
10720/25000 [===========>..................] - ETA: 39s - loss: 7.7238 - accuracy: 0.4963
10752/25000 [===========>..................] - ETA: 38s - loss: 7.7208 - accuracy: 0.4965
10784/25000 [===========>..................] - ETA: 38s - loss: 7.7221 - accuracy: 0.4964
10816/25000 [===========>..................] - ETA: 38s - loss: 7.7191 - accuracy: 0.4966
10848/25000 [============>.................] - ETA: 38s - loss: 7.7217 - accuracy: 0.4964
10880/25000 [============>.................] - ETA: 38s - loss: 7.7188 - accuracy: 0.4966
10912/25000 [============>.................] - ETA: 38s - loss: 7.7200 - accuracy: 0.4965
10944/25000 [============>.................] - ETA: 38s - loss: 7.7227 - accuracy: 0.4963
10976/25000 [============>.................] - ETA: 38s - loss: 7.7141 - accuracy: 0.4969
11008/25000 [============>.................] - ETA: 38s - loss: 7.7168 - accuracy: 0.4967
11040/25000 [============>.................] - ETA: 38s - loss: 7.7194 - accuracy: 0.4966
11072/25000 [============>.................] - ETA: 38s - loss: 7.7206 - accuracy: 0.4965
11104/25000 [============>.................] - ETA: 37s - loss: 7.7163 - accuracy: 0.4968
11136/25000 [============>.................] - ETA: 37s - loss: 7.7121 - accuracy: 0.4970
11168/25000 [============>.................] - ETA: 37s - loss: 7.7119 - accuracy: 0.4970
11200/25000 [============>.................] - ETA: 37s - loss: 7.7063 - accuracy: 0.4974
11232/25000 [============>.................] - ETA: 37s - loss: 7.7144 - accuracy: 0.4969
11264/25000 [============>.................] - ETA: 37s - loss: 7.7143 - accuracy: 0.4969
11296/25000 [============>.................] - ETA: 37s - loss: 7.7101 - accuracy: 0.4972
11328/25000 [============>.................] - ETA: 37s - loss: 7.7059 - accuracy: 0.4974
11360/25000 [============>.................] - ETA: 37s - loss: 7.7071 - accuracy: 0.4974
11392/25000 [============>.................] - ETA: 37s - loss: 7.7097 - accuracy: 0.4972
11424/25000 [============>.................] - ETA: 37s - loss: 7.7082 - accuracy: 0.4973
11456/25000 [============>.................] - ETA: 36s - loss: 7.7094 - accuracy: 0.4972
11488/25000 [============>.................] - ETA: 36s - loss: 7.7040 - accuracy: 0.4976
11520/25000 [============>.................] - ETA: 36s - loss: 7.6999 - accuracy: 0.4978
11552/25000 [============>.................] - ETA: 36s - loss: 7.6998 - accuracy: 0.4978
11584/25000 [============>.................] - ETA: 36s - loss: 7.7010 - accuracy: 0.4978
11616/25000 [============>.................] - ETA: 36s - loss: 7.6983 - accuracy: 0.4979
11648/25000 [============>.................] - ETA: 36s - loss: 7.6956 - accuracy: 0.4981
11680/25000 [=============>................] - ETA: 36s - loss: 7.6968 - accuracy: 0.4980
11712/25000 [=============>................] - ETA: 36s - loss: 7.7020 - accuracy: 0.4977
11744/25000 [=============>................] - ETA: 36s - loss: 7.7019 - accuracy: 0.4977
11776/25000 [=============>................] - ETA: 36s - loss: 7.7031 - accuracy: 0.4976
11808/25000 [=============>................] - ETA: 35s - loss: 7.6991 - accuracy: 0.4979
11840/25000 [=============>................] - ETA: 35s - loss: 7.7029 - accuracy: 0.4976
11872/25000 [=============>................] - ETA: 35s - loss: 7.6976 - accuracy: 0.4980
11904/25000 [=============>................] - ETA: 35s - loss: 7.6950 - accuracy: 0.4982
11936/25000 [=============>................] - ETA: 35s - loss: 7.6949 - accuracy: 0.4982
11968/25000 [=============>................] - ETA: 35s - loss: 7.6922 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 35s - loss: 7.6871 - accuracy: 0.4987
12032/25000 [=============>................] - ETA: 35s - loss: 7.6896 - accuracy: 0.4985
12064/25000 [=============>................] - ETA: 35s - loss: 7.6857 - accuracy: 0.4988
12096/25000 [=============>................] - ETA: 35s - loss: 7.6818 - accuracy: 0.4990
12128/25000 [=============>................] - ETA: 34s - loss: 7.6856 - accuracy: 0.4988
12160/25000 [=============>................] - ETA: 34s - loss: 7.6893 - accuracy: 0.4985
12192/25000 [=============>................] - ETA: 34s - loss: 7.6918 - accuracy: 0.4984
12224/25000 [=============>................] - ETA: 34s - loss: 7.6854 - accuracy: 0.4988
12256/25000 [=============>................] - ETA: 34s - loss: 7.6841 - accuracy: 0.4989
12288/25000 [=============>................] - ETA: 34s - loss: 7.6816 - accuracy: 0.4990
12320/25000 [=============>................] - ETA: 34s - loss: 7.6828 - accuracy: 0.4989
12352/25000 [=============>................] - ETA: 34s - loss: 7.6840 - accuracy: 0.4989
12384/25000 [=============>................] - ETA: 34s - loss: 7.6778 - accuracy: 0.4993
12416/25000 [=============>................] - ETA: 34s - loss: 7.6814 - accuracy: 0.4990
12448/25000 [=============>................] - ETA: 34s - loss: 7.6752 - accuracy: 0.4994
12480/25000 [=============>................] - ETA: 33s - loss: 7.6863 - accuracy: 0.4987
12512/25000 [==============>...............] - ETA: 33s - loss: 7.6801 - accuracy: 0.4991
12544/25000 [==============>...............] - ETA: 33s - loss: 7.6801 - accuracy: 0.4991
12576/25000 [==============>...............] - ETA: 33s - loss: 7.6764 - accuracy: 0.4994
12608/25000 [==============>...............] - ETA: 33s - loss: 7.6739 - accuracy: 0.4995
12640/25000 [==============>...............] - ETA: 33s - loss: 7.6678 - accuracy: 0.4999
12672/25000 [==============>...............] - ETA: 33s - loss: 7.6702 - accuracy: 0.4998
12704/25000 [==============>...............] - ETA: 33s - loss: 7.6690 - accuracy: 0.4998
12736/25000 [==============>...............] - ETA: 33s - loss: 7.6750 - accuracy: 0.4995
12768/25000 [==============>...............] - ETA: 33s - loss: 7.6750 - accuracy: 0.4995
12800/25000 [==============>...............] - ETA: 33s - loss: 7.6702 - accuracy: 0.4998
12832/25000 [==============>...............] - ETA: 32s - loss: 7.6702 - accuracy: 0.4998
12864/25000 [==============>...............] - ETA: 32s - loss: 7.6690 - accuracy: 0.4998
12896/25000 [==============>...............] - ETA: 32s - loss: 7.6714 - accuracy: 0.4997
12928/25000 [==============>...............] - ETA: 32s - loss: 7.6725 - accuracy: 0.4996
12960/25000 [==============>...............] - ETA: 32s - loss: 7.6737 - accuracy: 0.4995
12992/25000 [==============>...............] - ETA: 32s - loss: 7.6725 - accuracy: 0.4996
13024/25000 [==============>...............] - ETA: 32s - loss: 7.6737 - accuracy: 0.4995
13056/25000 [==============>...............] - ETA: 32s - loss: 7.6784 - accuracy: 0.4992
13088/25000 [==============>...............] - ETA: 32s - loss: 7.6772 - accuracy: 0.4993
13120/25000 [==============>...............] - ETA: 32s - loss: 7.6760 - accuracy: 0.4994
13152/25000 [==============>...............] - ETA: 32s - loss: 7.6771 - accuracy: 0.4993
13184/25000 [==============>...............] - ETA: 31s - loss: 7.6771 - accuracy: 0.4993
13216/25000 [==============>...............] - ETA: 31s - loss: 7.6759 - accuracy: 0.4994
13248/25000 [==============>...............] - ETA: 31s - loss: 7.6712 - accuracy: 0.4997
13280/25000 [==============>...............] - ETA: 31s - loss: 7.6747 - accuracy: 0.4995
13312/25000 [==============>...............] - ETA: 31s - loss: 7.6770 - accuracy: 0.4993
13344/25000 [===============>..............] - ETA: 31s - loss: 7.6735 - accuracy: 0.4996
13376/25000 [===============>..............] - ETA: 31s - loss: 7.6769 - accuracy: 0.4993
13408/25000 [===============>..............] - ETA: 31s - loss: 7.6792 - accuracy: 0.4992
13440/25000 [===============>..............] - ETA: 31s - loss: 7.6803 - accuracy: 0.4991
13472/25000 [===============>..............] - ETA: 31s - loss: 7.6746 - accuracy: 0.4995
13504/25000 [===============>..............] - ETA: 31s - loss: 7.6746 - accuracy: 0.4995
13536/25000 [===============>..............] - ETA: 30s - loss: 7.6768 - accuracy: 0.4993
13568/25000 [===============>..............] - ETA: 30s - loss: 7.6836 - accuracy: 0.4989
13600/25000 [===============>..............] - ETA: 30s - loss: 7.6847 - accuracy: 0.4988
13632/25000 [===============>..............] - ETA: 30s - loss: 7.6891 - accuracy: 0.4985
13664/25000 [===============>..............] - ETA: 30s - loss: 7.6835 - accuracy: 0.4989
13696/25000 [===============>..............] - ETA: 30s - loss: 7.6834 - accuracy: 0.4989
13728/25000 [===============>..............] - ETA: 30s - loss: 7.6823 - accuracy: 0.4990
13760/25000 [===============>..............] - ETA: 30s - loss: 7.6766 - accuracy: 0.4993
13792/25000 [===============>..............] - ETA: 30s - loss: 7.6788 - accuracy: 0.4992
13824/25000 [===============>..............] - ETA: 30s - loss: 7.6777 - accuracy: 0.4993
13856/25000 [===============>..............] - ETA: 30s - loss: 7.6744 - accuracy: 0.4995
13888/25000 [===============>..............] - ETA: 29s - loss: 7.6755 - accuracy: 0.4994
13920/25000 [===============>..............] - ETA: 29s - loss: 7.6732 - accuracy: 0.4996
13952/25000 [===============>..............] - ETA: 29s - loss: 7.6721 - accuracy: 0.4996
13984/25000 [===============>..............] - ETA: 29s - loss: 7.6699 - accuracy: 0.4998
14016/25000 [===============>..............] - ETA: 29s - loss: 7.6710 - accuracy: 0.4997
14048/25000 [===============>..............] - ETA: 29s - loss: 7.6710 - accuracy: 0.4997
14080/25000 [===============>..............] - ETA: 29s - loss: 7.6721 - accuracy: 0.4996
14112/25000 [===============>..............] - ETA: 29s - loss: 7.6731 - accuracy: 0.4996
14144/25000 [===============>..............] - ETA: 29s - loss: 7.6731 - accuracy: 0.4996
14176/25000 [================>.............] - ETA: 29s - loss: 7.6774 - accuracy: 0.4993
14208/25000 [================>.............] - ETA: 29s - loss: 7.6763 - accuracy: 0.4994
14240/25000 [================>.............] - ETA: 28s - loss: 7.6763 - accuracy: 0.4994
14272/25000 [================>.............] - ETA: 28s - loss: 7.6752 - accuracy: 0.4994
14304/25000 [================>.............] - ETA: 28s - loss: 7.6752 - accuracy: 0.4994
14336/25000 [================>.............] - ETA: 28s - loss: 7.6720 - accuracy: 0.4997
14368/25000 [================>.............] - ETA: 28s - loss: 7.6709 - accuracy: 0.4997
14400/25000 [================>.............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
14432/25000 [================>.............] - ETA: 28s - loss: 7.6613 - accuracy: 0.5003
14464/25000 [================>.............] - ETA: 28s - loss: 7.6603 - accuracy: 0.5004
14496/25000 [================>.............] - ETA: 28s - loss: 7.6613 - accuracy: 0.5003
14528/25000 [================>.............] - ETA: 28s - loss: 7.6635 - accuracy: 0.5002
14560/25000 [================>.............] - ETA: 28s - loss: 7.6656 - accuracy: 0.5001
14592/25000 [================>.............] - ETA: 28s - loss: 7.6603 - accuracy: 0.5004
14624/25000 [================>.............] - ETA: 27s - loss: 7.6572 - accuracy: 0.5006
14656/25000 [================>.............] - ETA: 27s - loss: 7.6603 - accuracy: 0.5004
14688/25000 [================>.............] - ETA: 27s - loss: 7.6572 - accuracy: 0.5006
14720/25000 [================>.............] - ETA: 27s - loss: 7.6541 - accuracy: 0.5008
14752/25000 [================>.............] - ETA: 27s - loss: 7.6541 - accuracy: 0.5008
14784/25000 [================>.............] - ETA: 27s - loss: 7.6511 - accuracy: 0.5010
14816/25000 [================>.............] - ETA: 27s - loss: 7.6521 - accuracy: 0.5009
14848/25000 [================>.............] - ETA: 27s - loss: 7.6491 - accuracy: 0.5011
14880/25000 [================>.............] - ETA: 27s - loss: 7.6491 - accuracy: 0.5011
14912/25000 [================>.............] - ETA: 27s - loss: 7.6481 - accuracy: 0.5012
14944/25000 [================>.............] - ETA: 27s - loss: 7.6482 - accuracy: 0.5012
14976/25000 [================>.............] - ETA: 26s - loss: 7.6533 - accuracy: 0.5009
15008/25000 [=================>............] - ETA: 26s - loss: 7.6513 - accuracy: 0.5010
15040/25000 [=================>............] - ETA: 26s - loss: 7.6503 - accuracy: 0.5011
15072/25000 [=================>............] - ETA: 26s - loss: 7.6514 - accuracy: 0.5010
15104/25000 [=================>............] - ETA: 26s - loss: 7.6433 - accuracy: 0.5015
15136/25000 [=================>............] - ETA: 26s - loss: 7.6443 - accuracy: 0.5015
15168/25000 [=================>............] - ETA: 26s - loss: 7.6424 - accuracy: 0.5016
15200/25000 [=================>............] - ETA: 26s - loss: 7.6485 - accuracy: 0.5012
15232/25000 [=================>............] - ETA: 26s - loss: 7.6455 - accuracy: 0.5014
15264/25000 [=================>............] - ETA: 26s - loss: 7.6435 - accuracy: 0.5015
15296/25000 [=================>............] - ETA: 26s - loss: 7.6406 - accuracy: 0.5017
15328/25000 [=================>............] - ETA: 25s - loss: 7.6376 - accuracy: 0.5019
15360/25000 [=================>............] - ETA: 25s - loss: 7.6367 - accuracy: 0.5020
15392/25000 [=================>............] - ETA: 25s - loss: 7.6427 - accuracy: 0.5016
15424/25000 [=================>............] - ETA: 25s - loss: 7.6418 - accuracy: 0.5016
15456/25000 [=================>............] - ETA: 25s - loss: 7.6418 - accuracy: 0.5016
15488/25000 [=================>............] - ETA: 25s - loss: 7.6419 - accuracy: 0.5016
15520/25000 [=================>............] - ETA: 25s - loss: 7.6409 - accuracy: 0.5017
15552/25000 [=================>............] - ETA: 25s - loss: 7.6370 - accuracy: 0.5019
15584/25000 [=================>............] - ETA: 25s - loss: 7.6351 - accuracy: 0.5021
15616/25000 [=================>............] - ETA: 25s - loss: 7.6313 - accuracy: 0.5023
15648/25000 [=================>............] - ETA: 25s - loss: 7.6362 - accuracy: 0.5020
15680/25000 [=================>............] - ETA: 24s - loss: 7.6383 - accuracy: 0.5018
15712/25000 [=================>............] - ETA: 24s - loss: 7.6373 - accuracy: 0.5019
15744/25000 [=================>............] - ETA: 24s - loss: 7.6374 - accuracy: 0.5019
15776/25000 [=================>............] - ETA: 24s - loss: 7.6394 - accuracy: 0.5018
15808/25000 [=================>............] - ETA: 24s - loss: 7.6346 - accuracy: 0.5021
15840/25000 [==================>...........] - ETA: 24s - loss: 7.6356 - accuracy: 0.5020
15872/25000 [==================>...........] - ETA: 24s - loss: 7.6386 - accuracy: 0.5018
15904/25000 [==================>...........] - ETA: 24s - loss: 7.6416 - accuracy: 0.5016
15936/25000 [==================>...........] - ETA: 24s - loss: 7.6445 - accuracy: 0.5014
15968/25000 [==================>...........] - ETA: 24s - loss: 7.6474 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 24s - loss: 7.6494 - accuracy: 0.5011
16032/25000 [==================>...........] - ETA: 24s - loss: 7.6504 - accuracy: 0.5011
16064/25000 [==================>...........] - ETA: 23s - loss: 7.6494 - accuracy: 0.5011
16096/25000 [==================>...........] - ETA: 23s - loss: 7.6504 - accuracy: 0.5011
16128/25000 [==================>...........] - ETA: 23s - loss: 7.6505 - accuracy: 0.5011
16160/25000 [==================>...........] - ETA: 23s - loss: 7.6505 - accuracy: 0.5011
16192/25000 [==================>...........] - ETA: 23s - loss: 7.6505 - accuracy: 0.5010
16224/25000 [==================>...........] - ETA: 23s - loss: 7.6515 - accuracy: 0.5010
16256/25000 [==================>...........] - ETA: 23s - loss: 7.6515 - accuracy: 0.5010
16288/25000 [==================>...........] - ETA: 23s - loss: 7.6525 - accuracy: 0.5009
16320/25000 [==================>...........] - ETA: 23s - loss: 7.6478 - accuracy: 0.5012
16352/25000 [==================>...........] - ETA: 23s - loss: 7.6488 - accuracy: 0.5012
16384/25000 [==================>...........] - ETA: 23s - loss: 7.6479 - accuracy: 0.5012
16416/25000 [==================>...........] - ETA: 22s - loss: 7.6498 - accuracy: 0.5011
16448/25000 [==================>...........] - ETA: 22s - loss: 7.6489 - accuracy: 0.5012
16480/25000 [==================>...........] - ETA: 22s - loss: 7.6508 - accuracy: 0.5010
16512/25000 [==================>...........] - ETA: 22s - loss: 7.6499 - accuracy: 0.5011
16544/25000 [==================>...........] - ETA: 22s - loss: 7.6555 - accuracy: 0.5007
16576/25000 [==================>...........] - ETA: 22s - loss: 7.6518 - accuracy: 0.5010
16608/25000 [==================>...........] - ETA: 22s - loss: 7.6491 - accuracy: 0.5011
16640/25000 [==================>...........] - ETA: 22s - loss: 7.6463 - accuracy: 0.5013
16672/25000 [===================>..........] - ETA: 22s - loss: 7.6427 - accuracy: 0.5016
16704/25000 [===================>..........] - ETA: 22s - loss: 7.6428 - accuracy: 0.5016
16736/25000 [===================>..........] - ETA: 22s - loss: 7.6455 - accuracy: 0.5014
16768/25000 [===================>..........] - ETA: 22s - loss: 7.6456 - accuracy: 0.5014
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6493 - accuracy: 0.5011
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6520 - accuracy: 0.5010
16864/25000 [===================>..........] - ETA: 21s - loss: 7.6493 - accuracy: 0.5011
16896/25000 [===================>..........] - ETA: 21s - loss: 7.6494 - accuracy: 0.5011
16928/25000 [===================>..........] - ETA: 21s - loss: 7.6521 - accuracy: 0.5009
16960/25000 [===================>..........] - ETA: 21s - loss: 7.6531 - accuracy: 0.5009
16992/25000 [===================>..........] - ETA: 21s - loss: 7.6522 - accuracy: 0.5009
17024/25000 [===================>..........] - ETA: 21s - loss: 7.6477 - accuracy: 0.5012
17056/25000 [===================>..........] - ETA: 21s - loss: 7.6459 - accuracy: 0.5013
17088/25000 [===================>..........] - ETA: 21s - loss: 7.6487 - accuracy: 0.5012
17120/25000 [===================>..........] - ETA: 21s - loss: 7.6505 - accuracy: 0.5011
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6514 - accuracy: 0.5010
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6532 - accuracy: 0.5009
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6524 - accuracy: 0.5009
17248/25000 [===================>..........] - ETA: 20s - loss: 7.6542 - accuracy: 0.5008
17280/25000 [===================>..........] - ETA: 20s - loss: 7.6551 - accuracy: 0.5008
17312/25000 [===================>..........] - ETA: 20s - loss: 7.6560 - accuracy: 0.5007
17344/25000 [===================>..........] - ETA: 20s - loss: 7.6507 - accuracy: 0.5010
17376/25000 [===================>..........] - ETA: 20s - loss: 7.6525 - accuracy: 0.5009
17408/25000 [===================>..........] - ETA: 20s - loss: 7.6481 - accuracy: 0.5012
17440/25000 [===================>..........] - ETA: 20s - loss: 7.6482 - accuracy: 0.5012
17472/25000 [===================>..........] - ETA: 20s - loss: 7.6464 - accuracy: 0.5013
17504/25000 [====================>.........] - ETA: 20s - loss: 7.6430 - accuracy: 0.5015
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6395 - accuracy: 0.5018
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6404 - accuracy: 0.5017
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6387 - accuracy: 0.5018
17632/25000 [====================>.........] - ETA: 19s - loss: 7.6388 - accuracy: 0.5018
17664/25000 [====================>.........] - ETA: 19s - loss: 7.6397 - accuracy: 0.5018
17696/25000 [====================>.........] - ETA: 19s - loss: 7.6406 - accuracy: 0.5017
17728/25000 [====================>.........] - ETA: 19s - loss: 7.6363 - accuracy: 0.5020
17760/25000 [====================>.........] - ETA: 19s - loss: 7.6321 - accuracy: 0.5023
17792/25000 [====================>.........] - ETA: 19s - loss: 7.6304 - accuracy: 0.5024
17824/25000 [====================>.........] - ETA: 19s - loss: 7.6313 - accuracy: 0.5023
17856/25000 [====================>.........] - ETA: 19s - loss: 7.6288 - accuracy: 0.5025
17888/25000 [====================>.........] - ETA: 19s - loss: 7.6280 - accuracy: 0.5025
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6238 - accuracy: 0.5028
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6265 - accuracy: 0.5026
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6265 - accuracy: 0.5026
18016/25000 [====================>.........] - ETA: 18s - loss: 7.6249 - accuracy: 0.5027
18048/25000 [====================>.........] - ETA: 18s - loss: 7.6224 - accuracy: 0.5029
18080/25000 [====================>.........] - ETA: 18s - loss: 7.6208 - accuracy: 0.5030
18112/25000 [====================>.........] - ETA: 18s - loss: 7.6218 - accuracy: 0.5029
18144/25000 [====================>.........] - ETA: 18s - loss: 7.6227 - accuracy: 0.5029
18176/25000 [====================>.........] - ETA: 18s - loss: 7.6228 - accuracy: 0.5029
18208/25000 [====================>.........] - ETA: 18s - loss: 7.6254 - accuracy: 0.5027
18240/25000 [====================>.........] - ETA: 18s - loss: 7.6229 - accuracy: 0.5029
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6196 - accuracy: 0.5031
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6205 - accuracy: 0.5030
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6164 - accuracy: 0.5033
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6190 - accuracy: 0.5031
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6166 - accuracy: 0.5033
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6167 - accuracy: 0.5033
18464/25000 [=====================>........] - ETA: 17s - loss: 7.6168 - accuracy: 0.5032
18496/25000 [=====================>........] - ETA: 17s - loss: 7.6152 - accuracy: 0.5034
18528/25000 [=====================>........] - ETA: 17s - loss: 7.6120 - accuracy: 0.5036
18560/25000 [=====================>........] - ETA: 17s - loss: 7.6113 - accuracy: 0.5036
18592/25000 [=====================>........] - ETA: 17s - loss: 7.6114 - accuracy: 0.5036
18624/25000 [=====================>........] - ETA: 17s - loss: 7.6098 - accuracy: 0.5037
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6107 - accuracy: 0.5036
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6116 - accuracy: 0.5036
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6085 - accuracy: 0.5038
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6110 - accuracy: 0.5036
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6103 - accuracy: 0.5037
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6120 - accuracy: 0.5036
18848/25000 [=====================>........] - ETA: 16s - loss: 7.6072 - accuracy: 0.5039
18880/25000 [=====================>........] - ETA: 16s - loss: 7.6081 - accuracy: 0.5038
18912/25000 [=====================>........] - ETA: 16s - loss: 7.6099 - accuracy: 0.5037
18944/25000 [=====================>........] - ETA: 16s - loss: 7.6108 - accuracy: 0.5036
18976/25000 [=====================>........] - ETA: 16s - loss: 7.6068 - accuracy: 0.5039
19008/25000 [=====================>........] - ETA: 16s - loss: 7.6077 - accuracy: 0.5038
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6111 - accuracy: 0.5036
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6119 - accuracy: 0.5036
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6136 - accuracy: 0.5035
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6113 - accuracy: 0.5036
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6106 - accuracy: 0.5037
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6107 - accuracy: 0.5036
19232/25000 [======================>.......] - ETA: 15s - loss: 7.6076 - accuracy: 0.5038
19264/25000 [======================>.......] - ETA: 15s - loss: 7.6093 - accuracy: 0.5037
19296/25000 [======================>.......] - ETA: 15s - loss: 7.6094 - accuracy: 0.5037
19328/25000 [======================>.......] - ETA: 15s - loss: 7.6079 - accuracy: 0.5038
19360/25000 [======================>.......] - ETA: 15s - loss: 7.6096 - accuracy: 0.5037
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6105 - accuracy: 0.5037
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6122 - accuracy: 0.5036
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6091 - accuracy: 0.5038
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6060 - accuracy: 0.5040
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6069 - accuracy: 0.5039
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6102 - accuracy: 0.5037
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6110 - accuracy: 0.5036
19616/25000 [======================>.......] - ETA: 14s - loss: 7.6080 - accuracy: 0.5038
19648/25000 [======================>.......] - ETA: 14s - loss: 7.6065 - accuracy: 0.5039
19680/25000 [======================>.......] - ETA: 14s - loss: 7.6074 - accuracy: 0.5039
19712/25000 [======================>.......] - ETA: 14s - loss: 7.6075 - accuracy: 0.5039
19744/25000 [======================>.......] - ETA: 14s - loss: 7.6123 - accuracy: 0.5035
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6154 - accuracy: 0.5033
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6124 - accuracy: 0.5035
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6141 - accuracy: 0.5034
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6118 - accuracy: 0.5036
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6150 - accuracy: 0.5034
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6159 - accuracy: 0.5033
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6175 - accuracy: 0.5032
20000/25000 [=======================>......] - ETA: 13s - loss: 7.6199 - accuracy: 0.5031
20032/25000 [=======================>......] - ETA: 13s - loss: 7.6192 - accuracy: 0.5031
20064/25000 [=======================>......] - ETA: 13s - loss: 7.6185 - accuracy: 0.5031
20096/25000 [=======================>......] - ETA: 13s - loss: 7.6155 - accuracy: 0.5033
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6156 - accuracy: 0.5033
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6141 - accuracy: 0.5034
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6119 - accuracy: 0.5036
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6143 - accuracy: 0.5034
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6174 - accuracy: 0.5032
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6175 - accuracy: 0.5032
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6176 - accuracy: 0.5032
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6207 - accuracy: 0.5030
20384/25000 [=======================>......] - ETA: 12s - loss: 7.6155 - accuracy: 0.5033
20416/25000 [=======================>......] - ETA: 12s - loss: 7.6186 - accuracy: 0.5031
20448/25000 [=======================>......] - ETA: 12s - loss: 7.6179 - accuracy: 0.5032
20480/25000 [=======================>......] - ETA: 12s - loss: 7.6195 - accuracy: 0.5031
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6188 - accuracy: 0.5031
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6189 - accuracy: 0.5031
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6145 - accuracy: 0.5034
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6086 - accuracy: 0.5038
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6124 - accuracy: 0.5035
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6080 - accuracy: 0.5038
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6103 - accuracy: 0.5037
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6112 - accuracy: 0.5036
20768/25000 [=======================>......] - ETA: 11s - loss: 7.6105 - accuracy: 0.5037
20800/25000 [=======================>......] - ETA: 11s - loss: 7.6128 - accuracy: 0.5035
20832/25000 [=======================>......] - ETA: 11s - loss: 7.6129 - accuracy: 0.5035
20864/25000 [========================>.....] - ETA: 11s - loss: 7.6122 - accuracy: 0.5035
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6138 - accuracy: 0.5034
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6168 - accuracy: 0.5032
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6139 - accuracy: 0.5034
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6184 - accuracy: 0.5031
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6207 - accuracy: 0.5030
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6258 - accuracy: 0.5027
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6252 - accuracy: 0.5027
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6267 - accuracy: 0.5026
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6246 - accuracy: 0.5027
21184/25000 [========================>.....] - ETA: 10s - loss: 7.6239 - accuracy: 0.5028
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6233 - accuracy: 0.5028
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6248 - accuracy: 0.5027 
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6263 - accuracy: 0.5026
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6285 - accuracy: 0.5025
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6329 - accuracy: 0.5022
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6365 - accuracy: 0.5020
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6358 - accuracy: 0.5020
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6394 - accuracy: 0.5018
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6402 - accuracy: 0.5017
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6431 - accuracy: 0.5015
21536/25000 [========================>.....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016
21568/25000 [========================>.....] - ETA: 9s - loss: 7.6396 - accuracy: 0.5018
21600/25000 [========================>.....] - ETA: 9s - loss: 7.6389 - accuracy: 0.5018
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6404 - accuracy: 0.5017
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6411 - accuracy: 0.5017
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6412 - accuracy: 0.5017
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6398 - accuracy: 0.5017
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6413 - accuracy: 0.5017
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6413 - accuracy: 0.5017
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6413 - accuracy: 0.5016
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6421 - accuracy: 0.5016
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6449 - accuracy: 0.5014
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6463 - accuracy: 0.5013
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6464 - accuracy: 0.5013
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6429 - accuracy: 0.5015
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6436 - accuracy: 0.5015
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6395 - accuracy: 0.5018
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6402 - accuracy: 0.5017
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6444 - accuracy: 0.5014
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6465 - accuracy: 0.5013
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6431 - accuracy: 0.5015
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6445 - accuracy: 0.5014
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6473 - accuracy: 0.5013
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6487 - accuracy: 0.5012
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6508 - accuracy: 0.5010
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6549 - accuracy: 0.5008
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6563 - accuracy: 0.5007
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6578 - accuracy: 0.5006
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6612 - accuracy: 0.5004
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6599 - accuracy: 0.5004
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6612 - accuracy: 0.5004
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6606 - accuracy: 0.5004
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6572 - accuracy: 0.5006
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6559 - accuracy: 0.5007
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6552 - accuracy: 0.5007
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6552 - accuracy: 0.5007
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6546 - accuracy: 0.5008
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6546 - accuracy: 0.5008
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6553 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6526 - accuracy: 0.5009
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6527 - accuracy: 0.5009
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6527 - accuracy: 0.5009
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6554 - accuracy: 0.5007
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6534 - accuracy: 0.5009
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6514 - accuracy: 0.5010
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6554 - accuracy: 0.5007
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6561 - accuracy: 0.5007
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6535 - accuracy: 0.5009
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6561 - accuracy: 0.5007
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6561 - accuracy: 0.5007
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6542 - accuracy: 0.5008
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6529 - accuracy: 0.5009
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6510 - accuracy: 0.5010
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6510 - accuracy: 0.5010
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6543 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6530 - accuracy: 0.5009
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6511 - accuracy: 0.5010
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6530 - accuracy: 0.5009
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6530 - accuracy: 0.5009
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6537 - accuracy: 0.5008
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5003
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
24192/25000 [============================>.] - ETA: 2s - loss: 7.6565 - accuracy: 0.5007
24224/25000 [============================>.] - ETA: 2s - loss: 7.6559 - accuracy: 0.5007
24256/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24288/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24320/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24352/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24416/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24448/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24544/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24768/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24800/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24832/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24896/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 78s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
