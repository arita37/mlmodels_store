
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6672e19fe4cfa7df885e45d91d645534b8989485', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6672e19fe4cfa7df885e45d91d645534b8989485

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
 40%|████      | 2/5 [00:53<01:19, 26.62s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.40421468001792843, 'embedding_size_factor': 1.2216316877450648, 'layers.choice': 0, 'learning_rate': 0.0008529614546730347, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 7.148639889564363e-10} and reward: 0.383
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xde\xa7?\xcf]\xebX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x8b\xcd\xab)\xf5\x18X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?K\xf3(\xc6\xba\x97\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x08\x90\x02\x98\xdeR\xf8u.' and reward: 0.383
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xde\xa7?\xcf]\xebX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x8b\xcd\xab)\xf5\x18X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?K\xf3(\xc6\xba\x97\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x08\x90\x02\x98\xdeR\xf8u.' and reward: 0.383
 60%|██████    | 3/5 [01:46<01:09, 34.71s/it] 60%|██████    | 3/5 [01:46<01:11, 35.61s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.18943608614718338, 'embedding_size_factor': 1.4446020640690742, 'layers.choice': 2, 'learning_rate': 0.00014461510250320832, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.3552470371747977e-09} and reward: 0.3556
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc8?q\x11W\x9a"X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\x1d\x17\r\xce\x92\x84X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?"\xf4zE\x823\x0bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>,\xd2D)\xcd\xab\xe5u.' and reward: 0.3556
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc8?q\x11W\x9a"X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\x1d\x17\r\xce\x92\x84X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?"\xf4zE\x823\x0bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>,\xd2D)\xcd\xab\xe5u.' and reward: 0.3556
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.346009016037
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -42.89s of remaining time.
Ensemble size: 86
Ensemble weights: 
[0.36046512 0.59302326 0.04651163]
	0.3904	 = Validation accuracy score
	1.08s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 164.0s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fe3f58a1b38> 

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
 [-0.02280932 -0.03185092 -0.02718803 -0.09910353  0.08984938  0.03869525]
 [-0.05691714 -0.13262083  0.1139318   0.02493565  0.12968127 -0.0329993 ]
 [ 0.1298952   0.25509495  0.3514303   0.19815594  0.09233751  0.14348686]
 [-0.07285152  0.13627277 -0.17143884  0.28150994 -0.153644   -0.07395258]
 [-0.22623332  0.12359724  0.4243663   0.14380392 -0.12644602  0.28515351]
 [ 0.12682471  0.19967908  0.1193822   0.22534844  0.0557353  -0.69635677]
 [-1.01326323  0.06014396  0.52397811  0.70458639 -0.49137601  0.06812066]
 [ 0.24504773  0.34220687  0.97350085  0.81366527 -0.11168986 -0.15283926]
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
{'loss': 0.3934505991637707, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 23:41:24.750950: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.48536285758018494, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 23:41:25.874180: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3424256/17464789 [====>.........................] - ETA: 0s
11354112/17464789 [==================>...........] - ETA: 0s
16310272/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 23:41:37.392787: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 23:41:37.397549: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 23:41:37.397701: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5577fc94b410 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 23:41:37.397715: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:38 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:50 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:15 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 1:57 - loss: 6.9479 - accuracy: 0.5469
  160/25000 [..............................] - ETA: 1:46 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:38 - loss: 7.5069 - accuracy: 0.5104
  224/25000 [..............................] - ETA: 1:33 - loss: 7.6666 - accuracy: 0.5000
  256/25000 [..............................] - ETA: 1:29 - loss: 7.6067 - accuracy: 0.5039
  288/25000 [..............................] - ETA: 1:25 - loss: 7.7199 - accuracy: 0.4965
  320/25000 [..............................] - ETA: 1:22 - loss: 7.7145 - accuracy: 0.4969
  352/25000 [..............................] - ETA: 1:20 - loss: 7.8409 - accuracy: 0.4886
  384/25000 [..............................] - ETA: 1:18 - loss: 8.0260 - accuracy: 0.4766
  416/25000 [..............................] - ETA: 1:17 - loss: 7.9615 - accuracy: 0.4808
  448/25000 [..............................] - ETA: 1:15 - loss: 8.1458 - accuracy: 0.4688
  480/25000 [..............................] - ETA: 1:14 - loss: 8.0819 - accuracy: 0.4729
  512/25000 [..............................] - ETA: 1:13 - loss: 7.9960 - accuracy: 0.4785
  544/25000 [..............................] - ETA: 1:12 - loss: 7.8357 - accuracy: 0.4890
  576/25000 [..............................] - ETA: 1:11 - loss: 7.8263 - accuracy: 0.4896
  608/25000 [..............................] - ETA: 1:11 - loss: 7.7927 - accuracy: 0.4918
  640/25000 [..............................] - ETA: 1:10 - loss: 7.7625 - accuracy: 0.4938
  672/25000 [..............................] - ETA: 1:10 - loss: 7.7123 - accuracy: 0.4970
  704/25000 [..............................] - ETA: 1:09 - loss: 7.6013 - accuracy: 0.5043
  736/25000 [..............................] - ETA: 1:09 - loss: 7.5833 - accuracy: 0.5054
  768/25000 [..............................] - ETA: 1:09 - loss: 7.5468 - accuracy: 0.5078
  800/25000 [..............................] - ETA: 1:08 - loss: 7.5133 - accuracy: 0.5100
  832/25000 [..............................] - ETA: 1:08 - loss: 7.4639 - accuracy: 0.5132
  864/25000 [>.............................] - ETA: 1:08 - loss: 7.5424 - accuracy: 0.5081
  896/25000 [>.............................] - ETA: 1:07 - loss: 7.5297 - accuracy: 0.5089
  928/25000 [>.............................] - ETA: 1:07 - loss: 7.6171 - accuracy: 0.5032
  960/25000 [>.............................] - ETA: 1:07 - loss: 7.6187 - accuracy: 0.5031
  992/25000 [>.............................] - ETA: 1:07 - loss: 7.6202 - accuracy: 0.5030
 1024/25000 [>.............................] - ETA: 1:06 - loss: 7.6816 - accuracy: 0.4990
 1056/25000 [>.............................] - ETA: 1:06 - loss: 7.6231 - accuracy: 0.5028
 1088/25000 [>.............................] - ETA: 1:06 - loss: 7.5398 - accuracy: 0.5083
 1120/25000 [>.............................] - ETA: 1:06 - loss: 7.5434 - accuracy: 0.5080
 1152/25000 [>.............................] - ETA: 1:06 - loss: 7.5468 - accuracy: 0.5078
 1184/25000 [>.............................] - ETA: 1:05 - loss: 7.5501 - accuracy: 0.5076
 1216/25000 [>.............................] - ETA: 1:05 - loss: 7.5279 - accuracy: 0.5090
 1248/25000 [>.............................] - ETA: 1:05 - loss: 7.5192 - accuracy: 0.5096
 1280/25000 [>.............................] - ETA: 1:05 - loss: 7.4989 - accuracy: 0.5109
 1312/25000 [>.............................] - ETA: 1:05 - loss: 7.5030 - accuracy: 0.5107
 1344/25000 [>.............................] - ETA: 1:04 - loss: 7.5183 - accuracy: 0.5097
 1376/25000 [>.............................] - ETA: 1:04 - loss: 7.4549 - accuracy: 0.5138
 1408/25000 [>.............................] - ETA: 1:04 - loss: 7.4379 - accuracy: 0.5149
 1440/25000 [>.............................] - ETA: 1:04 - loss: 7.4430 - accuracy: 0.5146
 1472/25000 [>.............................] - ETA: 1:03 - loss: 7.5208 - accuracy: 0.5095
 1504/25000 [>.............................] - ETA: 1:03 - loss: 7.4933 - accuracy: 0.5113
 1536/25000 [>.............................] - ETA: 1:03 - loss: 7.4969 - accuracy: 0.5111
 1568/25000 [>.............................] - ETA: 1:03 - loss: 7.5199 - accuracy: 0.5096
 1600/25000 [>.............................] - ETA: 1:03 - loss: 7.5037 - accuracy: 0.5106
 1632/25000 [>.............................] - ETA: 1:03 - loss: 7.4975 - accuracy: 0.5110
 1664/25000 [>.............................] - ETA: 1:03 - loss: 7.4731 - accuracy: 0.5126
 1696/25000 [=>............................] - ETA: 1:02 - loss: 7.4768 - accuracy: 0.5124
 1728/25000 [=>............................] - ETA: 1:02 - loss: 7.4980 - accuracy: 0.5110
 1760/25000 [=>............................] - ETA: 1:02 - loss: 7.4750 - accuracy: 0.5125
 1792/25000 [=>............................] - ETA: 1:02 - loss: 7.4955 - accuracy: 0.5112
 1824/25000 [=>............................] - ETA: 1:02 - loss: 7.4817 - accuracy: 0.5121
 1856/25000 [=>............................] - ETA: 1:02 - loss: 7.5097 - accuracy: 0.5102
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.5204 - accuracy: 0.5095
 1920/25000 [=>............................] - ETA: 1:01 - loss: 7.5309 - accuracy: 0.5089
 1952/25000 [=>............................] - ETA: 1:01 - loss: 7.5566 - accuracy: 0.5072
 1984/25000 [=>............................] - ETA: 1:01 - loss: 7.5739 - accuracy: 0.5060
 2016/25000 [=>............................] - ETA: 1:01 - loss: 7.5601 - accuracy: 0.5069
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.5992 - accuracy: 0.5044
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.6150 - accuracy: 0.5034
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.6303 - accuracy: 0.5024
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.6452 - accuracy: 0.5014
 2176/25000 [=>............................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 2208/25000 [=>............................] - ETA: 1:00 - loss: 7.6736 - accuracy: 0.4995
 2240/25000 [=>............................] - ETA: 1:00 - loss: 7.6461 - accuracy: 0.5013
 2272/25000 [=>............................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000 
 2304/25000 [=>............................] - ETA: 59s - loss: 7.6733 - accuracy: 0.4996
 2336/25000 [=>............................] - ETA: 59s - loss: 7.6535 - accuracy: 0.5009
 2368/25000 [=>............................] - ETA: 59s - loss: 7.6537 - accuracy: 0.5008
 2400/25000 [=>............................] - ETA: 59s - loss: 7.6411 - accuracy: 0.5017
 2432/25000 [=>............................] - ETA: 59s - loss: 7.6540 - accuracy: 0.5008
 2464/25000 [=>............................] - ETA: 59s - loss: 7.6542 - accuracy: 0.5008
 2496/25000 [=>............................] - ETA: 59s - loss: 7.6298 - accuracy: 0.5024
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.6424 - accuracy: 0.5016
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.6786 - accuracy: 0.4992
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.6844 - accuracy: 0.4988
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.7017 - accuracy: 0.4977
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.7070 - accuracy: 0.4974
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.7180 - accuracy: 0.4967
 2720/25000 [==>...........................] - ETA: 58s - loss: 7.6892 - accuracy: 0.4985
 2752/25000 [==>...........................] - ETA: 58s - loss: 7.6945 - accuracy: 0.4982
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.6886 - accuracy: 0.4986
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.6938 - accuracy: 0.4982
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.6882 - accuracy: 0.4986
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.6826 - accuracy: 0.4990
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.6719 - accuracy: 0.4997
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.7031 - accuracy: 0.4976
 2976/25000 [==>...........................] - ETA: 57s - loss: 7.7078 - accuracy: 0.4973
 3008/25000 [==>...........................] - ETA: 57s - loss: 7.7023 - accuracy: 0.4977
 3040/25000 [==>...........................] - ETA: 57s - loss: 7.6918 - accuracy: 0.4984
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.7115 - accuracy: 0.4971
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.7111 - accuracy: 0.4971
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.7008 - accuracy: 0.4978
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.7102 - accuracy: 0.4972
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.7145 - accuracy: 0.4969
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.7046 - accuracy: 0.4975
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.6901 - accuracy: 0.4985
 3296/25000 [==>...........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 3328/25000 [==>...........................] - ETA: 56s - loss: 7.6390 - accuracy: 0.5018
 3360/25000 [===>..........................] - ETA: 56s - loss: 7.6255 - accuracy: 0.5027
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.6214 - accuracy: 0.5029
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.6039 - accuracy: 0.5041
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.6223 - accuracy: 0.5029
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.6139 - accuracy: 0.5034
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.6100 - accuracy: 0.5037
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.6278 - accuracy: 0.5025
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.6367 - accuracy: 0.5020
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.6242 - accuracy: 0.5028
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.6288 - accuracy: 0.5025
 3680/25000 [===>..........................] - ETA: 55s - loss: 7.6375 - accuracy: 0.5019
 3712/25000 [===>..........................] - ETA: 55s - loss: 7.6377 - accuracy: 0.5019
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.6380 - accuracy: 0.5019
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.6382 - accuracy: 0.5019
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.6304 - accuracy: 0.5024
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.6187 - accuracy: 0.5031
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.6112 - accuracy: 0.5036
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.6116 - accuracy: 0.5036
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.6277 - accuracy: 0.5025
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.6396 - accuracy: 0.5018
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.6398 - accuracy: 0.5017
 4032/25000 [===>..........................] - ETA: 54s - loss: 7.6476 - accuracy: 0.5012
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.6853 - accuracy: 0.4988
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.6852 - accuracy: 0.4988
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.6814 - accuracy: 0.4990
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.6813 - accuracy: 0.4990
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.7065 - accuracy: 0.4974
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.7062 - accuracy: 0.4974
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.7024 - accuracy: 0.4977
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.7092 - accuracy: 0.4972
 4352/25000 [====>.........................] - ETA: 53s - loss: 7.7089 - accuracy: 0.4972
 4384/25000 [====>.........................] - ETA: 53s - loss: 7.7191 - accuracy: 0.4966
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.7118 - accuracy: 0.4971
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.7149 - accuracy: 0.4969
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.7180 - accuracy: 0.4967
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.7210 - accuracy: 0.4965
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.7071 - accuracy: 0.4974
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.7135 - accuracy: 0.4969
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.7265 - accuracy: 0.4961
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.7327 - accuracy: 0.4957
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.7323 - accuracy: 0.4957
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.7253 - accuracy: 0.4962
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.7249 - accuracy: 0.4962
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.7245 - accuracy: 0.4962
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.7273 - accuracy: 0.4960
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.7174 - accuracy: 0.4967
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.7108 - accuracy: 0.4971
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.7199 - accuracy: 0.4965
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.7195 - accuracy: 0.4966
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.7315 - accuracy: 0.4958
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.7311 - accuracy: 0.4958
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.7185 - accuracy: 0.4966
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.7091 - accuracy: 0.4972
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.7118 - accuracy: 0.4971
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.7085 - accuracy: 0.4973
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.7053 - accuracy: 0.4975
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.7021 - accuracy: 0.4977
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.7078 - accuracy: 0.4973
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7075 - accuracy: 0.4973
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.7160 - accuracy: 0.4968
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.7099 - accuracy: 0.4972
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.6982 - accuracy: 0.4979
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.7094 - accuracy: 0.4972
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.7035 - accuracy: 0.4976
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.6920 - accuracy: 0.4983
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.6918 - accuracy: 0.4984
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.6889 - accuracy: 0.4985
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.6722 - accuracy: 0.4996
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.6694 - accuracy: 0.4998
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.6721 - accuracy: 0.4996
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.6721 - accuracy: 0.4996
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.6802 - accuracy: 0.4991
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.6720 - accuracy: 0.4996
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.6506 - accuracy: 0.5010
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.6560 - accuracy: 0.5007
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.6507 - accuracy: 0.5010
 5824/25000 [=====>........................] - ETA: 48s - loss: 7.6508 - accuracy: 0.5010
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.6535 - accuracy: 0.5009
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.6562 - accuracy: 0.5007
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.6718 - accuracy: 0.4997
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.6692 - accuracy: 0.4998
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.6717 - accuracy: 0.4997
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.6717 - accuracy: 0.4997
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.6716 - accuracy: 0.4997
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.6816 - accuracy: 0.4990
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.6890 - accuracy: 0.4985
 6208/25000 [======>.......................] - ETA: 47s - loss: 7.6913 - accuracy: 0.4984
 6240/25000 [======>.......................] - ETA: 47s - loss: 7.7010 - accuracy: 0.4978
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.6886 - accuracy: 0.4986
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.7007 - accuracy: 0.4978
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.6884 - accuracy: 0.4986
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.6931 - accuracy: 0.4983
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.6906 - accuracy: 0.4984
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.6928 - accuracy: 0.4983
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.6927 - accuracy: 0.4983
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.6973 - accuracy: 0.4980
 6528/25000 [======>.......................] - ETA: 46s - loss: 7.6948 - accuracy: 0.4982
 6560/25000 [======>.......................] - ETA: 46s - loss: 7.6923 - accuracy: 0.4983
 6592/25000 [======>.......................] - ETA: 46s - loss: 7.6992 - accuracy: 0.4979
 6624/25000 [======>.......................] - ETA: 46s - loss: 7.7037 - accuracy: 0.4976
 6656/25000 [======>.......................] - ETA: 46s - loss: 7.6989 - accuracy: 0.4979
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.7033 - accuracy: 0.4976
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.7008 - accuracy: 0.4978
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.6939 - accuracy: 0.4982
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.6757 - accuracy: 0.4994
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.6824 - accuracy: 0.4990
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6845 - accuracy: 0.4988
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.6778 - accuracy: 0.4993
 6912/25000 [=======>......................] - ETA: 45s - loss: 7.6755 - accuracy: 0.4994
 6944/25000 [=======>......................] - ETA: 45s - loss: 7.6821 - accuracy: 0.4990
 6976/25000 [=======>......................] - ETA: 45s - loss: 7.6842 - accuracy: 0.4989
 7008/25000 [=======>......................] - ETA: 45s - loss: 7.6885 - accuracy: 0.4986
 7040/25000 [=======>......................] - ETA: 45s - loss: 7.6862 - accuracy: 0.4987
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.6970 - accuracy: 0.4980
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.6947 - accuracy: 0.4982
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.6989 - accuracy: 0.4979
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.6966 - accuracy: 0.4980
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6922 - accuracy: 0.4983
 7232/25000 [=======>......................] - ETA: 44s - loss: 7.6815 - accuracy: 0.4990
 7264/25000 [=======>......................] - ETA: 44s - loss: 7.6814 - accuracy: 0.4990
 7296/25000 [=======>......................] - ETA: 44s - loss: 7.6813 - accuracy: 0.4990
 7328/25000 [=======>......................] - ETA: 44s - loss: 7.6771 - accuracy: 0.4993
 7360/25000 [=======>......................] - ETA: 44s - loss: 7.6770 - accuracy: 0.4993
 7392/25000 [=======>......................] - ETA: 44s - loss: 7.6770 - accuracy: 0.4993
 7424/25000 [=======>......................] - ETA: 44s - loss: 7.6811 - accuracy: 0.4991
 7456/25000 [=======>......................] - ETA: 44s - loss: 7.6810 - accuracy: 0.4991
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.6769 - accuracy: 0.4993
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.6727 - accuracy: 0.4996
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.6768 - accuracy: 0.4993
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.6747 - accuracy: 0.4995
 7616/25000 [========>.....................] - ETA: 43s - loss: 7.6747 - accuracy: 0.4995
 7648/25000 [========>.....................] - ETA: 43s - loss: 7.6766 - accuracy: 0.4993
 7680/25000 [========>.....................] - ETA: 43s - loss: 7.6786 - accuracy: 0.4992
 7712/25000 [========>.....................] - ETA: 43s - loss: 7.6746 - accuracy: 0.4995
 7744/25000 [========>.....................] - ETA: 43s - loss: 7.6785 - accuracy: 0.4992
 7776/25000 [========>.....................] - ETA: 43s - loss: 7.6706 - accuracy: 0.4997
 7808/25000 [========>.....................] - ETA: 43s - loss: 7.6784 - accuracy: 0.4992
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.6803 - accuracy: 0.4991
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.6803 - accuracy: 0.4991
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.6821 - accuracy: 0.4990
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.6839 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 42s - loss: 7.6800 - accuracy: 0.4991
 8032/25000 [========>.....................] - ETA: 42s - loss: 7.6914 - accuracy: 0.4984
 8064/25000 [========>.....................] - ETA: 42s - loss: 7.6970 - accuracy: 0.4980
 8096/25000 [========>.....................] - ETA: 42s - loss: 7.6988 - accuracy: 0.4979
 8128/25000 [========>.....................] - ETA: 42s - loss: 7.6987 - accuracy: 0.4979
 8160/25000 [========>.....................] - ETA: 42s - loss: 7.7023 - accuracy: 0.4977
 8192/25000 [========>.....................] - ETA: 42s - loss: 7.6947 - accuracy: 0.4982
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.6946 - accuracy: 0.4982
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.7000 - accuracy: 0.4978
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.7055 - accuracy: 0.4975
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.7016 - accuracy: 0.4977
 8352/25000 [=========>....................] - ETA: 41s - loss: 7.7033 - accuracy: 0.4976
 8384/25000 [=========>....................] - ETA: 41s - loss: 7.6977 - accuracy: 0.4980
 8416/25000 [=========>....................] - ETA: 41s - loss: 7.6976 - accuracy: 0.4980
 8448/25000 [=========>....................] - ETA: 41s - loss: 7.7065 - accuracy: 0.4974
 8480/25000 [=========>....................] - ETA: 41s - loss: 7.7100 - accuracy: 0.4972
 8512/25000 [=========>....................] - ETA: 41s - loss: 7.7008 - accuracy: 0.4978
 8544/25000 [=========>....................] - ETA: 41s - loss: 7.7025 - accuracy: 0.4977
 8576/25000 [=========>....................] - ETA: 41s - loss: 7.7006 - accuracy: 0.4978
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.6987 - accuracy: 0.4979
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.6950 - accuracy: 0.4981
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.7002 - accuracy: 0.4978
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7019 - accuracy: 0.4977
 8736/25000 [=========>....................] - ETA: 40s - loss: 7.6912 - accuracy: 0.4984
 8768/25000 [=========>....................] - ETA: 40s - loss: 7.6876 - accuracy: 0.4986
 8800/25000 [=========>....................] - ETA: 40s - loss: 7.6806 - accuracy: 0.4991
 8832/25000 [=========>....................] - ETA: 40s - loss: 7.6840 - accuracy: 0.4989
 8864/25000 [=========>....................] - ETA: 40s - loss: 7.6856 - accuracy: 0.4988
 8896/25000 [=========>....................] - ETA: 40s - loss: 7.6770 - accuracy: 0.4993
 8928/25000 [=========>....................] - ETA: 40s - loss: 7.6786 - accuracy: 0.4992
 8960/25000 [=========>....................] - ETA: 40s - loss: 7.6837 - accuracy: 0.4989
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.6803 - accuracy: 0.4991
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.6734 - accuracy: 0.4996
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.6632 - accuracy: 0.5002
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.6599 - accuracy: 0.5004
 9120/25000 [=========>....................] - ETA: 39s - loss: 7.6582 - accuracy: 0.5005
 9152/25000 [=========>....................] - ETA: 39s - loss: 7.6532 - accuracy: 0.5009
 9184/25000 [==========>...................] - ETA: 39s - loss: 7.6566 - accuracy: 0.5007
 9216/25000 [==========>...................] - ETA: 39s - loss: 7.6566 - accuracy: 0.5007
 9248/25000 [==========>...................] - ETA: 39s - loss: 7.6534 - accuracy: 0.5009
 9280/25000 [==========>...................] - ETA: 39s - loss: 7.6617 - accuracy: 0.5003
 9312/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9344/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.6634 - accuracy: 0.5002
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.6682 - accuracy: 0.4999
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.6682 - accuracy: 0.4999
 9504/25000 [==========>...................] - ETA: 38s - loss: 7.6747 - accuracy: 0.4995
 9536/25000 [==========>...................] - ETA: 38s - loss: 7.6779 - accuracy: 0.4993
 9568/25000 [==========>...................] - ETA: 38s - loss: 7.6794 - accuracy: 0.4992
 9600/25000 [==========>...................] - ETA: 38s - loss: 7.6746 - accuracy: 0.4995
 9632/25000 [==========>...................] - ETA: 38s - loss: 7.6778 - accuracy: 0.4993
 9664/25000 [==========>...................] - ETA: 38s - loss: 7.6793 - accuracy: 0.4992
 9696/25000 [==========>...................] - ETA: 38s - loss: 7.6856 - accuracy: 0.4988
 9728/25000 [==========>...................] - ETA: 38s - loss: 7.6887 - accuracy: 0.4986
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.6902 - accuracy: 0.4985
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.6917 - accuracy: 0.4984
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.6947 - accuracy: 0.4982
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.6993 - accuracy: 0.4979
 9888/25000 [==========>...................] - ETA: 37s - loss: 7.7023 - accuracy: 0.4977
 9920/25000 [==========>...................] - ETA: 37s - loss: 7.7114 - accuracy: 0.4971
 9952/25000 [==========>...................] - ETA: 37s - loss: 7.7128 - accuracy: 0.4970
 9984/25000 [==========>...................] - ETA: 37s - loss: 7.7081 - accuracy: 0.4973
10016/25000 [===========>..................] - ETA: 37s - loss: 7.7049 - accuracy: 0.4975
10048/25000 [===========>..................] - ETA: 37s - loss: 7.7048 - accuracy: 0.4975
10080/25000 [===========>..................] - ETA: 37s - loss: 7.7016 - accuracy: 0.4977
10112/25000 [===========>..................] - ETA: 37s - loss: 7.7091 - accuracy: 0.4972
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7029 - accuracy: 0.4976
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7088 - accuracy: 0.4972
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7027 - accuracy: 0.4976
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6981 - accuracy: 0.4979
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6950 - accuracy: 0.4982
10304/25000 [===========>..................] - ETA: 36s - loss: 7.6919 - accuracy: 0.4984
10336/25000 [===========>..................] - ETA: 36s - loss: 7.6978 - accuracy: 0.4980
10368/25000 [===========>..................] - ETA: 36s - loss: 7.7065 - accuracy: 0.4974
10400/25000 [===========>..................] - ETA: 36s - loss: 7.7005 - accuracy: 0.4978
10432/25000 [===========>..................] - ETA: 36s - loss: 7.7048 - accuracy: 0.4975
10464/25000 [===========>..................] - ETA: 36s - loss: 7.7003 - accuracy: 0.4978
10496/25000 [===========>..................] - ETA: 36s - loss: 7.6929 - accuracy: 0.4983
10528/25000 [===========>..................] - ETA: 36s - loss: 7.6943 - accuracy: 0.4982
10560/25000 [===========>..................] - ETA: 36s - loss: 7.6928 - accuracy: 0.4983
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6854 - accuracy: 0.4988
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6811 - accuracy: 0.4991
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6824 - accuracy: 0.4990
10688/25000 [===========>..................] - ETA: 35s - loss: 7.6738 - accuracy: 0.4995
10720/25000 [===========>..................] - ETA: 35s - loss: 7.6709 - accuracy: 0.4997
10752/25000 [===========>..................] - ETA: 35s - loss: 7.6680 - accuracy: 0.4999
10784/25000 [===========>..................] - ETA: 35s - loss: 7.6680 - accuracy: 0.4999
10816/25000 [===========>..................] - ETA: 35s - loss: 7.6723 - accuracy: 0.4996
10848/25000 [============>.................] - ETA: 35s - loss: 7.6709 - accuracy: 0.4997
10880/25000 [============>.................] - ETA: 35s - loss: 7.6708 - accuracy: 0.4997
10912/25000 [============>.................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
10944/25000 [============>.................] - ETA: 35s - loss: 7.6694 - accuracy: 0.4998
10976/25000 [============>.................] - ETA: 35s - loss: 7.6764 - accuracy: 0.4994
11008/25000 [============>.................] - ETA: 35s - loss: 7.6792 - accuracy: 0.4992
11040/25000 [============>.................] - ETA: 35s - loss: 7.6805 - accuracy: 0.4991
11072/25000 [============>.................] - ETA: 35s - loss: 7.6722 - accuracy: 0.4996
11104/25000 [============>.................] - ETA: 34s - loss: 7.6652 - accuracy: 0.5001
11136/25000 [============>.................] - ETA: 34s - loss: 7.6652 - accuracy: 0.5001
11168/25000 [============>.................] - ETA: 34s - loss: 7.6584 - accuracy: 0.5005
11200/25000 [============>.................] - ETA: 34s - loss: 7.6570 - accuracy: 0.5006
11232/25000 [============>.................] - ETA: 34s - loss: 7.6543 - accuracy: 0.5008
11264/25000 [============>.................] - ETA: 34s - loss: 7.6544 - accuracy: 0.5008
11296/25000 [============>.................] - ETA: 34s - loss: 7.6530 - accuracy: 0.5009
11328/25000 [============>.................] - ETA: 34s - loss: 7.6477 - accuracy: 0.5012
11360/25000 [============>.................] - ETA: 34s - loss: 7.6450 - accuracy: 0.5014
11392/25000 [============>.................] - ETA: 34s - loss: 7.6424 - accuracy: 0.5016
11424/25000 [============>.................] - ETA: 34s - loss: 7.6492 - accuracy: 0.5011
11456/25000 [============>.................] - ETA: 34s - loss: 7.6425 - accuracy: 0.5016
11488/25000 [============>.................] - ETA: 33s - loss: 7.6399 - accuracy: 0.5017
11520/25000 [============>.................] - ETA: 33s - loss: 7.6413 - accuracy: 0.5016
11552/25000 [============>.................] - ETA: 33s - loss: 7.6427 - accuracy: 0.5016
11584/25000 [============>.................] - ETA: 33s - loss: 7.6415 - accuracy: 0.5016
11616/25000 [============>.................] - ETA: 33s - loss: 7.6468 - accuracy: 0.5013
11648/25000 [============>.................] - ETA: 33s - loss: 7.6456 - accuracy: 0.5014
11680/25000 [=============>................] - ETA: 33s - loss: 7.6522 - accuracy: 0.5009
11712/25000 [=============>................] - ETA: 33s - loss: 7.6509 - accuracy: 0.5010
11744/25000 [=============>................] - ETA: 33s - loss: 7.6496 - accuracy: 0.5011
11776/25000 [=============>................] - ETA: 33s - loss: 7.6445 - accuracy: 0.5014
11808/25000 [=============>................] - ETA: 33s - loss: 7.6419 - accuracy: 0.5016
11840/25000 [=============>................] - ETA: 33s - loss: 7.6433 - accuracy: 0.5015
11872/25000 [=============>................] - ETA: 32s - loss: 7.6447 - accuracy: 0.5014
11904/25000 [=============>................] - ETA: 32s - loss: 7.6486 - accuracy: 0.5012
11936/25000 [=============>................] - ETA: 32s - loss: 7.6525 - accuracy: 0.5009
11968/25000 [=============>................] - ETA: 32s - loss: 7.6525 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 32s - loss: 7.6500 - accuracy: 0.5011
12032/25000 [=============>................] - ETA: 32s - loss: 7.6488 - accuracy: 0.5012
12064/25000 [=============>................] - ETA: 32s - loss: 7.6488 - accuracy: 0.5012
12096/25000 [=============>................] - ETA: 32s - loss: 7.6514 - accuracy: 0.5010
12128/25000 [=============>................] - ETA: 32s - loss: 7.6502 - accuracy: 0.5011
12160/25000 [=============>................] - ETA: 32s - loss: 7.6502 - accuracy: 0.5011
12192/25000 [=============>................] - ETA: 32s - loss: 7.6503 - accuracy: 0.5011
12224/25000 [=============>................] - ETA: 32s - loss: 7.6491 - accuracy: 0.5011
12256/25000 [=============>................] - ETA: 32s - loss: 7.6454 - accuracy: 0.5014
12288/25000 [=============>................] - ETA: 31s - loss: 7.6429 - accuracy: 0.5015
12320/25000 [=============>................] - ETA: 31s - loss: 7.6492 - accuracy: 0.5011
12352/25000 [=============>................] - ETA: 31s - loss: 7.6455 - accuracy: 0.5014
12384/25000 [=============>................] - ETA: 31s - loss: 7.6468 - accuracy: 0.5013
12416/25000 [=============>................] - ETA: 31s - loss: 7.6518 - accuracy: 0.5010
12448/25000 [=============>................] - ETA: 31s - loss: 7.6494 - accuracy: 0.5011
12480/25000 [=============>................] - ETA: 31s - loss: 7.6556 - accuracy: 0.5007
12512/25000 [==============>...............] - ETA: 31s - loss: 7.6495 - accuracy: 0.5011
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6507 - accuracy: 0.5010
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6471 - accuracy: 0.5013
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6459 - accuracy: 0.5013
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6448 - accuracy: 0.5014
12672/25000 [==============>...............] - ETA: 30s - loss: 7.6460 - accuracy: 0.5013
12704/25000 [==============>...............] - ETA: 30s - loss: 7.6473 - accuracy: 0.5013
12736/25000 [==============>...............] - ETA: 30s - loss: 7.6462 - accuracy: 0.5013
12768/25000 [==============>...............] - ETA: 30s - loss: 7.6378 - accuracy: 0.5019
12800/25000 [==============>...............] - ETA: 30s - loss: 7.6415 - accuracy: 0.5016
12832/25000 [==============>...............] - ETA: 30s - loss: 7.6427 - accuracy: 0.5016
12864/25000 [==============>...............] - ETA: 30s - loss: 7.6440 - accuracy: 0.5015
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6428 - accuracy: 0.5016
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6453 - accuracy: 0.5014
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6536 - accuracy: 0.5008
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6489 - accuracy: 0.5012
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6513 - accuracy: 0.5010
13056/25000 [==============>...............] - ETA: 29s - loss: 7.6525 - accuracy: 0.5009
13088/25000 [==============>...............] - ETA: 29s - loss: 7.6572 - accuracy: 0.5006
13120/25000 [==============>...............] - ETA: 29s - loss: 7.6584 - accuracy: 0.5005
13152/25000 [==============>...............] - ETA: 29s - loss: 7.6585 - accuracy: 0.5005
13184/25000 [==============>...............] - ETA: 29s - loss: 7.6608 - accuracy: 0.5004
13216/25000 [==============>...............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6712 - accuracy: 0.4997
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6747 - accuracy: 0.4995
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6747 - accuracy: 0.4995
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6724 - accuracy: 0.4996
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6701 - accuracy: 0.4998
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13440/25000 [===============>..............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
13472/25000 [===============>..............] - ETA: 28s - loss: 7.6655 - accuracy: 0.5001
13504/25000 [===============>..............] - ETA: 28s - loss: 7.6643 - accuracy: 0.5001
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6598 - accuracy: 0.5004
13568/25000 [===============>..............] - ETA: 28s - loss: 7.6576 - accuracy: 0.5006
13600/25000 [===============>..............] - ETA: 28s - loss: 7.6576 - accuracy: 0.5006
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6565 - accuracy: 0.5007
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6509 - accuracy: 0.5010
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6498 - accuracy: 0.5011
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6532 - accuracy: 0.5009
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6499 - accuracy: 0.5011
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6544 - accuracy: 0.5008
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6555 - accuracy: 0.5007
13856/25000 [===============>..............] - ETA: 27s - loss: 7.6522 - accuracy: 0.5009
13888/25000 [===============>..............] - ETA: 27s - loss: 7.6479 - accuracy: 0.5012
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6457 - accuracy: 0.5014
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6457 - accuracy: 0.5014
13984/25000 [===============>..............] - ETA: 27s - loss: 7.6524 - accuracy: 0.5009
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6557 - accuracy: 0.5007
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6568 - accuracy: 0.5006
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6514 - accuracy: 0.5010
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6481 - accuracy: 0.5012
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6471 - accuracy: 0.5013
14176/25000 [================>.............] - ETA: 27s - loss: 7.6515 - accuracy: 0.5010
14208/25000 [================>.............] - ETA: 27s - loss: 7.6547 - accuracy: 0.5008
14240/25000 [================>.............] - ETA: 26s - loss: 7.6537 - accuracy: 0.5008
14272/25000 [================>.............] - ETA: 26s - loss: 7.6591 - accuracy: 0.5005
14304/25000 [================>.............] - ETA: 26s - loss: 7.6634 - accuracy: 0.5002
14336/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14368/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14400/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14432/25000 [================>.............] - ETA: 26s - loss: 7.6687 - accuracy: 0.4999
14464/25000 [================>.............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
14496/25000 [================>.............] - ETA: 26s - loss: 7.6698 - accuracy: 0.4998
14528/25000 [================>.............] - ETA: 26s - loss: 7.6708 - accuracy: 0.4997
14560/25000 [================>.............] - ETA: 26s - loss: 7.6719 - accuracy: 0.4997
14592/25000 [================>.............] - ETA: 26s - loss: 7.6729 - accuracy: 0.4996
14624/25000 [================>.............] - ETA: 25s - loss: 7.6719 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 25s - loss: 7.6719 - accuracy: 0.4997
14688/25000 [================>.............] - ETA: 25s - loss: 7.6718 - accuracy: 0.4997
14720/25000 [================>.............] - ETA: 25s - loss: 7.6729 - accuracy: 0.4996
14752/25000 [================>.............] - ETA: 25s - loss: 7.6697 - accuracy: 0.4998
14784/25000 [================>.............] - ETA: 25s - loss: 7.6718 - accuracy: 0.4997
14816/25000 [================>.............] - ETA: 25s - loss: 7.6739 - accuracy: 0.4995
14848/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
14880/25000 [================>.............] - ETA: 25s - loss: 7.6687 - accuracy: 0.4999
14912/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
14944/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
14976/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
15008/25000 [=================>............] - ETA: 25s - loss: 7.6636 - accuracy: 0.5002
15040/25000 [=================>............] - ETA: 24s - loss: 7.6595 - accuracy: 0.5005
15072/25000 [=================>............] - ETA: 24s - loss: 7.6575 - accuracy: 0.5006
15104/25000 [=================>............] - ETA: 24s - loss: 7.6636 - accuracy: 0.5002
15136/25000 [=================>............] - ETA: 24s - loss: 7.6646 - accuracy: 0.5001
15168/25000 [=================>............] - ETA: 24s - loss: 7.6626 - accuracy: 0.5003
15200/25000 [=================>............] - ETA: 24s - loss: 7.6565 - accuracy: 0.5007
15232/25000 [=================>............] - ETA: 24s - loss: 7.6566 - accuracy: 0.5007
15264/25000 [=================>............] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
15296/25000 [=================>............] - ETA: 24s - loss: 7.6476 - accuracy: 0.5012
15328/25000 [=================>............] - ETA: 24s - loss: 7.6486 - accuracy: 0.5012
15360/25000 [=================>............] - ETA: 24s - loss: 7.6526 - accuracy: 0.5009
15392/25000 [=================>............] - ETA: 24s - loss: 7.6557 - accuracy: 0.5007
15424/25000 [=================>............] - ETA: 23s - loss: 7.6567 - accuracy: 0.5006
15456/25000 [=================>............] - ETA: 23s - loss: 7.6587 - accuracy: 0.5005
15488/25000 [=================>............] - ETA: 23s - loss: 7.6597 - accuracy: 0.5005
15520/25000 [=================>............] - ETA: 23s - loss: 7.6597 - accuracy: 0.5005
15552/25000 [=================>............] - ETA: 23s - loss: 7.6558 - accuracy: 0.5007
15584/25000 [=================>............] - ETA: 23s - loss: 7.6558 - accuracy: 0.5007
15616/25000 [=================>............] - ETA: 23s - loss: 7.6558 - accuracy: 0.5007
15648/25000 [=================>............] - ETA: 23s - loss: 7.6588 - accuracy: 0.5005
15680/25000 [=================>............] - ETA: 23s - loss: 7.6568 - accuracy: 0.5006
15712/25000 [=================>............] - ETA: 23s - loss: 7.6569 - accuracy: 0.5006
15744/25000 [=================>............] - ETA: 23s - loss: 7.6549 - accuracy: 0.5008
15776/25000 [=================>............] - ETA: 23s - loss: 7.6530 - accuracy: 0.5009
15808/25000 [=================>............] - ETA: 22s - loss: 7.6540 - accuracy: 0.5008
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6540 - accuracy: 0.5008
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6550 - accuracy: 0.5008
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6551 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6551 - accuracy: 0.5008
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6532 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6532 - accuracy: 0.5009
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6571 - accuracy: 0.5006
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6542 - accuracy: 0.5008
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6542 - accuracy: 0.5008
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6514 - accuracy: 0.5010
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6543 - accuracy: 0.5008
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6534 - accuracy: 0.5009
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6506 - accuracy: 0.5010
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6506 - accuracy: 0.5010
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6534 - accuracy: 0.5009
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6516 - accuracy: 0.5010
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6582 - accuracy: 0.5006
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6545 - accuracy: 0.5008
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6535 - accuracy: 0.5009
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6582 - accuracy: 0.5005
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6582 - accuracy: 0.5005
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6592 - accuracy: 0.5005
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6601 - accuracy: 0.5004
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6620 - accuracy: 0.5003
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6602 - accuracy: 0.5004
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6519 - accuracy: 0.5010
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6519 - accuracy: 0.5010
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6538 - accuracy: 0.5008
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6529 - accuracy: 0.5009
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6520 - accuracy: 0.5010
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6438 - accuracy: 0.5015
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6429 - accuracy: 0.5015
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6430 - accuracy: 0.5015
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6376 - accuracy: 0.5019
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6394 - accuracy: 0.5018
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6377 - accuracy: 0.5019
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6377 - accuracy: 0.5019
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6369 - accuracy: 0.5019
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6352 - accuracy: 0.5021
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6352 - accuracy: 0.5020
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6380 - accuracy: 0.5019
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6380 - accuracy: 0.5019
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6381 - accuracy: 0.5019
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6390 - accuracy: 0.5018
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6382 - accuracy: 0.5019
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6400 - accuracy: 0.5017
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6418 - accuracy: 0.5016
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6410 - accuracy: 0.5017
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6428 - accuracy: 0.5016
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6411 - accuracy: 0.5017
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6394 - accuracy: 0.5018
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6403 - accuracy: 0.5017
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6386 - accuracy: 0.5018
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6439 - accuracy: 0.5015
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6396 - accuracy: 0.5018
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6405 - accuracy: 0.5017
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6388 - accuracy: 0.5018
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6362 - accuracy: 0.5020
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6328 - accuracy: 0.5022
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6286 - accuracy: 0.5025
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6286 - accuracy: 0.5025
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6296 - accuracy: 0.5024
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6305 - accuracy: 0.5024
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6340 - accuracy: 0.5021
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6349 - accuracy: 0.5021
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6375 - accuracy: 0.5019
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6393 - accuracy: 0.5018
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6436 - accuracy: 0.5015
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6445 - accuracy: 0.5014
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6420 - accuracy: 0.5016
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6420 - accuracy: 0.5016
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6412 - accuracy: 0.5017
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6387 - accuracy: 0.5018
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6396 - accuracy: 0.5018
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6414 - accuracy: 0.5016
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6431 - accuracy: 0.5015
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6431 - accuracy: 0.5015
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6457 - accuracy: 0.5014
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6449 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6491 - accuracy: 0.5011
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6491 - accuracy: 0.5011
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6508 - accuracy: 0.5010
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6517 - accuracy: 0.5010
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6534 - accuracy: 0.5009
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6526 - accuracy: 0.5009
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6559 - accuracy: 0.5007
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6575 - accuracy: 0.5006
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6584 - accuracy: 0.5005
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6584 - accuracy: 0.5005
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6560 - accuracy: 0.5007
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6552 - accuracy: 0.5007
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6552 - accuracy: 0.5007
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6576 - accuracy: 0.5006
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6585 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6577 - accuracy: 0.5006
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6585 - accuracy: 0.5005
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6569 - accuracy: 0.5006
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6594 - accuracy: 0.5005
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6554 - accuracy: 0.5007
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6586 - accuracy: 0.5005
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6571 - accuracy: 0.5006
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6563 - accuracy: 0.5007
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6571 - accuracy: 0.5006
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6611 - accuracy: 0.5004
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6587 - accuracy: 0.5005
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6603 - accuracy: 0.5004
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6635 - accuracy: 0.5002
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6643 - accuracy: 0.5002
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6651 - accuracy: 0.5001
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6682 - accuracy: 0.4999
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6658 - accuracy: 0.5001
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6720 - accuracy: 0.4996
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6689 - accuracy: 0.4998
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6689 - accuracy: 0.4998
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6682 - accuracy: 0.4999
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6643 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6651 - accuracy: 0.5001
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6651 - accuracy: 0.5001
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6643 - accuracy: 0.5001
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6598 - accuracy: 0.5004
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6629 - accuracy: 0.5002
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6629 - accuracy: 0.5002
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6704 - accuracy: 0.4998
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6718 - accuracy: 0.4997
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6711 - accuracy: 0.4997
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6763 - accuracy: 0.4994
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6770 - accuracy: 0.4993
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6740 - accuracy: 0.4995
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6740 - accuracy: 0.4995
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6747 - accuracy: 0.4995
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6732 - accuracy: 0.4996
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6747 - accuracy: 0.4995
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6806 - accuracy: 0.4991
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6813 - accuracy: 0.4990
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6776 - accuracy: 0.4993
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6783 - accuracy: 0.4992
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6790 - accuracy: 0.4992 
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6775 - accuracy: 0.4993
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6775 - accuracy: 0.4993
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6811 - accuracy: 0.4991
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6811 - accuracy: 0.4991
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6789 - accuracy: 0.4992
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6760 - accuracy: 0.4994
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6753 - accuracy: 0.4994
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6760 - accuracy: 0.4994
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6767 - accuracy: 0.4993
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6760 - accuracy: 0.4994
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6788 - accuracy: 0.4992
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6817 - accuracy: 0.4990
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6802 - accuracy: 0.4991
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6830 - accuracy: 0.4989
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6801 - accuracy: 0.4991
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6766 - accuracy: 0.4994
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6765 - accuracy: 0.4994
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6772 - accuracy: 0.4993
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6772 - accuracy: 0.4993
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6751 - accuracy: 0.4994
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6744 - accuracy: 0.4995
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6744 - accuracy: 0.4995
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6772 - accuracy: 0.4993
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6764 - accuracy: 0.4994
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6764 - accuracy: 0.4994
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6792 - accuracy: 0.4992
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6806 - accuracy: 0.4991
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6792 - accuracy: 0.4992
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6826 - accuracy: 0.4990
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6847 - accuracy: 0.4988
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6832 - accuracy: 0.4989
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6811 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6804 - accuracy: 0.4991
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6824 - accuracy: 0.4990
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6838 - accuracy: 0.4989
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6844 - accuracy: 0.4988
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6837 - accuracy: 0.4989
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6844 - accuracy: 0.4988
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6864 - accuracy: 0.4987
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6884 - accuracy: 0.4986
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6898 - accuracy: 0.4985
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6890 - accuracy: 0.4985
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6883 - accuracy: 0.4986
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6856 - accuracy: 0.4988
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6842 - accuracy: 0.4989
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6876 - accuracy: 0.4986
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6855 - accuracy: 0.4988
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6875 - accuracy: 0.4986
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6895 - accuracy: 0.4985
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6881 - accuracy: 0.4986
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6874 - accuracy: 0.4986
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6901 - accuracy: 0.4985
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6914 - accuracy: 0.4984
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6887 - accuracy: 0.4986
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6873 - accuracy: 0.4987
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6853 - accuracy: 0.4988
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6846 - accuracy: 0.4988
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6852 - accuracy: 0.4988
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6805 - accuracy: 0.4991
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6838 - accuracy: 0.4989
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6804 - accuracy: 0.4991
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6824 - accuracy: 0.4990
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6811 - accuracy: 0.4991
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6797 - accuracy: 0.4991
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6804 - accuracy: 0.4991
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6777 - accuracy: 0.4993
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6751 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6790 - accuracy: 0.4992
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6770 - accuracy: 0.4993
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6763 - accuracy: 0.4994
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24192/25000 [============================>.] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24288/25000 [============================>.] - ETA: 1s - loss: 7.6673 - accuracy: 0.5000
24320/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24352/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24384/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24544/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 74s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

