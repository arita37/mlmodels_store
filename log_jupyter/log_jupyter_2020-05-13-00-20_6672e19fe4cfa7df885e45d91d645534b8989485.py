
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
 40%|████      | 2/5 [00:51<01:16, 25.65s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.07354425858532627, 'embedding_size_factor': 1.1509067316150743, 'layers.choice': 1, 'learning_rate': 0.0005463731593656693, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 3.408845804613538e-08} and reward: 0.3812
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb2\xd3\xcb\xe9n\xbb-X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2j\x1d-P\x87$X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?A\xe7Ol\xe9\xe3\xc0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>bM\x14\xfd\xe8\x8e9u.' and reward: 0.3812
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb2\xd3\xcb\xe9n\xbb-X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2j\x1d-P\x87$X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?A\xe7Ol\xe9\xe3\xc0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>bM\x14\xfd\xe8\x8e9u.' and reward: 0.3812
 60%|██████    | 3/5 [01:58<01:16, 38.18s/it] 60%|██████    | 3/5 [01:58<01:19, 39.57s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4866315965294797, 'embedding_size_factor': 0.8653817219415203, 'layers.choice': 2, 'learning_rate': 0.0028557256903515853, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.271677242951962e-06} and reward: 0.3724
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdf$\xf8\xda\x12\xd7)X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xb15\x02Ip\x9aX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?gd\xe4\x0eH\xf7\x81X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xc3\x0ec\xc5\x86\xea{u.' and reward: 0.3724
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdf$\xf8\xda\x12\xd7)X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xb15\x02Ip\x9aX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?gd\xe4\x0eH\xf7\x81X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xc3\x0ec\xc5\x86\xea{u.' and reward: 0.3724
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 170.21442699432373
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -52.59s of remaining time.
Ensemble size: 16
Ensemble weights: 
[0.625 0.25  0.125]
	0.3868	 = Validation accuracy score
	1.02s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 173.65s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f0a880a1b00> 

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
 [-3.63700092e-02  2.78046489e-01  8.46589506e-02  8.47081691e-02
   8.98853317e-02 -1.15086690e-01]
 [ 1.65358052e-01  8.72223750e-02  1.30735353e-01  1.19031295e-02
   8.80242512e-02  6.63277954e-02]
 [ 9.74235162e-02 -7.67038837e-02  7.75572844e-04 -1.24986798e-01
   1.24270223e-01 -4.29548509e-02]
 [-7.81360492e-02 -5.08292615e-02 -8.34223907e-03  7.11321682e-02
  -2.53734052e-01 -7.47883469e-02]
 [ 7.17812777e-02 -1.42918259e-01  4.61453438e-01 -1.21177390e-01
   1.54067069e-01  2.00253233e-01]
 [ 2.81941772e-01  4.39030826e-01  3.04376483e-01  2.23444536e-01
   3.86724621e-02 -2.95825392e-01]
 [-4.02029395e-01 -1.04178779e-01  1.86989978e-02  7.39206225e-02
   1.34976521e-01 -3.17718834e-01]
 [-3.92356277e-01  8.08960497e-01  3.57977182e-01 -3.30796301e-01
   1.04985714e+00  2.45088995e-01]
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
{'loss': 0.501898143440485, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 00:24:21.440664: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.47611697018146515, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 00:24:22.594825: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2277376/17464789 [==>...........................] - ETA: 0s
10403840/17464789 [================>.............] - ETA: 0s
16244736/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 00:24:34.091650: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 00:24:34.096058: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 00:24:34.096225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55604cc5e0f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:24:34.096239: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:29 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:46 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:12 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 1:54 - loss: 6.7083 - accuracy: 0.5625
  160/25000 [..............................] - ETA: 1:43 - loss: 6.9958 - accuracy: 0.5437
  192/25000 [..............................] - ETA: 1:36 - loss: 6.9479 - accuracy: 0.5469
  224/25000 [..............................] - ETA: 1:30 - loss: 7.0506 - accuracy: 0.5402
  256/25000 [..............................] - ETA: 1:27 - loss: 7.0677 - accuracy: 0.5391
  288/25000 [..............................] - ETA: 1:24 - loss: 7.0277 - accuracy: 0.5417
  320/25000 [..............................] - ETA: 1:21 - loss: 7.1395 - accuracy: 0.5344
  352/25000 [..............................] - ETA: 1:19 - loss: 7.3617 - accuracy: 0.5199
  384/25000 [..............................] - ETA: 1:18 - loss: 7.3472 - accuracy: 0.5208
  416/25000 [..............................] - ETA: 1:16 - loss: 7.4455 - accuracy: 0.5144
  448/25000 [..............................] - ETA: 1:14 - loss: 7.5297 - accuracy: 0.5089
  480/25000 [..............................] - ETA: 1:13 - loss: 7.6027 - accuracy: 0.5042
  512/25000 [..............................] - ETA: 1:12 - loss: 7.5768 - accuracy: 0.5059
  544/25000 [..............................] - ETA: 1:11 - loss: 7.7230 - accuracy: 0.4963
  576/25000 [..............................] - ETA: 1:11 - loss: 7.7199 - accuracy: 0.4965
  608/25000 [..............................] - ETA: 1:10 - loss: 7.6666 - accuracy: 0.5000
  640/25000 [..............................] - ETA: 1:09 - loss: 7.7385 - accuracy: 0.4953
  672/25000 [..............................] - ETA: 1:09 - loss: 7.8492 - accuracy: 0.4881
  704/25000 [..............................] - ETA: 1:08 - loss: 7.8409 - accuracy: 0.4886
  736/25000 [..............................] - ETA: 1:08 - loss: 7.9375 - accuracy: 0.4823
  768/25000 [..............................] - ETA: 1:07 - loss: 7.8064 - accuracy: 0.4909
  800/25000 [..............................] - ETA: 1:07 - loss: 7.8200 - accuracy: 0.4900
  832/25000 [..............................] - ETA: 1:07 - loss: 7.7956 - accuracy: 0.4916
  864/25000 [>.............................] - ETA: 1:06 - loss: 7.8618 - accuracy: 0.4873
  896/25000 [>.............................] - ETA: 1:06 - loss: 7.8377 - accuracy: 0.4888
  928/25000 [>.............................] - ETA: 1:06 - loss: 7.7492 - accuracy: 0.4946
  960/25000 [>.............................] - ETA: 1:06 - loss: 7.8104 - accuracy: 0.4906
  992/25000 [>.............................] - ETA: 1:05 - loss: 7.7594 - accuracy: 0.4940
 1024/25000 [>.............................] - ETA: 1:05 - loss: 7.7415 - accuracy: 0.4951
 1056/25000 [>.............................] - ETA: 1:05 - loss: 7.7102 - accuracy: 0.4972
 1088/25000 [>.............................] - ETA: 1:04 - loss: 7.7512 - accuracy: 0.4945
 1120/25000 [>.............................] - ETA: 1:04 - loss: 7.7351 - accuracy: 0.4955
 1152/25000 [>.............................] - ETA: 1:04 - loss: 7.7199 - accuracy: 0.4965
 1184/25000 [>.............................] - ETA: 1:04 - loss: 7.7184 - accuracy: 0.4966
 1216/25000 [>.............................] - ETA: 1:04 - loss: 7.7675 - accuracy: 0.4934
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.7403 - accuracy: 0.4952
 1280/25000 [>.............................] - ETA: 1:03 - loss: 7.6906 - accuracy: 0.4984
 1312/25000 [>.............................] - ETA: 1:03 - loss: 7.6900 - accuracy: 0.4985
 1344/25000 [>.............................] - ETA: 1:03 - loss: 7.6780 - accuracy: 0.4993
 1376/25000 [>.............................] - ETA: 1:02 - loss: 7.6443 - accuracy: 0.5015
 1408/25000 [>.............................] - ETA: 1:02 - loss: 7.5795 - accuracy: 0.5057
 1440/25000 [>.............................] - ETA: 1:02 - loss: 7.5814 - accuracy: 0.5056
 1472/25000 [>.............................] - ETA: 1:02 - loss: 7.5833 - accuracy: 0.5054
 1504/25000 [>.............................] - ETA: 1:02 - loss: 7.5851 - accuracy: 0.5053
 1536/25000 [>.............................] - ETA: 1:02 - loss: 7.5568 - accuracy: 0.5072
 1568/25000 [>.............................] - ETA: 1:01 - loss: 7.5297 - accuracy: 0.5089
 1600/25000 [>.............................] - ETA: 1:01 - loss: 7.5229 - accuracy: 0.5094
 1632/25000 [>.............................] - ETA: 1:01 - loss: 7.5163 - accuracy: 0.5098
 1664/25000 [>.............................] - ETA: 1:01 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:01 - loss: 7.5581 - accuracy: 0.5071
 1728/25000 [=>............................] - ETA: 1:00 - loss: 7.5513 - accuracy: 0.5075
 1760/25000 [=>............................] - ETA: 1:00 - loss: 7.5534 - accuracy: 0.5074
 1792/25000 [=>............................] - ETA: 1:00 - loss: 7.5639 - accuracy: 0.5067
 1824/25000 [=>............................] - ETA: 1:00 - loss: 7.5405 - accuracy: 0.5082
 1856/25000 [=>............................] - ETA: 1:00 - loss: 7.5510 - accuracy: 0.5075
 1888/25000 [=>............................] - ETA: 1:00 - loss: 7.5286 - accuracy: 0.5090
 1920/25000 [=>............................] - ETA: 1:00 - loss: 7.5069 - accuracy: 0.5104
 1952/25000 [=>............................] - ETA: 59s - loss: 7.4938 - accuracy: 0.5113 
 1984/25000 [=>............................] - ETA: 59s - loss: 7.4889 - accuracy: 0.5116
 2016/25000 [=>............................] - ETA: 59s - loss: 7.4993 - accuracy: 0.5109
 2048/25000 [=>............................] - ETA: 59s - loss: 7.5094 - accuracy: 0.5103
 2080/25000 [=>............................] - ETA: 59s - loss: 7.5192 - accuracy: 0.5096
 2112/25000 [=>............................] - ETA: 59s - loss: 7.5359 - accuracy: 0.5085
 2144/25000 [=>............................] - ETA: 59s - loss: 7.5665 - accuracy: 0.5065
 2176/25000 [=>............................] - ETA: 58s - loss: 7.5750 - accuracy: 0.5060
 2208/25000 [=>............................] - ETA: 58s - loss: 7.5972 - accuracy: 0.5045
 2240/25000 [=>............................] - ETA: 58s - loss: 7.5776 - accuracy: 0.5058
 2272/25000 [=>............................] - ETA: 58s - loss: 7.5721 - accuracy: 0.5062
 2304/25000 [=>............................] - ETA: 58s - loss: 7.5934 - accuracy: 0.5048
 2336/25000 [=>............................] - ETA: 58s - loss: 7.6075 - accuracy: 0.5039
 2368/25000 [=>............................] - ETA: 58s - loss: 7.6083 - accuracy: 0.5038
 2400/25000 [=>............................] - ETA: 58s - loss: 7.6475 - accuracy: 0.5013
 2432/25000 [=>............................] - ETA: 58s - loss: 7.6477 - accuracy: 0.5012
 2464/25000 [=>............................] - ETA: 57s - loss: 7.6542 - accuracy: 0.5008
 2496/25000 [=>............................] - ETA: 57s - loss: 7.6850 - accuracy: 0.4988
 2528/25000 [==>...........................] - ETA: 57s - loss: 7.6727 - accuracy: 0.4996
 2560/25000 [==>...........................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 2592/25000 [==>...........................] - ETA: 57s - loss: 7.6725 - accuracy: 0.4996
 2624/25000 [==>...........................] - ETA: 57s - loss: 7.6491 - accuracy: 0.5011
 2656/25000 [==>...........................] - ETA: 57s - loss: 7.6493 - accuracy: 0.5011
 2688/25000 [==>...........................] - ETA: 56s - loss: 7.6609 - accuracy: 0.5004
 2720/25000 [==>...........................] - ETA: 56s - loss: 7.6384 - accuracy: 0.5018
 2752/25000 [==>...........................] - ETA: 56s - loss: 7.6499 - accuracy: 0.5011
 2784/25000 [==>...........................] - ETA: 56s - loss: 7.6281 - accuracy: 0.5025
 2816/25000 [==>...........................] - ETA: 56s - loss: 7.6122 - accuracy: 0.5036
 2848/25000 [==>...........................] - ETA: 56s - loss: 7.6128 - accuracy: 0.5035
 2880/25000 [==>...........................] - ETA: 56s - loss: 7.6134 - accuracy: 0.5035
 2912/25000 [==>...........................] - ETA: 56s - loss: 7.5876 - accuracy: 0.5052
 2944/25000 [==>...........................] - ETA: 56s - loss: 7.6041 - accuracy: 0.5041
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.6306 - accuracy: 0.5024
 3008/25000 [==>...........................] - ETA: 55s - loss: 7.6156 - accuracy: 0.5033
 3040/25000 [==>...........................] - ETA: 55s - loss: 7.6061 - accuracy: 0.5039
 3072/25000 [==>...........................] - ETA: 55s - loss: 7.5967 - accuracy: 0.5046
 3104/25000 [==>...........................] - ETA: 55s - loss: 7.5925 - accuracy: 0.5048
 3136/25000 [==>...........................] - ETA: 55s - loss: 7.5982 - accuracy: 0.5045
 3168/25000 [==>...........................] - ETA: 55s - loss: 7.6037 - accuracy: 0.5041
 3200/25000 [==>...........................] - ETA: 55s - loss: 7.6139 - accuracy: 0.5034
 3232/25000 [==>...........................] - ETA: 55s - loss: 7.6097 - accuracy: 0.5037
 3264/25000 [==>...........................] - ETA: 54s - loss: 7.6055 - accuracy: 0.5040
 3296/25000 [==>...........................] - ETA: 54s - loss: 7.6108 - accuracy: 0.5036
 3328/25000 [==>...........................] - ETA: 54s - loss: 7.5929 - accuracy: 0.5048
 3360/25000 [===>..........................] - ETA: 54s - loss: 7.5890 - accuracy: 0.5051
 3392/25000 [===>..........................] - ETA: 54s - loss: 7.5988 - accuracy: 0.5044
 3424/25000 [===>..........................] - ETA: 54s - loss: 7.6039 - accuracy: 0.5041
 3456/25000 [===>..........................] - ETA: 54s - loss: 7.6178 - accuracy: 0.5032
 3488/25000 [===>..........................] - ETA: 54s - loss: 7.6139 - accuracy: 0.5034
 3520/25000 [===>..........................] - ETA: 54s - loss: 7.6187 - accuracy: 0.5031
 3552/25000 [===>..........................] - ETA: 54s - loss: 7.6105 - accuracy: 0.5037
 3584/25000 [===>..........................] - ETA: 54s - loss: 7.6238 - accuracy: 0.5028
 3616/25000 [===>..........................] - ETA: 53s - loss: 7.6242 - accuracy: 0.5028
 3648/25000 [===>..........................] - ETA: 53s - loss: 7.6204 - accuracy: 0.5030
 3680/25000 [===>..........................] - ETA: 53s - loss: 7.6250 - accuracy: 0.5027
 3712/25000 [===>..........................] - ETA: 53s - loss: 7.6171 - accuracy: 0.5032
 3744/25000 [===>..........................] - ETA: 53s - loss: 7.6298 - accuracy: 0.5024
 3776/25000 [===>..........................] - ETA: 53s - loss: 7.6260 - accuracy: 0.5026
 3808/25000 [===>..........................] - ETA: 53s - loss: 7.6264 - accuracy: 0.5026
 3840/25000 [===>..........................] - ETA: 53s - loss: 7.6347 - accuracy: 0.5021
 3872/25000 [===>..........................] - ETA: 53s - loss: 7.6349 - accuracy: 0.5021
 3904/25000 [===>..........................] - ETA: 53s - loss: 7.6391 - accuracy: 0.5018
 3936/25000 [===>..........................] - ETA: 53s - loss: 7.6510 - accuracy: 0.5010
 3968/25000 [===>..........................] - ETA: 52s - loss: 7.6782 - accuracy: 0.4992
 4000/25000 [===>..........................] - ETA: 52s - loss: 7.6896 - accuracy: 0.4985
 4032/25000 [===>..........................] - ETA: 52s - loss: 7.7008 - accuracy: 0.4978
 4064/25000 [===>..........................] - ETA: 52s - loss: 7.6968 - accuracy: 0.4980
 4096/25000 [===>..........................] - ETA: 52s - loss: 7.7003 - accuracy: 0.4978
 4128/25000 [===>..........................] - ETA: 52s - loss: 7.6963 - accuracy: 0.4981
 4160/25000 [===>..........................] - ETA: 52s - loss: 7.7072 - accuracy: 0.4974
 4192/25000 [====>.........................] - ETA: 52s - loss: 7.6995 - accuracy: 0.4979
 4224/25000 [====>.........................] - ETA: 52s - loss: 7.7065 - accuracy: 0.4974
 4256/25000 [====>.........................] - ETA: 52s - loss: 7.7207 - accuracy: 0.4965
 4288/25000 [====>.........................] - ETA: 52s - loss: 7.7274 - accuracy: 0.4960
 4320/25000 [====>.........................] - ETA: 52s - loss: 7.7128 - accuracy: 0.4970
 4352/25000 [====>.........................] - ETA: 51s - loss: 7.7054 - accuracy: 0.4975
 4384/25000 [====>.........................] - ETA: 51s - loss: 7.7226 - accuracy: 0.4964
 4416/25000 [====>.........................] - ETA: 51s - loss: 7.6875 - accuracy: 0.4986
 4448/25000 [====>.........................] - ETA: 51s - loss: 7.6873 - accuracy: 0.4987
 4480/25000 [====>.........................] - ETA: 51s - loss: 7.6940 - accuracy: 0.4982
 4512/25000 [====>.........................] - ETA: 51s - loss: 7.6870 - accuracy: 0.4987
 4544/25000 [====>.........................] - ETA: 51s - loss: 7.7037 - accuracy: 0.4976
 4576/25000 [====>.........................] - ETA: 51s - loss: 7.7135 - accuracy: 0.4969
 4608/25000 [====>.........................] - ETA: 51s - loss: 7.7099 - accuracy: 0.4972
 4640/25000 [====>.........................] - ETA: 51s - loss: 7.7096 - accuracy: 0.4972
 4672/25000 [====>.........................] - ETA: 51s - loss: 7.6896 - accuracy: 0.4985
 4704/25000 [====>.........................] - ETA: 51s - loss: 7.6992 - accuracy: 0.4979
 4736/25000 [====>.........................] - ETA: 51s - loss: 7.6990 - accuracy: 0.4979
 4768/25000 [====>.........................] - ETA: 50s - loss: 7.7149 - accuracy: 0.4969
 4800/25000 [====>.........................] - ETA: 50s - loss: 7.6954 - accuracy: 0.4981
 4832/25000 [====>.........................] - ETA: 50s - loss: 7.6825 - accuracy: 0.4990
 4864/25000 [====>.........................] - ETA: 50s - loss: 7.6887 - accuracy: 0.4986
 4896/25000 [====>.........................] - ETA: 50s - loss: 7.6791 - accuracy: 0.4992
 4928/25000 [====>.........................] - ETA: 50s - loss: 7.6853 - accuracy: 0.4988
 4960/25000 [====>.........................] - ETA: 50s - loss: 7.7037 - accuracy: 0.4976
 4992/25000 [====>.........................] - ETA: 50s - loss: 7.6973 - accuracy: 0.4980
 5024/25000 [=====>........................] - ETA: 50s - loss: 7.7032 - accuracy: 0.4976
 5056/25000 [=====>........................] - ETA: 50s - loss: 7.7060 - accuracy: 0.4974
 5088/25000 [=====>........................] - ETA: 49s - loss: 7.7148 - accuracy: 0.4969
 5120/25000 [=====>........................] - ETA: 49s - loss: 7.7205 - accuracy: 0.4965
 5152/25000 [=====>........................] - ETA: 49s - loss: 7.7232 - accuracy: 0.4963
 5184/25000 [=====>........................] - ETA: 49s - loss: 7.7258 - accuracy: 0.4961
 5216/25000 [=====>........................] - ETA: 49s - loss: 7.7225 - accuracy: 0.4964
 5248/25000 [=====>........................] - ETA: 49s - loss: 7.7280 - accuracy: 0.4960
 5280/25000 [=====>........................] - ETA: 49s - loss: 7.7247 - accuracy: 0.4962
 5312/25000 [=====>........................] - ETA: 49s - loss: 7.7157 - accuracy: 0.4968
 5344/25000 [=====>........................] - ETA: 49s - loss: 7.7269 - accuracy: 0.4961
 5376/25000 [=====>........................] - ETA: 49s - loss: 7.7151 - accuracy: 0.4968
 5408/25000 [=====>........................] - ETA: 49s - loss: 7.7177 - accuracy: 0.4967
 5440/25000 [=====>........................] - ETA: 48s - loss: 7.7230 - accuracy: 0.4963
 5472/25000 [=====>........................] - ETA: 48s - loss: 7.7227 - accuracy: 0.4963
 5504/25000 [=====>........................] - ETA: 48s - loss: 7.7223 - accuracy: 0.4964
 5536/25000 [=====>........................] - ETA: 48s - loss: 7.7276 - accuracy: 0.4960
 5568/25000 [=====>........................] - ETA: 48s - loss: 7.7244 - accuracy: 0.4962
 5600/25000 [=====>........................] - ETA: 48s - loss: 7.7269 - accuracy: 0.4961
 5632/25000 [=====>........................] - ETA: 48s - loss: 7.7292 - accuracy: 0.4959
 5664/25000 [=====>........................] - ETA: 48s - loss: 7.7235 - accuracy: 0.4963
 5696/25000 [=====>........................] - ETA: 48s - loss: 7.7258 - accuracy: 0.4961
 5728/25000 [=====>........................] - ETA: 48s - loss: 7.7309 - accuracy: 0.4958
 5760/25000 [=====>........................] - ETA: 47s - loss: 7.7305 - accuracy: 0.4958
 5792/25000 [=====>........................] - ETA: 47s - loss: 7.7434 - accuracy: 0.4950
 5824/25000 [=====>........................] - ETA: 47s - loss: 7.7403 - accuracy: 0.4952
 5856/25000 [======>.......................] - ETA: 47s - loss: 7.7268 - accuracy: 0.4961
 5888/25000 [======>.......................] - ETA: 47s - loss: 7.7239 - accuracy: 0.4963
 5920/25000 [======>.......................] - ETA: 47s - loss: 7.7236 - accuracy: 0.4963
 5952/25000 [======>.......................] - ETA: 47s - loss: 7.7233 - accuracy: 0.4963
 5984/25000 [======>.......................] - ETA: 47s - loss: 7.7179 - accuracy: 0.4967
 6016/25000 [======>.......................] - ETA: 47s - loss: 7.7125 - accuracy: 0.4970
 6048/25000 [======>.......................] - ETA: 47s - loss: 7.7097 - accuracy: 0.4972
 6080/25000 [======>.......................] - ETA: 47s - loss: 7.7196 - accuracy: 0.4965
 6112/25000 [======>.......................] - ETA: 47s - loss: 7.7268 - accuracy: 0.4961
 6144/25000 [======>.......................] - ETA: 46s - loss: 7.7390 - accuracy: 0.4953
 6176/25000 [======>.......................] - ETA: 46s - loss: 7.7436 - accuracy: 0.4950
 6208/25000 [======>.......................] - ETA: 46s - loss: 7.7358 - accuracy: 0.4955
 6240/25000 [======>.......................] - ETA: 46s - loss: 7.7453 - accuracy: 0.4949
 6272/25000 [======>.......................] - ETA: 46s - loss: 7.7400 - accuracy: 0.4952
 6304/25000 [======>.......................] - ETA: 46s - loss: 7.7323 - accuracy: 0.4957
 6336/25000 [======>.......................] - ETA: 46s - loss: 7.7368 - accuracy: 0.4954
 6368/25000 [======>.......................] - ETA: 46s - loss: 7.7340 - accuracy: 0.4956
 6400/25000 [======>.......................] - ETA: 46s - loss: 7.7337 - accuracy: 0.4956
 6432/25000 [======>.......................] - ETA: 46s - loss: 7.7358 - accuracy: 0.4955
 6464/25000 [======>.......................] - ETA: 46s - loss: 7.7212 - accuracy: 0.4964
 6496/25000 [======>.......................] - ETA: 46s - loss: 7.7209 - accuracy: 0.4965
 6528/25000 [======>.......................] - ETA: 45s - loss: 7.7159 - accuracy: 0.4968
 6560/25000 [======>.......................] - ETA: 45s - loss: 7.7180 - accuracy: 0.4966
 6592/25000 [======>.......................] - ETA: 45s - loss: 7.7062 - accuracy: 0.4974
 6624/25000 [======>.......................] - ETA: 45s - loss: 7.6990 - accuracy: 0.4979
 6656/25000 [======>.......................] - ETA: 45s - loss: 7.7035 - accuracy: 0.4976
 6688/25000 [=======>......................] - ETA: 45s - loss: 7.7010 - accuracy: 0.4978
 6720/25000 [=======>......................] - ETA: 45s - loss: 7.7008 - accuracy: 0.4978
 6752/25000 [=======>......................] - ETA: 45s - loss: 7.6984 - accuracy: 0.4979
 6784/25000 [=======>......................] - ETA: 45s - loss: 7.6915 - accuracy: 0.4984
 6816/25000 [=======>......................] - ETA: 45s - loss: 7.6891 - accuracy: 0.4985
 6848/25000 [=======>......................] - ETA: 45s - loss: 7.6711 - accuracy: 0.4997
 6880/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 6912/25000 [=======>......................] - ETA: 44s - loss: 7.6777 - accuracy: 0.4993
 6944/25000 [=======>......................] - ETA: 44s - loss: 7.6688 - accuracy: 0.4999
 6976/25000 [=======>......................] - ETA: 44s - loss: 7.6688 - accuracy: 0.4999
 7008/25000 [=======>......................] - ETA: 44s - loss: 7.6601 - accuracy: 0.5004
 7040/25000 [=======>......................] - ETA: 44s - loss: 7.6579 - accuracy: 0.5006
 7072/25000 [=======>......................] - ETA: 44s - loss: 7.6579 - accuracy: 0.5006
 7104/25000 [=======>......................] - ETA: 44s - loss: 7.6580 - accuracy: 0.5006
 7136/25000 [=======>......................] - ETA: 44s - loss: 7.6645 - accuracy: 0.5001
 7168/25000 [=======>......................] - ETA: 44s - loss: 7.6773 - accuracy: 0.4993
 7200/25000 [=======>......................] - ETA: 44s - loss: 7.6730 - accuracy: 0.4996
 7232/25000 [=======>......................] - ETA: 44s - loss: 7.6751 - accuracy: 0.4994
 7264/25000 [=======>......................] - ETA: 44s - loss: 7.6708 - accuracy: 0.4997
 7296/25000 [=======>......................] - ETA: 43s - loss: 7.6582 - accuracy: 0.5005
 7328/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7360/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7392/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7424/25000 [=======>......................] - ETA: 43s - loss: 7.6584 - accuracy: 0.5005
 7456/25000 [=======>......................] - ETA: 43s - loss: 7.6604 - accuracy: 0.5004
 7488/25000 [=======>......................] - ETA: 43s - loss: 7.6687 - accuracy: 0.4999
 7520/25000 [========>.....................] - ETA: 43s - loss: 7.6687 - accuracy: 0.4999
 7552/25000 [========>.....................] - ETA: 43s - loss: 7.6686 - accuracy: 0.4999
 7584/25000 [========>.....................] - ETA: 43s - loss: 7.6747 - accuracy: 0.4995
 7616/25000 [========>.....................] - ETA: 43s - loss: 7.6747 - accuracy: 0.4995
 7648/25000 [========>.....................] - ETA: 43s - loss: 7.6807 - accuracy: 0.4991
 7680/25000 [========>.....................] - ETA: 42s - loss: 7.6786 - accuracy: 0.4992
 7712/25000 [========>.....................] - ETA: 42s - loss: 7.6805 - accuracy: 0.4991
 7744/25000 [========>.....................] - ETA: 42s - loss: 7.6805 - accuracy: 0.4991
 7776/25000 [========>.....................] - ETA: 42s - loss: 7.6824 - accuracy: 0.4990
 7808/25000 [========>.....................] - ETA: 42s - loss: 7.6823 - accuracy: 0.4990
 7840/25000 [========>.....................] - ETA: 42s - loss: 7.6862 - accuracy: 0.4987
 7872/25000 [========>.....................] - ETA: 42s - loss: 7.6861 - accuracy: 0.4987
 7904/25000 [========>.....................] - ETA: 42s - loss: 7.6880 - accuracy: 0.4986
 7936/25000 [========>.....................] - ETA: 42s - loss: 7.6917 - accuracy: 0.4984
 7968/25000 [========>.....................] - ETA: 42s - loss: 7.6955 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 42s - loss: 7.6858 - accuracy: 0.4988
 8032/25000 [========>.....................] - ETA: 42s - loss: 7.6914 - accuracy: 0.4984
 8064/25000 [========>.....................] - ETA: 41s - loss: 7.6932 - accuracy: 0.4983
 8096/25000 [========>.....................] - ETA: 41s - loss: 7.6893 - accuracy: 0.4985
 8128/25000 [========>.....................] - ETA: 41s - loss: 7.6930 - accuracy: 0.4983
 8160/25000 [========>.....................] - ETA: 41s - loss: 7.6873 - accuracy: 0.4987
 8192/25000 [========>.....................] - ETA: 41s - loss: 7.6891 - accuracy: 0.4985
 8224/25000 [========>.....................] - ETA: 41s - loss: 7.6909 - accuracy: 0.4984
 8256/25000 [========>.....................] - ETA: 41s - loss: 7.6870 - accuracy: 0.4987
 8288/25000 [========>.....................] - ETA: 41s - loss: 7.6925 - accuracy: 0.4983
 8320/25000 [========>.....................] - ETA: 41s - loss: 7.6961 - accuracy: 0.4981
 8352/25000 [=========>....................] - ETA: 41s - loss: 7.7033 - accuracy: 0.4976
 8384/25000 [=========>....................] - ETA: 41s - loss: 7.7050 - accuracy: 0.4975
 8416/25000 [=========>....................] - ETA: 41s - loss: 7.7049 - accuracy: 0.4975
 8448/25000 [=========>....................] - ETA: 40s - loss: 7.7102 - accuracy: 0.4972
 8480/25000 [=========>....................] - ETA: 40s - loss: 7.7154 - accuracy: 0.4968
 8512/25000 [=========>....................] - ETA: 40s - loss: 7.7117 - accuracy: 0.4971
 8544/25000 [=========>....................] - ETA: 40s - loss: 7.7115 - accuracy: 0.4971
 8576/25000 [=========>....................] - ETA: 40s - loss: 7.7131 - accuracy: 0.4970
 8608/25000 [=========>....................] - ETA: 40s - loss: 7.7040 - accuracy: 0.4976
 8640/25000 [=========>....................] - ETA: 40s - loss: 7.6932 - accuracy: 0.4983
 8672/25000 [=========>....................] - ETA: 40s - loss: 7.7038 - accuracy: 0.4976
 8704/25000 [=========>....................] - ETA: 40s - loss: 7.7036 - accuracy: 0.4976
 8736/25000 [=========>....................] - ETA: 40s - loss: 7.6982 - accuracy: 0.4979
 8768/25000 [=========>....................] - ETA: 40s - loss: 7.6998 - accuracy: 0.4978
 8800/25000 [=========>....................] - ETA: 40s - loss: 7.7032 - accuracy: 0.4976
 8832/25000 [=========>....................] - ETA: 39s - loss: 7.6944 - accuracy: 0.4982
 8864/25000 [=========>....................] - ETA: 39s - loss: 7.6908 - accuracy: 0.4984
 8896/25000 [=========>....................] - ETA: 39s - loss: 7.6925 - accuracy: 0.4983
 8928/25000 [=========>....................] - ETA: 39s - loss: 7.6855 - accuracy: 0.4988
 8960/25000 [=========>....................] - ETA: 39s - loss: 7.6854 - accuracy: 0.4988
 8992/25000 [=========>....................] - ETA: 39s - loss: 7.6786 - accuracy: 0.4992
 9024/25000 [=========>....................] - ETA: 39s - loss: 7.6768 - accuracy: 0.4993
 9056/25000 [=========>....................] - ETA: 39s - loss: 7.6802 - accuracy: 0.4991
 9088/25000 [=========>....................] - ETA: 39s - loss: 7.6767 - accuracy: 0.4993
 9120/25000 [=========>....................] - ETA: 39s - loss: 7.6717 - accuracy: 0.4997
 9152/25000 [=========>....................] - ETA: 39s - loss: 7.6733 - accuracy: 0.4996
 9184/25000 [==========>...................] - ETA: 39s - loss: 7.6783 - accuracy: 0.4992
 9216/25000 [==========>...................] - ETA: 38s - loss: 7.6766 - accuracy: 0.4993
 9248/25000 [==========>...................] - ETA: 38s - loss: 7.6782 - accuracy: 0.4992
 9280/25000 [==========>...................] - ETA: 38s - loss: 7.6699 - accuracy: 0.4998
 9312/25000 [==========>...................] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
 9344/25000 [==========>...................] - ETA: 38s - loss: 7.6699 - accuracy: 0.4998
 9376/25000 [==========>...................] - ETA: 38s - loss: 7.6683 - accuracy: 0.4999
 9408/25000 [==========>...................] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
 9440/25000 [==========>...................] - ETA: 38s - loss: 7.6650 - accuracy: 0.5001
 9472/25000 [==========>...................] - ETA: 38s - loss: 7.6634 - accuracy: 0.5002
 9504/25000 [==========>...................] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
 9536/25000 [==========>...................] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
 9568/25000 [==========>...................] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6698 - accuracy: 0.4998
 9632/25000 [==========>...................] - ETA: 37s - loss: 7.6746 - accuracy: 0.4995
 9664/25000 [==========>...................] - ETA: 37s - loss: 7.6746 - accuracy: 0.4995
 9696/25000 [==========>...................] - ETA: 37s - loss: 7.6777 - accuracy: 0.4993
 9728/25000 [==========>...................] - ETA: 37s - loss: 7.6777 - accuracy: 0.4993
 9760/25000 [==========>...................] - ETA: 37s - loss: 7.6808 - accuracy: 0.4991
 9792/25000 [==========>...................] - ETA: 37s - loss: 7.6776 - accuracy: 0.4993
 9824/25000 [==========>...................] - ETA: 37s - loss: 7.6760 - accuracy: 0.4994
 9856/25000 [==========>...................] - ETA: 37s - loss: 7.6760 - accuracy: 0.4994
 9888/25000 [==========>...................] - ETA: 37s - loss: 7.6759 - accuracy: 0.4994
 9920/25000 [==========>...................] - ETA: 37s - loss: 7.6774 - accuracy: 0.4993
 9952/25000 [==========>...................] - ETA: 37s - loss: 7.6805 - accuracy: 0.4991
 9984/25000 [==========>...................] - ETA: 37s - loss: 7.6758 - accuracy: 0.4994
10016/25000 [===========>..................] - ETA: 36s - loss: 7.6819 - accuracy: 0.4990
10048/25000 [===========>..................] - ETA: 36s - loss: 7.6758 - accuracy: 0.4994
10080/25000 [===========>..................] - ETA: 36s - loss: 7.6681 - accuracy: 0.4999
10112/25000 [===========>..................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10144/25000 [===========>..................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
10176/25000 [===========>..................] - ETA: 36s - loss: 7.6606 - accuracy: 0.5004
10208/25000 [===========>..................] - ETA: 36s - loss: 7.6606 - accuracy: 0.5004
10240/25000 [===========>..................] - ETA: 36s - loss: 7.6561 - accuracy: 0.5007
10272/25000 [===========>..................] - ETA: 36s - loss: 7.6517 - accuracy: 0.5010
10304/25000 [===========>..................] - ETA: 36s - loss: 7.6517 - accuracy: 0.5010
10336/25000 [===========>..................] - ETA: 36s - loss: 7.6503 - accuracy: 0.5011
10368/25000 [===========>..................] - ETA: 36s - loss: 7.6459 - accuracy: 0.5014
10400/25000 [===========>..................] - ETA: 35s - loss: 7.6445 - accuracy: 0.5014
10432/25000 [===========>..................] - ETA: 35s - loss: 7.6387 - accuracy: 0.5018
10464/25000 [===========>..................] - ETA: 35s - loss: 7.6373 - accuracy: 0.5019
10496/25000 [===========>..................] - ETA: 35s - loss: 7.6330 - accuracy: 0.5022
10528/25000 [===========>..................] - ETA: 35s - loss: 7.6302 - accuracy: 0.5024
10560/25000 [===========>..................] - ETA: 35s - loss: 7.6318 - accuracy: 0.5023
10592/25000 [===========>..................] - ETA: 35s - loss: 7.6319 - accuracy: 0.5023
10624/25000 [===========>..................] - ETA: 35s - loss: 7.6277 - accuracy: 0.5025
10656/25000 [===========>..................] - ETA: 35s - loss: 7.6278 - accuracy: 0.5025
10688/25000 [===========>..................] - ETA: 35s - loss: 7.6265 - accuracy: 0.5026
10720/25000 [===========>..................] - ETA: 35s - loss: 7.6266 - accuracy: 0.5026
10752/25000 [===========>..................] - ETA: 35s - loss: 7.6310 - accuracy: 0.5023
10784/25000 [===========>..................] - ETA: 34s - loss: 7.6211 - accuracy: 0.5030
10816/25000 [===========>..................] - ETA: 34s - loss: 7.6269 - accuracy: 0.5026
10848/25000 [============>.................] - ETA: 34s - loss: 7.6256 - accuracy: 0.5027
10880/25000 [============>.................] - ETA: 34s - loss: 7.6257 - accuracy: 0.5027
10912/25000 [============>.................] - ETA: 34s - loss: 7.6287 - accuracy: 0.5025
10944/25000 [============>.................] - ETA: 34s - loss: 7.6330 - accuracy: 0.5022
10976/25000 [============>.................] - ETA: 34s - loss: 7.6289 - accuracy: 0.5025
11008/25000 [============>.................] - ETA: 34s - loss: 7.6248 - accuracy: 0.5027
11040/25000 [============>.................] - ETA: 34s - loss: 7.6236 - accuracy: 0.5028
11072/25000 [============>.................] - ETA: 34s - loss: 7.6223 - accuracy: 0.5029
11104/25000 [============>.................] - ETA: 34s - loss: 7.6197 - accuracy: 0.5031
11136/25000 [============>.................] - ETA: 34s - loss: 7.6212 - accuracy: 0.5030
11168/25000 [============>.................] - ETA: 33s - loss: 7.6172 - accuracy: 0.5032
11200/25000 [============>.................] - ETA: 33s - loss: 7.6160 - accuracy: 0.5033
11232/25000 [============>.................] - ETA: 33s - loss: 7.6120 - accuracy: 0.5036
11264/25000 [============>.................] - ETA: 33s - loss: 7.6163 - accuracy: 0.5033
11296/25000 [============>.................] - ETA: 33s - loss: 7.6191 - accuracy: 0.5031
11328/25000 [============>.................] - ETA: 33s - loss: 7.6220 - accuracy: 0.5029
11360/25000 [============>.................] - ETA: 33s - loss: 7.6180 - accuracy: 0.5032
11392/25000 [============>.................] - ETA: 33s - loss: 7.6235 - accuracy: 0.5028
11424/25000 [============>.................] - ETA: 33s - loss: 7.6290 - accuracy: 0.5025
11456/25000 [============>.................] - ETA: 33s - loss: 7.6318 - accuracy: 0.5023
11488/25000 [============>.................] - ETA: 33s - loss: 7.6346 - accuracy: 0.5021
11520/25000 [============>.................] - ETA: 33s - loss: 7.6373 - accuracy: 0.5019
11552/25000 [============>.................] - ETA: 33s - loss: 7.6414 - accuracy: 0.5016
11584/25000 [============>.................] - ETA: 32s - loss: 7.6415 - accuracy: 0.5016
11616/25000 [============>.................] - ETA: 32s - loss: 7.6415 - accuracy: 0.5016
11648/25000 [============>.................] - ETA: 32s - loss: 7.6429 - accuracy: 0.5015
11680/25000 [=============>................] - ETA: 32s - loss: 7.6456 - accuracy: 0.5014
11712/25000 [=============>................] - ETA: 32s - loss: 7.6378 - accuracy: 0.5019
11744/25000 [=============>................] - ETA: 32s - loss: 7.6418 - accuracy: 0.5016
11776/25000 [=============>................] - ETA: 32s - loss: 7.6419 - accuracy: 0.5016
11808/25000 [=============>................] - ETA: 32s - loss: 7.6458 - accuracy: 0.5014
11840/25000 [=============>................] - ETA: 32s - loss: 7.6472 - accuracy: 0.5013
11872/25000 [=============>................] - ETA: 32s - loss: 7.6485 - accuracy: 0.5012
11904/25000 [=============>................] - ETA: 32s - loss: 7.6473 - accuracy: 0.5013
11936/25000 [=============>................] - ETA: 32s - loss: 7.6461 - accuracy: 0.5013
11968/25000 [=============>................] - ETA: 31s - loss: 7.6487 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 31s - loss: 7.6526 - accuracy: 0.5009
12032/25000 [=============>................] - ETA: 31s - loss: 7.6564 - accuracy: 0.5007
12064/25000 [=============>................] - ETA: 31s - loss: 7.6565 - accuracy: 0.5007
12096/25000 [=============>................] - ETA: 31s - loss: 7.6527 - accuracy: 0.5009
12128/25000 [=============>................] - ETA: 31s - loss: 7.6540 - accuracy: 0.5008
12160/25000 [=============>................] - ETA: 31s - loss: 7.6565 - accuracy: 0.5007
12192/25000 [=============>................] - ETA: 31s - loss: 7.6578 - accuracy: 0.5006
12224/25000 [=============>................] - ETA: 31s - loss: 7.6553 - accuracy: 0.5007
12256/25000 [=============>................] - ETA: 31s - loss: 7.6529 - accuracy: 0.5009
12288/25000 [=============>................] - ETA: 31s - loss: 7.6554 - accuracy: 0.5007
12320/25000 [=============>................] - ETA: 31s - loss: 7.6529 - accuracy: 0.5009
12352/25000 [=============>................] - ETA: 30s - loss: 7.6554 - accuracy: 0.5007
12384/25000 [=============>................] - ETA: 30s - loss: 7.6542 - accuracy: 0.5008
12416/25000 [=============>................] - ETA: 30s - loss: 7.6493 - accuracy: 0.5011
12448/25000 [=============>................] - ETA: 30s - loss: 7.6481 - accuracy: 0.5012
12480/25000 [=============>................] - ETA: 30s - loss: 7.6457 - accuracy: 0.5014
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6531 - accuracy: 0.5009
12544/25000 [==============>...............] - ETA: 30s - loss: 7.6544 - accuracy: 0.5008
12576/25000 [==============>...............] - ETA: 30s - loss: 7.6520 - accuracy: 0.5010
12608/25000 [==============>...............] - ETA: 30s - loss: 7.6472 - accuracy: 0.5013
12640/25000 [==============>...............] - ETA: 30s - loss: 7.6460 - accuracy: 0.5013
12672/25000 [==============>...............] - ETA: 30s - loss: 7.6448 - accuracy: 0.5014
12704/25000 [==============>...............] - ETA: 30s - loss: 7.6401 - accuracy: 0.5017
12736/25000 [==============>...............] - ETA: 30s - loss: 7.6449 - accuracy: 0.5014
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6414 - accuracy: 0.5016
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6415 - accuracy: 0.5016
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6403 - accuracy: 0.5017
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6368 - accuracy: 0.5019
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6274 - accuracy: 0.5026
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6287 - accuracy: 0.5025
12960/25000 [==============>...............] - ETA: 29s - loss: 7.6311 - accuracy: 0.5023
12992/25000 [==============>...............] - ETA: 29s - loss: 7.6277 - accuracy: 0.5025
13024/25000 [==============>...............] - ETA: 29s - loss: 7.6242 - accuracy: 0.5028
13056/25000 [==============>...............] - ETA: 29s - loss: 7.6255 - accuracy: 0.5027
13088/25000 [==============>...............] - ETA: 29s - loss: 7.6303 - accuracy: 0.5024
13120/25000 [==============>...............] - ETA: 29s - loss: 7.6281 - accuracy: 0.5025
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6270 - accuracy: 0.5026
13184/25000 [==============>...............] - ETA: 28s - loss: 7.6259 - accuracy: 0.5027
13216/25000 [==============>...............] - ETA: 28s - loss: 7.6249 - accuracy: 0.5027
13248/25000 [==============>...............] - ETA: 28s - loss: 7.6273 - accuracy: 0.5026
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6262 - accuracy: 0.5026
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6252 - accuracy: 0.5027
13344/25000 [===============>..............] - ETA: 28s - loss: 7.6253 - accuracy: 0.5027
13376/25000 [===============>..............] - ETA: 28s - loss: 7.6208 - accuracy: 0.5030
13408/25000 [===============>..............] - ETA: 28s - loss: 7.6232 - accuracy: 0.5028
13440/25000 [===============>..............] - ETA: 28s - loss: 7.6301 - accuracy: 0.5024
13472/25000 [===============>..............] - ETA: 28s - loss: 7.6336 - accuracy: 0.5022
13504/25000 [===============>..............] - ETA: 28s - loss: 7.6394 - accuracy: 0.5018
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6428 - accuracy: 0.5016
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6406 - accuracy: 0.5017
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6418 - accuracy: 0.5016
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6441 - accuracy: 0.5015
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6475 - accuracy: 0.5012
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6465 - accuracy: 0.5013
13728/25000 [===============>..............] - ETA: 27s - loss: 7.6465 - accuracy: 0.5013
13760/25000 [===============>..............] - ETA: 27s - loss: 7.6521 - accuracy: 0.5009
13792/25000 [===============>..............] - ETA: 27s - loss: 7.6533 - accuracy: 0.5009
13824/25000 [===============>..............] - ETA: 27s - loss: 7.6455 - accuracy: 0.5014
13856/25000 [===============>..............] - ETA: 27s - loss: 7.6412 - accuracy: 0.5017
13888/25000 [===============>..............] - ETA: 27s - loss: 7.6456 - accuracy: 0.5014
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6490 - accuracy: 0.5011
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6468 - accuracy: 0.5013
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6502 - accuracy: 0.5011
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6480 - accuracy: 0.5012
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6470 - accuracy: 0.5013
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6470 - accuracy: 0.5013
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6514 - accuracy: 0.5010
14144/25000 [===============>..............] - ETA: 26s - loss: 7.6471 - accuracy: 0.5013
14176/25000 [================>.............] - ETA: 26s - loss: 7.6461 - accuracy: 0.5013
14208/25000 [================>.............] - ETA: 26s - loss: 7.6461 - accuracy: 0.5013
14240/25000 [================>.............] - ETA: 26s - loss: 7.6494 - accuracy: 0.5011
14272/25000 [================>.............] - ETA: 26s - loss: 7.6462 - accuracy: 0.5013
14304/25000 [================>.............] - ETA: 26s - loss: 7.6473 - accuracy: 0.5013
14336/25000 [================>.............] - ETA: 26s - loss: 7.6495 - accuracy: 0.5011
14368/25000 [================>.............] - ETA: 25s - loss: 7.6517 - accuracy: 0.5010
14400/25000 [================>.............] - ETA: 25s - loss: 7.6496 - accuracy: 0.5011
14432/25000 [================>.............] - ETA: 25s - loss: 7.6528 - accuracy: 0.5009
14464/25000 [================>.............] - ETA: 25s - loss: 7.6539 - accuracy: 0.5008
14496/25000 [================>.............] - ETA: 25s - loss: 7.6518 - accuracy: 0.5010
14528/25000 [================>.............] - ETA: 25s - loss: 7.6476 - accuracy: 0.5012
14560/25000 [================>.............] - ETA: 25s - loss: 7.6498 - accuracy: 0.5011
14592/25000 [================>.............] - ETA: 25s - loss: 7.6488 - accuracy: 0.5012
14624/25000 [================>.............] - ETA: 25s - loss: 7.6488 - accuracy: 0.5012
14656/25000 [================>.............] - ETA: 25s - loss: 7.6457 - accuracy: 0.5014
14688/25000 [================>.............] - ETA: 25s - loss: 7.6437 - accuracy: 0.5015
14720/25000 [================>.............] - ETA: 25s - loss: 7.6468 - accuracy: 0.5013
14752/25000 [================>.............] - ETA: 25s - loss: 7.6469 - accuracy: 0.5013
14784/25000 [================>.............] - ETA: 24s - loss: 7.6480 - accuracy: 0.5012
14816/25000 [================>.............] - ETA: 24s - loss: 7.6501 - accuracy: 0.5011
14848/25000 [================>.............] - ETA: 24s - loss: 7.6460 - accuracy: 0.5013
14880/25000 [================>.............] - ETA: 24s - loss: 7.6429 - accuracy: 0.5015
14912/25000 [================>.............] - ETA: 24s - loss: 7.6399 - accuracy: 0.5017
14944/25000 [================>.............] - ETA: 24s - loss: 7.6389 - accuracy: 0.5018
14976/25000 [================>.............] - ETA: 24s - loss: 7.6339 - accuracy: 0.5021
15008/25000 [=================>............] - ETA: 24s - loss: 7.6360 - accuracy: 0.5020
15040/25000 [=================>............] - ETA: 24s - loss: 7.6320 - accuracy: 0.5023
15072/25000 [=================>............] - ETA: 24s - loss: 7.6310 - accuracy: 0.5023
15104/25000 [=================>............] - ETA: 24s - loss: 7.6311 - accuracy: 0.5023
15136/25000 [=================>............] - ETA: 24s - loss: 7.6302 - accuracy: 0.5024
15168/25000 [=================>............] - ETA: 23s - loss: 7.6282 - accuracy: 0.5025
15200/25000 [=================>............] - ETA: 23s - loss: 7.6283 - accuracy: 0.5025
15232/25000 [=================>............] - ETA: 23s - loss: 7.6294 - accuracy: 0.5024
15264/25000 [=================>............] - ETA: 23s - loss: 7.6284 - accuracy: 0.5025
15296/25000 [=================>............] - ETA: 23s - loss: 7.6315 - accuracy: 0.5023
15328/25000 [=================>............] - ETA: 23s - loss: 7.6286 - accuracy: 0.5025
15360/25000 [=================>............] - ETA: 23s - loss: 7.6287 - accuracy: 0.5025
15392/25000 [=================>............] - ETA: 23s - loss: 7.6258 - accuracy: 0.5027
15424/25000 [=================>............] - ETA: 23s - loss: 7.6249 - accuracy: 0.5027
15456/25000 [=================>............] - ETA: 23s - loss: 7.6269 - accuracy: 0.5026
15488/25000 [=================>............] - ETA: 23s - loss: 7.6280 - accuracy: 0.5025
15520/25000 [=================>............] - ETA: 23s - loss: 7.6311 - accuracy: 0.5023
15552/25000 [=================>............] - ETA: 23s - loss: 7.6311 - accuracy: 0.5023
15584/25000 [=================>............] - ETA: 22s - loss: 7.6351 - accuracy: 0.5021
15616/25000 [=================>............] - ETA: 22s - loss: 7.6381 - accuracy: 0.5019
15648/25000 [=================>............] - ETA: 22s - loss: 7.6362 - accuracy: 0.5020
15680/25000 [=================>............] - ETA: 22s - loss: 7.6383 - accuracy: 0.5018
15712/25000 [=================>............] - ETA: 22s - loss: 7.6373 - accuracy: 0.5019
15744/25000 [=================>............] - ETA: 22s - loss: 7.6355 - accuracy: 0.5020
15776/25000 [=================>............] - ETA: 22s - loss: 7.6355 - accuracy: 0.5020
15808/25000 [=================>............] - ETA: 22s - loss: 7.6375 - accuracy: 0.5019
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6385 - accuracy: 0.5018
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6434 - accuracy: 0.5015
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6444 - accuracy: 0.5014
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6426 - accuracy: 0.5016
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6369 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6436 - accuracy: 0.5015
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6427 - accuracy: 0.5016
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6447 - accuracy: 0.5014
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6438 - accuracy: 0.5015
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6438 - accuracy: 0.5015
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6429 - accuracy: 0.5015
16192/25000 [==================>...........] - ETA: 21s - loss: 7.6439 - accuracy: 0.5015
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6468 - accuracy: 0.5013
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6449 - accuracy: 0.5014
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6440 - accuracy: 0.5015
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6441 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6469 - accuracy: 0.5013
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6479 - accuracy: 0.5012
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6479 - accuracy: 0.5012
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6498 - accuracy: 0.5011
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6508 - accuracy: 0.5010
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6453 - accuracy: 0.5014
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6472 - accuracy: 0.5013
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6472 - accuracy: 0.5013
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6463 - accuracy: 0.5013
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6482 - accuracy: 0.5012
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6464 - accuracy: 0.5013
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6437 - accuracy: 0.5015
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6446 - accuracy: 0.5014
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6447 - accuracy: 0.5014
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6447 - accuracy: 0.5014
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6448 - accuracy: 0.5014
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6484 - accuracy: 0.5012
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6448 - accuracy: 0.5014
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6467 - accuracy: 0.5013
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6476 - accuracy: 0.5012
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6441 - accuracy: 0.5015
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6414 - accuracy: 0.5016
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6450 - accuracy: 0.5014
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6469 - accuracy: 0.5013
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6496 - accuracy: 0.5011
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6505 - accuracy: 0.5010
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6515 - accuracy: 0.5010
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6515 - accuracy: 0.5010
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6524 - accuracy: 0.5009
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6533 - accuracy: 0.5009
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6524 - accuracy: 0.5009
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6542 - accuracy: 0.5008
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
17408/25000 [===================>..........] - ETA: 18s - loss: 7.6569 - accuracy: 0.5006
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6552 - accuracy: 0.5007
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6536 - accuracy: 0.5009
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6571 - accuracy: 0.5006
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6614 - accuracy: 0.5003
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6623 - accuracy: 0.5003
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6580 - accuracy: 0.5006
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6614 - accuracy: 0.5003
17792/25000 [====================>.........] - ETA: 17s - loss: 7.6632 - accuracy: 0.5002
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6615 - accuracy: 0.5003
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6597 - accuracy: 0.5004
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6580 - accuracy: 0.5006
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6606 - accuracy: 0.5004
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6598 - accuracy: 0.5004
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6581 - accuracy: 0.5006
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6573 - accuracy: 0.5006
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6564 - accuracy: 0.5007
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6632 - accuracy: 0.5002
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6624 - accuracy: 0.5003
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6624 - accuracy: 0.5003
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6582 - accuracy: 0.5006
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6590 - accuracy: 0.5005
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6574 - accuracy: 0.5006
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6582 - accuracy: 0.5005
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6574 - accuracy: 0.5006
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6599 - accuracy: 0.5004
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6591 - accuracy: 0.5005
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6566 - accuracy: 0.5007
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6617 - accuracy: 0.5003
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6641 - accuracy: 0.5002
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6601 - accuracy: 0.5004
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6601 - accuracy: 0.5004
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6617 - accuracy: 0.5003
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6617 - accuracy: 0.5003
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6585 - accuracy: 0.5005
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6618 - accuracy: 0.5003
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6601 - accuracy: 0.5004
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6585 - accuracy: 0.5005
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6569 - accuracy: 0.5006
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6602 - accuracy: 0.5004
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6586 - accuracy: 0.5005
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6570 - accuracy: 0.5006
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6578 - accuracy: 0.5006
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6563 - accuracy: 0.5007
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6547 - accuracy: 0.5008
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6531 - accuracy: 0.5009
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6523 - accuracy: 0.5009
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6555 - accuracy: 0.5007
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6548 - accuracy: 0.5008
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6540 - accuracy: 0.5008
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6524 - accuracy: 0.5009
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6517 - accuracy: 0.5010
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6533 - accuracy: 0.5009
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6549 - accuracy: 0.5008
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6541 - accuracy: 0.5008
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6557 - accuracy: 0.5007
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6565 - accuracy: 0.5007
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6557 - accuracy: 0.5007
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6565 - accuracy: 0.5007
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6565 - accuracy: 0.5007
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6565 - accuracy: 0.5007
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6581 - accuracy: 0.5006
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6566 - accuracy: 0.5007
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6543 - accuracy: 0.5008
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6574 - accuracy: 0.5006
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6574 - accuracy: 0.5006
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6582 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6582 - accuracy: 0.5005
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6605 - accuracy: 0.5004
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6590 - accuracy: 0.5005
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6605 - accuracy: 0.5004
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6620 - accuracy: 0.5003
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6628 - accuracy: 0.5002
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6628 - accuracy: 0.5002
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6613 - accuracy: 0.5003
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6591 - accuracy: 0.5005
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6636 - accuracy: 0.5002
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6636 - accuracy: 0.5002
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6636 - accuracy: 0.5002
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6621 - accuracy: 0.5003
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6614 - accuracy: 0.5003
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6607 - accuracy: 0.5004
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6599 - accuracy: 0.5004
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6600 - accuracy: 0.5004
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6563 - accuracy: 0.5007
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6541 - accuracy: 0.5008
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6541 - accuracy: 0.5008
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6526 - accuracy: 0.5009
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6527 - accuracy: 0.5009
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6475 - accuracy: 0.5012 
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6476 - accuracy: 0.5012
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6454 - accuracy: 0.5014
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6462 - accuracy: 0.5013
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6447 - accuracy: 0.5014
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6455 - accuracy: 0.5014
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6470 - accuracy: 0.5013
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6485 - accuracy: 0.5012
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6485 - accuracy: 0.5012
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6471 - accuracy: 0.5013
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6464 - accuracy: 0.5013
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6450 - accuracy: 0.5014
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6393 - accuracy: 0.5018
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6394 - accuracy: 0.5018
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6373 - accuracy: 0.5019
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6366 - accuracy: 0.5020
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6373 - accuracy: 0.5019
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6374 - accuracy: 0.5019
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6381 - accuracy: 0.5019
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6375 - accuracy: 0.5019
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6404 - accuracy: 0.5017
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6418 - accuracy: 0.5016
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6411 - accuracy: 0.5017
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6398 - accuracy: 0.5018
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6419 - accuracy: 0.5016
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6427 - accuracy: 0.5016
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6469 - accuracy: 0.5013
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6449 - accuracy: 0.5014
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6470 - accuracy: 0.5013
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6512 - accuracy: 0.5010
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6492 - accuracy: 0.5011
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6492 - accuracy: 0.5011
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6499 - accuracy: 0.5011
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6478 - accuracy: 0.5012
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6479 - accuracy: 0.5012
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6500 - accuracy: 0.5011
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6493 - accuracy: 0.5011
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6466 - accuracy: 0.5013
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6487 - accuracy: 0.5012
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6473 - accuracy: 0.5013
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6487 - accuracy: 0.5012
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6515 - accuracy: 0.5010
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6529 - accuracy: 0.5009
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6550 - accuracy: 0.5008
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6529 - accuracy: 0.5009
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6550 - accuracy: 0.5008
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6571 - accuracy: 0.5006
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6571 - accuracy: 0.5006
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6537 - accuracy: 0.5008
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6537 - accuracy: 0.5008
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6544 - accuracy: 0.5008
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6531 - accuracy: 0.5009
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6498 - accuracy: 0.5011
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6478 - accuracy: 0.5012
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6452 - accuracy: 0.5014
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6472 - accuracy: 0.5013
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6492 - accuracy: 0.5011
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6520 - accuracy: 0.5010
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6520 - accuracy: 0.5010
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6507 - accuracy: 0.5010
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6507 - accuracy: 0.5010
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6514 - accuracy: 0.5010
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6508 - accuracy: 0.5010
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6475 - accuracy: 0.5012
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6463 - accuracy: 0.5013
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6470 - accuracy: 0.5013
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6450 - accuracy: 0.5014
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6450 - accuracy: 0.5014
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6457 - accuracy: 0.5014
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6471 - accuracy: 0.5013
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6477 - accuracy: 0.5012
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6510 - accuracy: 0.5010
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6523 - accuracy: 0.5009
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6530 - accuracy: 0.5009
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6511 - accuracy: 0.5010
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6524 - accuracy: 0.5009
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6563 - accuracy: 0.5007
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6582 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6608 - accuracy: 0.5004
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6608 - accuracy: 0.5004
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6602 - accuracy: 0.5004
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6602 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24192/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24224/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24256/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24320/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24352/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24416/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24448/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24512/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24608/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24640/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24672/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24736/25000 [============================>.] - ETA: 0s - loss: 7.6561 - accuracy: 0.5007
24768/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24800/25000 [============================>.] - ETA: 0s - loss: 7.6567 - accuracy: 0.5006
24832/25000 [============================>.] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24864/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24960/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 71s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

