
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:54<01:22, 27.35s/it] 40%|████      | 2/5 [00:54<01:22, 27.35s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.44087443236435425, 'embedding_size_factor': 1.2436456251997443, 'layers.choice': 2, 'learning_rate': 0.004501687132779124, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 6.385725148246321e-08} and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdc7Ie)p\x13X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xe5\xf8\xf4\x80\xbd\xf4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?rp\\p1\x9a\x9fX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>q$<\xa5\xfb8$u.' and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdc7Ie)p\x13X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xe5\xf8\xf4\x80\xbd\xf4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?rp\\p1\x9a\x9fX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>q$<\xa5\xfb8$u.' and reward: 0.388
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 110.09117865562439
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.44087443236435425, 'embedding_size_factor': 1.2436456251997443, 'layers.choice': 2, 'learning_rate': 0.004501687132779124, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 6.385725148246321e-08}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the 8.1s of remaining time.
Ensemble size: 15
Ensemble weights: 
[0.73333333 0.26666667]
	0.3926	 = Validation accuracy score
	0.94s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 112.87s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fa4a9cc5b38> 

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
 [ 2.12006010e-02 -1.79943759e-02  8.10321495e-02  4.53825966e-02
   3.20241377e-02 -8.92542973e-02]
 [ 3.12917262e-01  5.92352822e-02  9.49128345e-03  2.53362581e-02
   9.89377871e-03 -7.89653510e-02]
 [-1.38245881e-01  1.37371346e-01  8.01553428e-02  3.67735177e-02
   1.27762295e-02  7.12989569e-02]
 [ 1.18502133e-01  9.73255113e-02  9.95452628e-02  3.46226245e-01
   2.10812822e-01 -1.43344969e-01]
 [ 1.72450408e-01  4.14034203e-02  4.19593602e-02 -1.21914193e-01
  -9.27101150e-02 -4.44202200e-02]
 [ 8.87957737e-02  4.16636348e-01 -6.95806384e-01 -8.06297779e-01
  -1.19810820e-01  1.71086982e-01]
 [-3.44261117e-02  5.76300323e-01  3.64154518e-01  4.33185726e-01
   3.55305940e-01 -3.47423136e-01]
 [ 3.67454857e-01 -4.03935760e-01  9.13398713e-02 -1.03704229e-01
  -1.04517765e-01 -7.45411497e-04]
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
{'loss': 0.5392566919326782, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 05:17:05.910614: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.48044345155358315, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 05:17:07.097592: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2539520/17464789 [===>..........................] - ETA: 0s
11190272/17464789 [==================>...........] - ETA: 0s
16211968/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 05:17:19.188168: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 05:17:19.193140: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 05:17:19.193278: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561cc23ff700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 05:17:19.193293: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:08 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 3:19 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:42 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 2:23 - loss: 7.5468 - accuracy: 0.5078
  160/25000 [..............................] - ETA: 2:12 - loss: 7.8583 - accuracy: 0.4875
  192/25000 [..............................] - ETA: 2:04 - loss: 7.6666 - accuracy: 0.5000
  224/25000 [..............................] - ETA: 1:58 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:54 - loss: 7.6067 - accuracy: 0.5039
  288/25000 [..............................] - ETA: 1:51 - loss: 7.4537 - accuracy: 0.5139
  320/25000 [..............................] - ETA: 1:49 - loss: 7.4750 - accuracy: 0.5125
  352/25000 [..............................] - ETA: 1:47 - loss: 7.5359 - accuracy: 0.5085
  384/25000 [..............................] - ETA: 1:46 - loss: 7.5468 - accuracy: 0.5078
  416/25000 [..............................] - ETA: 1:45 - loss: 7.5929 - accuracy: 0.5048
  448/25000 [..............................] - ETA: 1:43 - loss: 7.4955 - accuracy: 0.5112
  480/25000 [..............................] - ETA: 1:42 - loss: 7.6986 - accuracy: 0.4979
  512/25000 [..............................] - ETA: 1:41 - loss: 7.6666 - accuracy: 0.5000
  544/25000 [..............................] - ETA: 1:40 - loss: 7.6948 - accuracy: 0.4982
  576/25000 [..............................] - ETA: 1:39 - loss: 7.7465 - accuracy: 0.4948
  608/25000 [..............................] - ETA: 1:38 - loss: 7.7423 - accuracy: 0.4951
  640/25000 [..............................] - ETA: 1:38 - loss: 7.7385 - accuracy: 0.4953
  672/25000 [..............................] - ETA: 1:37 - loss: 7.6666 - accuracy: 0.5000
  704/25000 [..............................] - ETA: 1:37 - loss: 7.6013 - accuracy: 0.5043
  736/25000 [..............................] - ETA: 1:36 - loss: 7.6041 - accuracy: 0.5041
  768/25000 [..............................] - ETA: 1:35 - loss: 7.5868 - accuracy: 0.5052
  800/25000 [..............................] - ETA: 1:35 - loss: 7.6091 - accuracy: 0.5038
  832/25000 [..............................] - ETA: 1:35 - loss: 7.5745 - accuracy: 0.5060
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.5246 - accuracy: 0.5093
  896/25000 [>.............................] - ETA: 1:34 - loss: 7.5468 - accuracy: 0.5078
  928/25000 [>.............................] - ETA: 1:33 - loss: 7.4849 - accuracy: 0.5119
  960/25000 [>.............................] - ETA: 1:33 - loss: 7.4430 - accuracy: 0.5146
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.4348 - accuracy: 0.5151
 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.4570 - accuracy: 0.5137
 1056/25000 [>.............................] - ETA: 1:32 - loss: 7.4924 - accuracy: 0.5114
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.4552 - accuracy: 0.5138
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.4339 - accuracy: 0.5152
 1152/25000 [>.............................] - ETA: 1:31 - loss: 7.4004 - accuracy: 0.5174
 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.3947 - accuracy: 0.5177
 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.3640 - accuracy: 0.5197
 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.3840 - accuracy: 0.5184
 1280/25000 [>.............................] - ETA: 1:30 - loss: 7.4151 - accuracy: 0.5164
 1312/25000 [>.............................] - ETA: 1:30 - loss: 7.4212 - accuracy: 0.5160
 1344/25000 [>.............................] - ETA: 1:29 - loss: 7.4384 - accuracy: 0.5149
 1376/25000 [>.............................] - ETA: 1:29 - loss: 7.4438 - accuracy: 0.5145
 1408/25000 [>.............................] - ETA: 1:29 - loss: 7.4488 - accuracy: 0.5142
 1440/25000 [>.............................] - ETA: 1:29 - loss: 7.5282 - accuracy: 0.5090
 1472/25000 [>.............................] - ETA: 1:28 - loss: 7.5312 - accuracy: 0.5088
 1504/25000 [>.............................] - ETA: 1:28 - loss: 7.5137 - accuracy: 0.5100
 1536/25000 [>.............................] - ETA: 1:28 - loss: 7.4470 - accuracy: 0.5143
 1568/25000 [>.............................] - ETA: 1:28 - loss: 7.4906 - accuracy: 0.5115
 1600/25000 [>.............................] - ETA: 1:28 - loss: 7.4750 - accuracy: 0.5125
 1632/25000 [>.............................] - ETA: 1:27 - loss: 7.4975 - accuracy: 0.5110
 1664/25000 [>.............................] - ETA: 1:27 - loss: 7.5008 - accuracy: 0.5108
 1696/25000 [=>............................] - ETA: 1:27 - loss: 7.5310 - accuracy: 0.5088
 1728/25000 [=>............................] - ETA: 1:27 - loss: 7.4892 - accuracy: 0.5116
 1760/25000 [=>............................] - ETA: 1:27 - loss: 7.5447 - accuracy: 0.5080
 1792/25000 [=>............................] - ETA: 1:27 - loss: 7.5725 - accuracy: 0.5061
 1824/25000 [=>............................] - ETA: 1:26 - loss: 7.5573 - accuracy: 0.5071
 1856/25000 [=>............................] - ETA: 1:26 - loss: 7.5675 - accuracy: 0.5065
 1888/25000 [=>............................] - ETA: 1:26 - loss: 7.5286 - accuracy: 0.5090
 1920/25000 [=>............................] - ETA: 1:26 - loss: 7.5548 - accuracy: 0.5073
 1952/25000 [=>............................] - ETA: 1:26 - loss: 7.5174 - accuracy: 0.5097
 1984/25000 [=>............................] - ETA: 1:25 - loss: 7.5198 - accuracy: 0.5096
 2016/25000 [=>............................] - ETA: 1:25 - loss: 7.5373 - accuracy: 0.5084
 2048/25000 [=>............................] - ETA: 1:25 - loss: 7.5992 - accuracy: 0.5044
 2080/25000 [=>............................] - ETA: 1:25 - loss: 7.5929 - accuracy: 0.5048
 2112/25000 [=>............................] - ETA: 1:25 - loss: 7.6085 - accuracy: 0.5038
 2144/25000 [=>............................] - ETA: 1:24 - loss: 7.6166 - accuracy: 0.5033
 2176/25000 [=>............................] - ETA: 1:24 - loss: 7.6032 - accuracy: 0.5041
 2208/25000 [=>............................] - ETA: 1:24 - loss: 7.6388 - accuracy: 0.5018
 2240/25000 [=>............................] - ETA: 1:24 - loss: 7.6392 - accuracy: 0.5018
 2272/25000 [=>............................] - ETA: 1:24 - loss: 7.6734 - accuracy: 0.4996
 2304/25000 [=>............................] - ETA: 1:24 - loss: 7.6533 - accuracy: 0.5009
 2336/25000 [=>............................] - ETA: 1:23 - loss: 7.6404 - accuracy: 0.5017
 2368/25000 [=>............................] - ETA: 1:23 - loss: 7.6666 - accuracy: 0.5000
 2400/25000 [=>............................] - ETA: 1:23 - loss: 7.6602 - accuracy: 0.5004
 2432/25000 [=>............................] - ETA: 1:23 - loss: 7.6603 - accuracy: 0.5004
 2464/25000 [=>............................] - ETA: 1:23 - loss: 7.6728 - accuracy: 0.4996
 2496/25000 [=>............................] - ETA: 1:23 - loss: 7.7096 - accuracy: 0.4972
 2528/25000 [==>...........................] - ETA: 1:23 - loss: 7.7091 - accuracy: 0.4972
 2560/25000 [==>...........................] - ETA: 1:22 - loss: 7.7385 - accuracy: 0.4953
 2592/25000 [==>...........................] - ETA: 1:22 - loss: 7.7435 - accuracy: 0.4950
 2624/25000 [==>...........................] - ETA: 1:22 - loss: 7.7426 - accuracy: 0.4950
 2656/25000 [==>...........................] - ETA: 1:22 - loss: 7.7532 - accuracy: 0.4944
 2688/25000 [==>...........................] - ETA: 1:22 - loss: 7.7579 - accuracy: 0.4940
 2720/25000 [==>...........................] - ETA: 1:22 - loss: 7.7455 - accuracy: 0.4949
 2752/25000 [==>...........................] - ETA: 1:22 - loss: 7.7669 - accuracy: 0.4935
 2784/25000 [==>...........................] - ETA: 1:21 - loss: 7.7437 - accuracy: 0.4950
 2816/25000 [==>...........................] - ETA: 1:21 - loss: 7.7537 - accuracy: 0.4943
 2848/25000 [==>...........................] - ETA: 1:21 - loss: 7.7581 - accuracy: 0.4940
 2880/25000 [==>...........................] - ETA: 1:21 - loss: 7.7678 - accuracy: 0.4934
 2912/25000 [==>...........................] - ETA: 1:21 - loss: 7.7719 - accuracy: 0.4931
 2944/25000 [==>...........................] - ETA: 1:21 - loss: 7.7812 - accuracy: 0.4925
 2976/25000 [==>...........................] - ETA: 1:21 - loss: 7.7851 - accuracy: 0.4923
 3008/25000 [==>...........................] - ETA: 1:20 - loss: 7.7992 - accuracy: 0.4914
 3040/25000 [==>...........................] - ETA: 1:20 - loss: 7.7927 - accuracy: 0.4918
 3072/25000 [==>...........................] - ETA: 1:20 - loss: 7.8313 - accuracy: 0.4893
 3104/25000 [==>...........................] - ETA: 1:20 - loss: 7.8296 - accuracy: 0.4894
 3136/25000 [==>...........................] - ETA: 1:20 - loss: 7.8377 - accuracy: 0.4888
 3168/25000 [==>...........................] - ETA: 1:20 - loss: 7.8263 - accuracy: 0.4896
 3200/25000 [==>...........................] - ETA: 1:20 - loss: 7.8295 - accuracy: 0.4894
 3232/25000 [==>...........................] - ETA: 1:20 - loss: 7.8042 - accuracy: 0.4910
 3264/25000 [==>...........................] - ETA: 1:19 - loss: 7.8075 - accuracy: 0.4908
 3296/25000 [==>...........................] - ETA: 1:19 - loss: 7.8155 - accuracy: 0.4903
 3328/25000 [==>...........................] - ETA: 1:19 - loss: 7.7910 - accuracy: 0.4919
 3360/25000 [===>..........................] - ETA: 1:19 - loss: 7.7761 - accuracy: 0.4929
 3392/25000 [===>..........................] - ETA: 1:19 - loss: 7.7796 - accuracy: 0.4926
 3424/25000 [===>..........................] - ETA: 1:19 - loss: 7.7965 - accuracy: 0.4915
 3456/25000 [===>..........................] - ETA: 1:19 - loss: 7.8042 - accuracy: 0.4910
 3488/25000 [===>..........................] - ETA: 1:18 - loss: 7.7897 - accuracy: 0.4920
 3520/25000 [===>..........................] - ETA: 1:18 - loss: 7.7973 - accuracy: 0.4915
 3552/25000 [===>..........................] - ETA: 1:18 - loss: 7.8091 - accuracy: 0.4907
 3584/25000 [===>..........................] - ETA: 1:18 - loss: 7.8249 - accuracy: 0.4897
 3616/25000 [===>..........................] - ETA: 1:18 - loss: 7.8193 - accuracy: 0.4900
 3648/25000 [===>..........................] - ETA: 1:18 - loss: 7.8179 - accuracy: 0.4901
 3680/25000 [===>..........................] - ETA: 1:18 - loss: 7.8250 - accuracy: 0.4897
 3712/25000 [===>..........................] - ETA: 1:18 - loss: 7.8484 - accuracy: 0.4881
 3744/25000 [===>..........................] - ETA: 1:17 - loss: 7.8345 - accuracy: 0.4890
 3776/25000 [===>..........................] - ETA: 1:17 - loss: 7.8494 - accuracy: 0.4881
 3808/25000 [===>..........................] - ETA: 1:17 - loss: 7.8559 - accuracy: 0.4877
 3840/25000 [===>..........................] - ETA: 1:17 - loss: 7.8663 - accuracy: 0.4870
 3872/25000 [===>..........................] - ETA: 1:17 - loss: 7.8646 - accuracy: 0.4871
 3904/25000 [===>..........................] - ETA: 1:17 - loss: 7.8709 - accuracy: 0.4867
 3936/25000 [===>..........................] - ETA: 1:17 - loss: 7.8692 - accuracy: 0.4868
 3968/25000 [===>..........................] - ETA: 1:16 - loss: 7.8598 - accuracy: 0.4874
 4000/25000 [===>..........................] - ETA: 1:16 - loss: 7.8468 - accuracy: 0.4882
 4032/25000 [===>..........................] - ETA: 1:16 - loss: 7.8530 - accuracy: 0.4878
 4064/25000 [===>..........................] - ETA: 1:16 - loss: 7.8326 - accuracy: 0.4892
 4096/25000 [===>..........................] - ETA: 1:16 - loss: 7.8276 - accuracy: 0.4895
 4128/25000 [===>..........................] - ETA: 1:16 - loss: 7.8375 - accuracy: 0.4889
 4160/25000 [===>..........................] - ETA: 1:16 - loss: 7.8362 - accuracy: 0.4889
 4192/25000 [====>.........................] - ETA: 1:16 - loss: 7.8202 - accuracy: 0.4900
 4224/25000 [====>.........................] - ETA: 1:16 - loss: 7.8046 - accuracy: 0.4910
 4256/25000 [====>.........................] - ETA: 1:16 - loss: 7.8035 - accuracy: 0.4911
 4288/25000 [====>.........................] - ETA: 1:15 - loss: 7.7953 - accuracy: 0.4916
 4320/25000 [====>.........................] - ETA: 1:15 - loss: 7.8015 - accuracy: 0.4912
 4352/25000 [====>.........................] - ETA: 1:15 - loss: 7.7970 - accuracy: 0.4915
 4384/25000 [====>.........................] - ETA: 1:15 - loss: 7.7925 - accuracy: 0.4918
 4416/25000 [====>.........................] - ETA: 1:15 - loss: 7.7812 - accuracy: 0.4925
 4448/25000 [====>.........................] - ETA: 1:15 - loss: 7.7700 - accuracy: 0.4933
 4480/25000 [====>.........................] - ETA: 1:15 - loss: 7.7727 - accuracy: 0.4931
 4512/25000 [====>.........................] - ETA: 1:14 - loss: 7.7686 - accuracy: 0.4934
 4544/25000 [====>.........................] - ETA: 1:14 - loss: 7.7712 - accuracy: 0.4932
 4576/25000 [====>.........................] - ETA: 1:14 - loss: 7.7872 - accuracy: 0.4921
 4608/25000 [====>.........................] - ETA: 1:14 - loss: 7.7664 - accuracy: 0.4935
 4640/25000 [====>.........................] - ETA: 1:14 - loss: 7.7558 - accuracy: 0.4942
 4672/25000 [====>.........................] - ETA: 1:14 - loss: 7.7487 - accuracy: 0.4946
 4704/25000 [====>.........................] - ETA: 1:14 - loss: 7.7416 - accuracy: 0.4951
 4736/25000 [====>.........................] - ETA: 1:14 - loss: 7.7476 - accuracy: 0.4947
 4768/25000 [====>.........................] - ETA: 1:13 - loss: 7.7631 - accuracy: 0.4937
 4800/25000 [====>.........................] - ETA: 1:13 - loss: 7.7593 - accuracy: 0.4940
 4832/25000 [====>.........................] - ETA: 1:13 - loss: 7.7396 - accuracy: 0.4952
 4864/25000 [====>.........................] - ETA: 1:13 - loss: 7.7517 - accuracy: 0.4944
 4896/25000 [====>.........................] - ETA: 1:13 - loss: 7.7480 - accuracy: 0.4947
 4928/25000 [====>.........................] - ETA: 1:13 - loss: 7.7444 - accuracy: 0.4949
 4960/25000 [====>.........................] - ETA: 1:13 - loss: 7.7408 - accuracy: 0.4952
 4992/25000 [====>.........................] - ETA: 1:12 - loss: 7.7434 - accuracy: 0.4950
 5024/25000 [=====>........................] - ETA: 1:12 - loss: 7.7429 - accuracy: 0.4950
 5056/25000 [=====>........................] - ETA: 1:12 - loss: 7.7364 - accuracy: 0.4955
 5088/25000 [=====>........................] - ETA: 1:12 - loss: 7.7480 - accuracy: 0.4947
 5120/25000 [=====>........................] - ETA: 1:12 - loss: 7.7385 - accuracy: 0.4953
 5152/25000 [=====>........................] - ETA: 1:12 - loss: 7.7351 - accuracy: 0.4955
 5184/25000 [=====>........................] - ETA: 1:12 - loss: 7.7346 - accuracy: 0.4956
 5216/25000 [=====>........................] - ETA: 1:12 - loss: 7.7254 - accuracy: 0.4962
 5248/25000 [=====>........................] - ETA: 1:12 - loss: 7.7397 - accuracy: 0.4952
 5280/25000 [=====>........................] - ETA: 1:11 - loss: 7.7421 - accuracy: 0.4951
 5312/25000 [=====>........................] - ETA: 1:11 - loss: 7.7446 - accuracy: 0.4949
 5344/25000 [=====>........................] - ETA: 1:11 - loss: 7.7384 - accuracy: 0.4953
 5376/25000 [=====>........................] - ETA: 1:11 - loss: 7.7351 - accuracy: 0.4955
 5408/25000 [=====>........................] - ETA: 1:11 - loss: 7.7375 - accuracy: 0.4954
 5440/25000 [=====>........................] - ETA: 1:11 - loss: 7.7343 - accuracy: 0.4956
 5472/25000 [=====>........................] - ETA: 1:11 - loss: 7.7283 - accuracy: 0.4960
 5504/25000 [=====>........................] - ETA: 1:11 - loss: 7.7251 - accuracy: 0.4962
 5536/25000 [=====>........................] - ETA: 1:11 - loss: 7.7248 - accuracy: 0.4962
 5568/25000 [=====>........................] - ETA: 1:10 - loss: 7.7300 - accuracy: 0.4959
 5600/25000 [=====>........................] - ETA: 1:10 - loss: 7.7351 - accuracy: 0.4955
 5632/25000 [=====>........................] - ETA: 1:10 - loss: 7.7265 - accuracy: 0.4961
 5664/25000 [=====>........................] - ETA: 1:10 - loss: 7.7153 - accuracy: 0.4968
 5696/25000 [=====>........................] - ETA: 1:10 - loss: 7.7070 - accuracy: 0.4974
 5728/25000 [=====>........................] - ETA: 1:10 - loss: 7.6934 - accuracy: 0.4983
 5760/25000 [=====>........................] - ETA: 1:10 - loss: 7.6799 - accuracy: 0.4991
 5792/25000 [=====>........................] - ETA: 1:10 - loss: 7.6746 - accuracy: 0.4995
 5824/25000 [=====>........................] - ETA: 1:09 - loss: 7.6745 - accuracy: 0.4995
 5856/25000 [======>.......................] - ETA: 1:09 - loss: 7.6797 - accuracy: 0.4991
 5888/25000 [======>.......................] - ETA: 1:09 - loss: 7.6848 - accuracy: 0.4988
 5920/25000 [======>.......................] - ETA: 1:09 - loss: 7.6770 - accuracy: 0.4993
 5952/25000 [======>.......................] - ETA: 1:09 - loss: 7.6795 - accuracy: 0.4992
 5984/25000 [======>.......................] - ETA: 1:09 - loss: 7.6717 - accuracy: 0.4997
 6016/25000 [======>.......................] - ETA: 1:09 - loss: 7.6666 - accuracy: 0.5000
 6048/25000 [======>.......................] - ETA: 1:09 - loss: 7.6590 - accuracy: 0.5005
 6080/25000 [======>.......................] - ETA: 1:08 - loss: 7.6515 - accuracy: 0.5010
 6112/25000 [======>.......................] - ETA: 1:08 - loss: 7.6465 - accuracy: 0.5013
 6144/25000 [======>.......................] - ETA: 1:08 - loss: 7.6566 - accuracy: 0.5007
 6176/25000 [======>.......................] - ETA: 1:08 - loss: 7.6517 - accuracy: 0.5010
 6208/25000 [======>.......................] - ETA: 1:08 - loss: 7.6543 - accuracy: 0.5008
 6240/25000 [======>.......................] - ETA: 1:08 - loss: 7.6494 - accuracy: 0.5011
 6272/25000 [======>.......................] - ETA: 1:08 - loss: 7.6520 - accuracy: 0.5010
 6304/25000 [======>.......................] - ETA: 1:08 - loss: 7.6374 - accuracy: 0.5019
 6336/25000 [======>.......................] - ETA: 1:08 - loss: 7.6424 - accuracy: 0.5016
 6368/25000 [======>.......................] - ETA: 1:07 - loss: 7.6329 - accuracy: 0.5022
 6400/25000 [======>.......................] - ETA: 1:07 - loss: 7.6355 - accuracy: 0.5020
 6432/25000 [======>.......................] - ETA: 1:07 - loss: 7.6404 - accuracy: 0.5017
 6464/25000 [======>.......................] - ETA: 1:07 - loss: 7.6453 - accuracy: 0.5014
 6496/25000 [======>.......................] - ETA: 1:07 - loss: 7.6477 - accuracy: 0.5012
 6528/25000 [======>.......................] - ETA: 1:07 - loss: 7.6502 - accuracy: 0.5011
 6560/25000 [======>.......................] - ETA: 1:07 - loss: 7.6503 - accuracy: 0.5011
 6592/25000 [======>.......................] - ETA: 1:06 - loss: 7.6527 - accuracy: 0.5009
 6624/25000 [======>.......................] - ETA: 1:06 - loss: 7.6504 - accuracy: 0.5011
 6656/25000 [======>.......................] - ETA: 1:06 - loss: 7.6436 - accuracy: 0.5015
 6688/25000 [=======>......................] - ETA: 1:06 - loss: 7.6529 - accuracy: 0.5009
 6720/25000 [=======>......................] - ETA: 1:06 - loss: 7.6461 - accuracy: 0.5013
 6752/25000 [=======>......................] - ETA: 1:06 - loss: 7.6348 - accuracy: 0.5021
 6784/25000 [=======>......................] - ETA: 1:06 - loss: 7.6350 - accuracy: 0.5021
 6816/25000 [=======>......................] - ETA: 1:06 - loss: 7.6329 - accuracy: 0.5022
 6848/25000 [=======>......................] - ETA: 1:06 - loss: 7.6353 - accuracy: 0.5020
 6880/25000 [=======>......................] - ETA: 1:06 - loss: 7.6243 - accuracy: 0.5028
 6912/25000 [=======>......................] - ETA: 1:05 - loss: 7.6245 - accuracy: 0.5027
 6944/25000 [=======>......................] - ETA: 1:05 - loss: 7.6313 - accuracy: 0.5023
 6976/25000 [=======>......................] - ETA: 1:05 - loss: 7.6315 - accuracy: 0.5023
 7008/25000 [=======>......................] - ETA: 1:05 - loss: 7.6338 - accuracy: 0.5021
 7040/25000 [=======>......................] - ETA: 1:05 - loss: 7.6296 - accuracy: 0.5024
 7072/25000 [=======>......................] - ETA: 1:05 - loss: 7.6211 - accuracy: 0.5030
 7104/25000 [=======>......................] - ETA: 1:05 - loss: 7.6386 - accuracy: 0.5018
 7136/25000 [=======>......................] - ETA: 1:05 - loss: 7.6494 - accuracy: 0.5011
 7168/25000 [=======>......................] - ETA: 1:04 - loss: 7.6559 - accuracy: 0.5007
 7200/25000 [=======>......................] - ETA: 1:04 - loss: 7.6496 - accuracy: 0.5011
 7232/25000 [=======>......................] - ETA: 1:04 - loss: 7.6475 - accuracy: 0.5012
 7264/25000 [=======>......................] - ETA: 1:04 - loss: 7.6497 - accuracy: 0.5011
 7296/25000 [=======>......................] - ETA: 1:04 - loss: 7.6582 - accuracy: 0.5005
 7328/25000 [=======>......................] - ETA: 1:04 - loss: 7.6582 - accuracy: 0.5005
 7360/25000 [=======>......................] - ETA: 1:04 - loss: 7.6437 - accuracy: 0.5015
 7392/25000 [=======>......................] - ETA: 1:04 - loss: 7.6521 - accuracy: 0.5009
 7424/25000 [=======>......................] - ETA: 1:04 - loss: 7.6646 - accuracy: 0.5001
 7456/25000 [=======>......................] - ETA: 1:03 - loss: 7.6584 - accuracy: 0.5005
 7488/25000 [=======>......................] - ETA: 1:03 - loss: 7.6625 - accuracy: 0.5003
 7520/25000 [========>.....................] - ETA: 1:03 - loss: 7.6564 - accuracy: 0.5007
 7552/25000 [========>.....................] - ETA: 1:03 - loss: 7.6483 - accuracy: 0.5012
 7584/25000 [========>.....................] - ETA: 1:03 - loss: 7.6504 - accuracy: 0.5011
 7616/25000 [========>.....................] - ETA: 1:03 - loss: 7.6586 - accuracy: 0.5005
 7648/25000 [========>.....................] - ETA: 1:03 - loss: 7.6586 - accuracy: 0.5005
 7680/25000 [========>.....................] - ETA: 1:03 - loss: 7.6586 - accuracy: 0.5005
 7712/25000 [========>.....................] - ETA: 1:02 - loss: 7.6607 - accuracy: 0.5004
 7744/25000 [========>.....................] - ETA: 1:02 - loss: 7.6567 - accuracy: 0.5006
 7776/25000 [========>.....................] - ETA: 1:02 - loss: 7.6627 - accuracy: 0.5003
 7808/25000 [========>.....................] - ETA: 1:02 - loss: 7.6588 - accuracy: 0.5005
 7840/25000 [========>.....................] - ETA: 1:02 - loss: 7.6568 - accuracy: 0.5006
 7872/25000 [========>.....................] - ETA: 1:02 - loss: 7.6608 - accuracy: 0.5004
 7904/25000 [========>.....................] - ETA: 1:02 - loss: 7.6589 - accuracy: 0.5005
 7936/25000 [========>.....................] - ETA: 1:02 - loss: 7.6608 - accuracy: 0.5004
 7968/25000 [========>.....................] - ETA: 1:01 - loss: 7.6628 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 1:01 - loss: 7.6647 - accuracy: 0.5001
 8032/25000 [========>.....................] - ETA: 1:01 - loss: 7.6704 - accuracy: 0.4998
 8064/25000 [========>.....................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 8096/25000 [========>.....................] - ETA: 1:01 - loss: 7.6704 - accuracy: 0.4998
 8128/25000 [========>.....................] - ETA: 1:01 - loss: 7.6647 - accuracy: 0.5001
 8160/25000 [========>.....................] - ETA: 1:01 - loss: 7.6647 - accuracy: 0.5001
 8192/25000 [========>.....................] - ETA: 1:01 - loss: 7.6647 - accuracy: 0.5001
 8224/25000 [========>.....................] - ETA: 1:01 - loss: 7.6685 - accuracy: 0.4999
 8256/25000 [========>.....................] - ETA: 1:00 - loss: 7.6833 - accuracy: 0.4989
 8288/25000 [========>.....................] - ETA: 1:00 - loss: 7.6870 - accuracy: 0.4987
 8320/25000 [========>.....................] - ETA: 1:00 - loss: 7.6777 - accuracy: 0.4993
 8352/25000 [=========>....................] - ETA: 1:00 - loss: 7.6813 - accuracy: 0.4990
 8384/25000 [=========>....................] - ETA: 1:00 - loss: 7.6739 - accuracy: 0.4995
 8416/25000 [=========>....................] - ETA: 1:00 - loss: 7.6739 - accuracy: 0.4995
 8448/25000 [=========>....................] - ETA: 1:00 - loss: 7.6702 - accuracy: 0.4998
 8480/25000 [=========>....................] - ETA: 1:00 - loss: 7.6594 - accuracy: 0.5005
 8512/25000 [=========>....................] - ETA: 59s - loss: 7.6576 - accuracy: 0.5006 
 8544/25000 [=========>....................] - ETA: 59s - loss: 7.6523 - accuracy: 0.5009
 8576/25000 [=========>....................] - ETA: 59s - loss: 7.6541 - accuracy: 0.5008
 8608/25000 [=========>....................] - ETA: 59s - loss: 7.6488 - accuracy: 0.5012
 8640/25000 [=========>....................] - ETA: 59s - loss: 7.6471 - accuracy: 0.5013
 8672/25000 [=========>....................] - ETA: 59s - loss: 7.6383 - accuracy: 0.5018
 8704/25000 [=========>....................] - ETA: 59s - loss: 7.6455 - accuracy: 0.5014
 8736/25000 [=========>....................] - ETA: 59s - loss: 7.6420 - accuracy: 0.5016
 8768/25000 [=========>....................] - ETA: 59s - loss: 7.6421 - accuracy: 0.5016
 8800/25000 [=========>....................] - ETA: 58s - loss: 7.6475 - accuracy: 0.5013
 8832/25000 [=========>....................] - ETA: 58s - loss: 7.6475 - accuracy: 0.5012
 8864/25000 [=========>....................] - ETA: 58s - loss: 7.6528 - accuracy: 0.5009
 8896/25000 [=========>....................] - ETA: 58s - loss: 7.6563 - accuracy: 0.5007
 8928/25000 [=========>....................] - ETA: 58s - loss: 7.6597 - accuracy: 0.5004
 8960/25000 [=========>....................] - ETA: 58s - loss: 7.6632 - accuracy: 0.5002
 8992/25000 [=========>....................] - ETA: 58s - loss: 7.6717 - accuracy: 0.4997
 9024/25000 [=========>....................] - ETA: 58s - loss: 7.6683 - accuracy: 0.4999
 9056/25000 [=========>....................] - ETA: 57s - loss: 7.6683 - accuracy: 0.4999
 9088/25000 [=========>....................] - ETA: 57s - loss: 7.6700 - accuracy: 0.4998
 9120/25000 [=========>....................] - ETA: 57s - loss: 7.6649 - accuracy: 0.5001
 9152/25000 [=========>....................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 9184/25000 [==========>...................] - ETA: 57s - loss: 7.6616 - accuracy: 0.5003
 9216/25000 [==========>...................] - ETA: 57s - loss: 7.6600 - accuracy: 0.5004
 9248/25000 [==========>...................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 9280/25000 [==========>...................] - ETA: 57s - loss: 7.6650 - accuracy: 0.5001
 9312/25000 [==========>...................] - ETA: 57s - loss: 7.6699 - accuracy: 0.4998
 9344/25000 [==========>...................] - ETA: 56s - loss: 7.6732 - accuracy: 0.4996
 9376/25000 [==========>...................] - ETA: 56s - loss: 7.6764 - accuracy: 0.4994
 9408/25000 [==========>...................] - ETA: 56s - loss: 7.6797 - accuracy: 0.4991
 9440/25000 [==========>...................] - ETA: 56s - loss: 7.6877 - accuracy: 0.4986
 9472/25000 [==========>...................] - ETA: 56s - loss: 7.6941 - accuracy: 0.4982
 9504/25000 [==========>...................] - ETA: 56s - loss: 7.6940 - accuracy: 0.4982
 9536/25000 [==========>...................] - ETA: 56s - loss: 7.6907 - accuracy: 0.4984
 9568/25000 [==========>...................] - ETA: 56s - loss: 7.6826 - accuracy: 0.4990
 9600/25000 [==========>...................] - ETA: 55s - loss: 7.6810 - accuracy: 0.4991
 9632/25000 [==========>...................] - ETA: 55s - loss: 7.6825 - accuracy: 0.4990
 9664/25000 [==========>...................] - ETA: 55s - loss: 7.6793 - accuracy: 0.4992
 9696/25000 [==========>...................] - ETA: 55s - loss: 7.6761 - accuracy: 0.4994
 9728/25000 [==========>...................] - ETA: 55s - loss: 7.6761 - accuracy: 0.4994
 9760/25000 [==========>...................] - ETA: 55s - loss: 7.6792 - accuracy: 0.4992
 9792/25000 [==========>...................] - ETA: 55s - loss: 7.6760 - accuracy: 0.4994
 9824/25000 [==========>...................] - ETA: 55s - loss: 7.6807 - accuracy: 0.4991
 9856/25000 [==========>...................] - ETA: 55s - loss: 7.6713 - accuracy: 0.4997
 9888/25000 [==========>...................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 9920/25000 [==========>...................] - ETA: 54s - loss: 7.6651 - accuracy: 0.5001
 9952/25000 [==========>...................] - ETA: 54s - loss: 7.6697 - accuracy: 0.4998
 9984/25000 [==========>...................] - ETA: 54s - loss: 7.6697 - accuracy: 0.4998
10016/25000 [===========>..................] - ETA: 54s - loss: 7.6620 - accuracy: 0.5003
10048/25000 [===========>..................] - ETA: 54s - loss: 7.6590 - accuracy: 0.5005
10080/25000 [===========>..................] - ETA: 54s - loss: 7.6560 - accuracy: 0.5007
10112/25000 [===========>..................] - ETA: 54s - loss: 7.6560 - accuracy: 0.5007
10144/25000 [===========>..................] - ETA: 53s - loss: 7.6560 - accuracy: 0.5007
10176/25000 [===========>..................] - ETA: 53s - loss: 7.6531 - accuracy: 0.5009
10208/25000 [===========>..................] - ETA: 53s - loss: 7.6546 - accuracy: 0.5008
10240/25000 [===========>..................] - ETA: 53s - loss: 7.6591 - accuracy: 0.5005
10272/25000 [===========>..................] - ETA: 53s - loss: 7.6592 - accuracy: 0.5005
10304/25000 [===========>..................] - ETA: 53s - loss: 7.6607 - accuracy: 0.5004
10336/25000 [===========>..................] - ETA: 53s - loss: 7.6562 - accuracy: 0.5007
10368/25000 [===========>..................] - ETA: 53s - loss: 7.6592 - accuracy: 0.5005
10400/25000 [===========>..................] - ETA: 52s - loss: 7.6622 - accuracy: 0.5003
10432/25000 [===========>..................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
10464/25000 [===========>..................] - ETA: 52s - loss: 7.6681 - accuracy: 0.4999
10496/25000 [===========>..................] - ETA: 52s - loss: 7.6725 - accuracy: 0.4996
10528/25000 [===========>..................] - ETA: 52s - loss: 7.6710 - accuracy: 0.4997
10560/25000 [===========>..................] - ETA: 52s - loss: 7.6739 - accuracy: 0.4995
10592/25000 [===========>..................] - ETA: 52s - loss: 7.6768 - accuracy: 0.4993
10624/25000 [===========>..................] - ETA: 52s - loss: 7.6825 - accuracy: 0.4990
10656/25000 [===========>..................] - ETA: 52s - loss: 7.6796 - accuracy: 0.4992
10688/25000 [===========>..................] - ETA: 51s - loss: 7.6795 - accuracy: 0.4992
10720/25000 [===========>..................] - ETA: 51s - loss: 7.6824 - accuracy: 0.4990
10752/25000 [===========>..................] - ETA: 51s - loss: 7.6780 - accuracy: 0.4993
10784/25000 [===========>..................] - ETA: 51s - loss: 7.6723 - accuracy: 0.4996
10816/25000 [===========>..................] - ETA: 51s - loss: 7.6723 - accuracy: 0.4996
10848/25000 [============>.................] - ETA: 51s - loss: 7.6709 - accuracy: 0.4997
10880/25000 [============>.................] - ETA: 51s - loss: 7.6652 - accuracy: 0.5001
10912/25000 [============>.................] - ETA: 51s - loss: 7.6694 - accuracy: 0.4998
10944/25000 [============>.................] - ETA: 51s - loss: 7.6764 - accuracy: 0.4994
10976/25000 [============>.................] - ETA: 50s - loss: 7.6834 - accuracy: 0.4989
11008/25000 [============>.................] - ETA: 50s - loss: 7.6819 - accuracy: 0.4990
11040/25000 [============>.................] - ETA: 50s - loss: 7.6777 - accuracy: 0.4993
11072/25000 [============>.................] - ETA: 50s - loss: 7.6763 - accuracy: 0.4994
11104/25000 [============>.................] - ETA: 50s - loss: 7.6777 - accuracy: 0.4993
11136/25000 [============>.................] - ETA: 50s - loss: 7.6776 - accuracy: 0.4993
11168/25000 [============>.................] - ETA: 50s - loss: 7.6803 - accuracy: 0.4991
11200/25000 [============>.................] - ETA: 50s - loss: 7.6748 - accuracy: 0.4995
11232/25000 [============>.................] - ETA: 50s - loss: 7.6734 - accuracy: 0.4996
11264/25000 [============>.................] - ETA: 49s - loss: 7.6680 - accuracy: 0.4999
11296/25000 [============>.................] - ETA: 49s - loss: 7.6693 - accuracy: 0.4998
11328/25000 [============>.................] - ETA: 49s - loss: 7.6693 - accuracy: 0.4998
11360/25000 [============>.................] - ETA: 49s - loss: 7.6639 - accuracy: 0.5002
11392/25000 [============>.................] - ETA: 49s - loss: 7.6680 - accuracy: 0.4999
11424/25000 [============>.................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
11456/25000 [============>.................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
11488/25000 [============>.................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
11520/25000 [============>.................] - ETA: 48s - loss: 7.6733 - accuracy: 0.4996
11552/25000 [============>.................] - ETA: 48s - loss: 7.6746 - accuracy: 0.4995
11584/25000 [============>.................] - ETA: 48s - loss: 7.6719 - accuracy: 0.4997
11616/25000 [============>.................] - ETA: 48s - loss: 7.6785 - accuracy: 0.4992
11648/25000 [============>.................] - ETA: 48s - loss: 7.6785 - accuracy: 0.4992
11680/25000 [=============>................] - ETA: 48s - loss: 7.6811 - accuracy: 0.4991
11712/25000 [=============>................] - ETA: 48s - loss: 7.6771 - accuracy: 0.4993
11744/25000 [=============>................] - ETA: 48s - loss: 7.6705 - accuracy: 0.4997
11776/25000 [=============>................] - ETA: 48s - loss: 7.6757 - accuracy: 0.4994
11808/25000 [=============>................] - ETA: 47s - loss: 7.6783 - accuracy: 0.4992
11840/25000 [=============>................] - ETA: 47s - loss: 7.6783 - accuracy: 0.4992
11872/25000 [=============>................] - ETA: 47s - loss: 7.6834 - accuracy: 0.4989
11904/25000 [=============>................] - ETA: 47s - loss: 7.6872 - accuracy: 0.4987
11936/25000 [=============>................] - ETA: 47s - loss: 7.6897 - accuracy: 0.4985
11968/25000 [=============>................] - ETA: 47s - loss: 7.6871 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 47s - loss: 7.6896 - accuracy: 0.4985
12032/25000 [=============>................] - ETA: 47s - loss: 7.6934 - accuracy: 0.4983
12064/25000 [=============>................] - ETA: 46s - loss: 7.6908 - accuracy: 0.4984
12096/25000 [=============>................] - ETA: 46s - loss: 7.6907 - accuracy: 0.4984
12128/25000 [=============>................] - ETA: 46s - loss: 7.6932 - accuracy: 0.4983
12160/25000 [=============>................] - ETA: 46s - loss: 7.6969 - accuracy: 0.4980
12192/25000 [=============>................] - ETA: 46s - loss: 7.6943 - accuracy: 0.4982
12224/25000 [=============>................] - ETA: 46s - loss: 7.6930 - accuracy: 0.4983
12256/25000 [=============>................] - ETA: 46s - loss: 7.6916 - accuracy: 0.4984
12288/25000 [=============>................] - ETA: 46s - loss: 7.6891 - accuracy: 0.4985
12320/25000 [=============>................] - ETA: 46s - loss: 7.6828 - accuracy: 0.4989
12352/25000 [=============>................] - ETA: 45s - loss: 7.6753 - accuracy: 0.4994
12384/25000 [=============>................] - ETA: 45s - loss: 7.6740 - accuracy: 0.4995
12416/25000 [=============>................] - ETA: 45s - loss: 7.6777 - accuracy: 0.4993
12448/25000 [=============>................] - ETA: 45s - loss: 7.6777 - accuracy: 0.4993
12480/25000 [=============>................] - ETA: 45s - loss: 7.6789 - accuracy: 0.4992
12512/25000 [==============>...............] - ETA: 45s - loss: 7.6801 - accuracy: 0.4991
12544/25000 [==============>...............] - ETA: 45s - loss: 7.6862 - accuracy: 0.4987
12576/25000 [==============>...............] - ETA: 45s - loss: 7.6898 - accuracy: 0.4985
12608/25000 [==============>...............] - ETA: 44s - loss: 7.6849 - accuracy: 0.4988
12640/25000 [==============>...............] - ETA: 44s - loss: 7.6800 - accuracy: 0.4991
12672/25000 [==============>...............] - ETA: 44s - loss: 7.6715 - accuracy: 0.4997
12704/25000 [==============>...............] - ETA: 44s - loss: 7.6739 - accuracy: 0.4995
12736/25000 [==============>...............] - ETA: 44s - loss: 7.6738 - accuracy: 0.4995
12768/25000 [==============>...............] - ETA: 44s - loss: 7.6762 - accuracy: 0.4994
12800/25000 [==============>...............] - ETA: 44s - loss: 7.6774 - accuracy: 0.4993
12832/25000 [==============>...............] - ETA: 44s - loss: 7.6762 - accuracy: 0.4994
12864/25000 [==============>...............] - ETA: 44s - loss: 7.6762 - accuracy: 0.4994
12896/25000 [==============>...............] - ETA: 43s - loss: 7.6738 - accuracy: 0.4995
12928/25000 [==============>...............] - ETA: 43s - loss: 7.6714 - accuracy: 0.4997
12960/25000 [==============>...............] - ETA: 43s - loss: 7.6725 - accuracy: 0.4996
12992/25000 [==============>...............] - ETA: 43s - loss: 7.6713 - accuracy: 0.4997
13024/25000 [==============>...............] - ETA: 43s - loss: 7.6643 - accuracy: 0.5002
13056/25000 [==============>...............] - ETA: 43s - loss: 7.6643 - accuracy: 0.5002
13088/25000 [==============>...............] - ETA: 43s - loss: 7.6631 - accuracy: 0.5002
13120/25000 [==============>...............] - ETA: 43s - loss: 7.6631 - accuracy: 0.5002
13152/25000 [==============>...............] - ETA: 43s - loss: 7.6608 - accuracy: 0.5004
13184/25000 [==============>...............] - ETA: 42s - loss: 7.6620 - accuracy: 0.5003
13216/25000 [==============>...............] - ETA: 42s - loss: 7.6655 - accuracy: 0.5001
13248/25000 [==============>...............] - ETA: 42s - loss: 7.6736 - accuracy: 0.4995
13280/25000 [==============>...............] - ETA: 42s - loss: 7.6678 - accuracy: 0.4999
13312/25000 [==============>...............] - ETA: 42s - loss: 7.6701 - accuracy: 0.4998
13344/25000 [===============>..............] - ETA: 42s - loss: 7.6724 - accuracy: 0.4996
13376/25000 [===============>..............] - ETA: 42s - loss: 7.6735 - accuracy: 0.4996
13408/25000 [===============>..............] - ETA: 42s - loss: 7.6723 - accuracy: 0.4996
13440/25000 [===============>..............] - ETA: 41s - loss: 7.6735 - accuracy: 0.4996
13472/25000 [===============>..............] - ETA: 41s - loss: 7.6712 - accuracy: 0.4997
13504/25000 [===============>..............] - ETA: 41s - loss: 7.6700 - accuracy: 0.4998
13536/25000 [===============>..............] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
13568/25000 [===============>..............] - ETA: 41s - loss: 7.6711 - accuracy: 0.4997
13600/25000 [===============>..............] - ETA: 41s - loss: 7.6677 - accuracy: 0.4999
13632/25000 [===============>..............] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
13664/25000 [===============>..............] - ETA: 41s - loss: 7.6689 - accuracy: 0.4999
13696/25000 [===============>..............] - ETA: 41s - loss: 7.6722 - accuracy: 0.4996
13728/25000 [===============>..............] - ETA: 40s - loss: 7.6722 - accuracy: 0.4996
13760/25000 [===============>..............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
13792/25000 [===============>..............] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
13824/25000 [===============>..............] - ETA: 40s - loss: 7.6677 - accuracy: 0.4999
13856/25000 [===============>..............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
13888/25000 [===============>..............] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
13920/25000 [===============>..............] - ETA: 40s - loss: 7.6688 - accuracy: 0.4999
13952/25000 [===============>..............] - ETA: 40s - loss: 7.6699 - accuracy: 0.4998
13984/25000 [===============>..............] - ETA: 39s - loss: 7.6677 - accuracy: 0.4999
14016/25000 [===============>..............] - ETA: 39s - loss: 7.6655 - accuracy: 0.5001
14048/25000 [===============>..............] - ETA: 39s - loss: 7.6677 - accuracy: 0.4999
14080/25000 [===============>..............] - ETA: 39s - loss: 7.6677 - accuracy: 0.4999
14112/25000 [===============>..............] - ETA: 39s - loss: 7.6710 - accuracy: 0.4997
14144/25000 [===============>..............] - ETA: 39s - loss: 7.6688 - accuracy: 0.4999
14176/25000 [================>.............] - ETA: 39s - loss: 7.6720 - accuracy: 0.4996
14208/25000 [================>.............] - ETA: 39s - loss: 7.6709 - accuracy: 0.4997
14240/25000 [================>.............] - ETA: 39s - loss: 7.6677 - accuracy: 0.4999
14272/25000 [================>.............] - ETA: 38s - loss: 7.6655 - accuracy: 0.5001
14304/25000 [================>.............] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
14336/25000 [================>.............] - ETA: 38s - loss: 7.6623 - accuracy: 0.5003
14368/25000 [================>.............] - ETA: 38s - loss: 7.6602 - accuracy: 0.5004
14400/25000 [================>.............] - ETA: 38s - loss: 7.6602 - accuracy: 0.5004
14432/25000 [================>.............] - ETA: 38s - loss: 7.6613 - accuracy: 0.5003
14464/25000 [================>.............] - ETA: 38s - loss: 7.6603 - accuracy: 0.5004
14496/25000 [================>.............] - ETA: 38s - loss: 7.6592 - accuracy: 0.5005
14528/25000 [================>.............] - ETA: 37s - loss: 7.6582 - accuracy: 0.5006
14560/25000 [================>.............] - ETA: 37s - loss: 7.6571 - accuracy: 0.5006
14592/25000 [================>.............] - ETA: 37s - loss: 7.6572 - accuracy: 0.5006
14624/25000 [================>.............] - ETA: 37s - loss: 7.6572 - accuracy: 0.5006
14656/25000 [================>.............] - ETA: 37s - loss: 7.6582 - accuracy: 0.5005
14688/25000 [================>.............] - ETA: 37s - loss: 7.6562 - accuracy: 0.5007
14720/25000 [================>.............] - ETA: 37s - loss: 7.6531 - accuracy: 0.5009
14752/25000 [================>.............] - ETA: 37s - loss: 7.6552 - accuracy: 0.5007
14784/25000 [================>.............] - ETA: 37s - loss: 7.6583 - accuracy: 0.5005
14816/25000 [================>.............] - ETA: 36s - loss: 7.6625 - accuracy: 0.5003
14848/25000 [================>.............] - ETA: 36s - loss: 7.6625 - accuracy: 0.5003
14880/25000 [================>.............] - ETA: 36s - loss: 7.6646 - accuracy: 0.5001
14912/25000 [================>.............] - ETA: 36s - loss: 7.6625 - accuracy: 0.5003
14944/25000 [================>.............] - ETA: 36s - loss: 7.6635 - accuracy: 0.5002
14976/25000 [================>.............] - ETA: 36s - loss: 7.6646 - accuracy: 0.5001
15008/25000 [=================>............] - ETA: 36s - loss: 7.6636 - accuracy: 0.5002
15040/25000 [=================>............] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
15072/25000 [=================>............] - ETA: 35s - loss: 7.6656 - accuracy: 0.5001
15104/25000 [=================>............] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
15136/25000 [=================>............] - ETA: 35s - loss: 7.6646 - accuracy: 0.5001
15168/25000 [=================>............] - ETA: 35s - loss: 7.6646 - accuracy: 0.5001
15200/25000 [=================>............] - ETA: 35s - loss: 7.6636 - accuracy: 0.5002
15232/25000 [=================>............] - ETA: 35s - loss: 7.6626 - accuracy: 0.5003
15264/25000 [=================>............] - ETA: 35s - loss: 7.6596 - accuracy: 0.5005
15296/25000 [=================>............] - ETA: 35s - loss: 7.6656 - accuracy: 0.5001
15328/25000 [=================>............] - ETA: 35s - loss: 7.6706 - accuracy: 0.4997
15360/25000 [=================>............] - ETA: 34s - loss: 7.6656 - accuracy: 0.5001
15392/25000 [=================>............] - ETA: 34s - loss: 7.6636 - accuracy: 0.5002
15424/25000 [=================>............] - ETA: 34s - loss: 7.6676 - accuracy: 0.4999
15456/25000 [=================>............] - ETA: 34s - loss: 7.6676 - accuracy: 0.4999
15488/25000 [=================>............] - ETA: 34s - loss: 7.6676 - accuracy: 0.4999
15520/25000 [=================>............] - ETA: 34s - loss: 7.6637 - accuracy: 0.5002
15552/25000 [=================>............] - ETA: 34s - loss: 7.6637 - accuracy: 0.5002
15584/25000 [=================>............] - ETA: 34s - loss: 7.6637 - accuracy: 0.5002
15616/25000 [=================>............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
15648/25000 [=================>............] - ETA: 33s - loss: 7.6637 - accuracy: 0.5002
15680/25000 [=================>............] - ETA: 33s - loss: 7.6637 - accuracy: 0.5002
15712/25000 [=================>............] - ETA: 33s - loss: 7.6647 - accuracy: 0.5001
15744/25000 [=================>............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
15776/25000 [=================>............] - ETA: 33s - loss: 7.6656 - accuracy: 0.5001
15808/25000 [=================>............] - ETA: 33s - loss: 7.6676 - accuracy: 0.4999
15840/25000 [==================>...........] - ETA: 33s - loss: 7.6686 - accuracy: 0.4999
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6637 - accuracy: 0.5002
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6657 - accuracy: 0.5001
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6695 - accuracy: 0.4998
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6676 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6685 - accuracy: 0.4999
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6704 - accuracy: 0.4998
16064/25000 [==================>...........] - ETA: 32s - loss: 7.6704 - accuracy: 0.4998
16096/25000 [==================>...........] - ETA: 32s - loss: 7.6714 - accuracy: 0.4997
16128/25000 [==================>...........] - ETA: 32s - loss: 7.6704 - accuracy: 0.4998
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6676 - accuracy: 0.4999
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6695 - accuracy: 0.4998
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6695 - accuracy: 0.4998
16256/25000 [==================>...........] - ETA: 31s - loss: 7.6723 - accuracy: 0.4996
16288/25000 [==================>...........] - ETA: 31s - loss: 7.6704 - accuracy: 0.4998
16320/25000 [==================>...........] - ETA: 31s - loss: 7.6760 - accuracy: 0.4994
16352/25000 [==================>...........] - ETA: 31s - loss: 7.6722 - accuracy: 0.4996
16384/25000 [==================>...........] - ETA: 31s - loss: 7.6741 - accuracy: 0.4995
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6722 - accuracy: 0.4996
16448/25000 [==================>...........] - ETA: 30s - loss: 7.6731 - accuracy: 0.4996
16480/25000 [==================>...........] - ETA: 30s - loss: 7.6713 - accuracy: 0.4997
16512/25000 [==================>...........] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
16544/25000 [==================>...........] - ETA: 30s - loss: 7.6657 - accuracy: 0.5001
16576/25000 [==================>...........] - ETA: 30s - loss: 7.6629 - accuracy: 0.5002
16608/25000 [==================>...........] - ETA: 30s - loss: 7.6611 - accuracy: 0.5004
16640/25000 [==================>...........] - ETA: 30s - loss: 7.6639 - accuracy: 0.5002
16672/25000 [===================>..........] - ETA: 30s - loss: 7.6611 - accuracy: 0.5004
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6639 - accuracy: 0.5002
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6602 - accuracy: 0.5004
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6584 - accuracy: 0.5005
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6566 - accuracy: 0.5007
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6602 - accuracy: 0.5004
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6621 - accuracy: 0.5003
16896/25000 [===================>..........] - ETA: 29s - loss: 7.6675 - accuracy: 0.4999
16928/25000 [===================>..........] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6594 - accuracy: 0.5005
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6594 - accuracy: 0.5005
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6576 - accuracy: 0.5006
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6540 - accuracy: 0.5008
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6541 - accuracy: 0.5008
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6532 - accuracy: 0.5009
17152/25000 [===================>..........] - ETA: 28s - loss: 7.6514 - accuracy: 0.5010
17184/25000 [===================>..........] - ETA: 28s - loss: 7.6532 - accuracy: 0.5009
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6550 - accuracy: 0.5008
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6533 - accuracy: 0.5009
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6524 - accuracy: 0.5009
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6507 - accuracy: 0.5010
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6498 - accuracy: 0.5011
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6446 - accuracy: 0.5014
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6428 - accuracy: 0.5016
17440/25000 [===================>..........] - ETA: 27s - loss: 7.6455 - accuracy: 0.5014
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6508 - accuracy: 0.5010
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6535 - accuracy: 0.5009
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6570 - accuracy: 0.5006
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6579 - accuracy: 0.5006
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6588 - accuracy: 0.5005
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6579 - accuracy: 0.5006
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6631 - accuracy: 0.5002
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6588 - accuracy: 0.5005
17728/25000 [====================>.........] - ETA: 26s - loss: 7.6597 - accuracy: 0.5005
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6563 - accuracy: 0.5007
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6589 - accuracy: 0.5005
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6572 - accuracy: 0.5006
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6546 - accuracy: 0.5008
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6529 - accuracy: 0.5009
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6521 - accuracy: 0.5009
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6547 - accuracy: 0.5008
17984/25000 [====================>.........] - ETA: 25s - loss: 7.6607 - accuracy: 0.5004
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6624 - accuracy: 0.5003
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6607 - accuracy: 0.5004
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6590 - accuracy: 0.5005
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6556 - accuracy: 0.5007
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6582 - accuracy: 0.5006
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6565 - accuracy: 0.5007
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6574 - accuracy: 0.5006
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6582 - accuracy: 0.5005
18272/25000 [====================>.........] - ETA: 24s - loss: 7.6599 - accuracy: 0.5004
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6616 - accuracy: 0.5003
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6616 - accuracy: 0.5003
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6624 - accuracy: 0.5003
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6608 - accuracy: 0.5004
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6600 - accuracy: 0.5004
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6575 - accuracy: 0.5006
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6567 - accuracy: 0.5006
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6608 - accuracy: 0.5004
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6608 - accuracy: 0.5004
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6600 - accuracy: 0.5004
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6658 - accuracy: 0.5001
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6642 - accuracy: 0.5002
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6650 - accuracy: 0.5001
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6617 - accuracy: 0.5003
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6617 - accuracy: 0.5003
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6585 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6569 - accuracy: 0.5006
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6585 - accuracy: 0.5005
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6569 - accuracy: 0.5006
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6561 - accuracy: 0.5007
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6593 - accuracy: 0.5005
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6561 - accuracy: 0.5007
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6562 - accuracy: 0.5007
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6538 - accuracy: 0.5008
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6498 - accuracy: 0.5011
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6482 - accuracy: 0.5012
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6466 - accuracy: 0.5013
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6443 - accuracy: 0.5015
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6475 - accuracy: 0.5012
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6491 - accuracy: 0.5011
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6491 - accuracy: 0.5011
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6508 - accuracy: 0.5010
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6563 - accuracy: 0.5007
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6579 - accuracy: 0.5006
19424/25000 [======================>.......] - ETA: 20s - loss: 7.6603 - accuracy: 0.5004
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6611 - accuracy: 0.5004
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6611 - accuracy: 0.5004
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6572 - accuracy: 0.5006
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6572 - accuracy: 0.5006
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6572 - accuracy: 0.5006
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6572 - accuracy: 0.5006
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6565 - accuracy: 0.5007
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6612 - accuracy: 0.5004
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6581 - accuracy: 0.5006
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6542 - accuracy: 0.5008
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6565 - accuracy: 0.5007
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6542 - accuracy: 0.5008
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6527 - accuracy: 0.5009
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6566 - accuracy: 0.5007
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6559 - accuracy: 0.5007
19968/25000 [======================>.......] - ETA: 18s - loss: 7.6566 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6597 - accuracy: 0.5005
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6597 - accuracy: 0.5004
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6613 - accuracy: 0.5003
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6613 - accuracy: 0.5003
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6621 - accuracy: 0.5003
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6613 - accuracy: 0.5003
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6628 - accuracy: 0.5002
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6606 - accuracy: 0.5004
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6591 - accuracy: 0.5005
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6576 - accuracy: 0.5006
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6561 - accuracy: 0.5007
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6569 - accuracy: 0.5006
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6584 - accuracy: 0.5005
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6591 - accuracy: 0.5005
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6599 - accuracy: 0.5004
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6607 - accuracy: 0.5004
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6607 - accuracy: 0.5004
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6622 - accuracy: 0.5003
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6614 - accuracy: 0.5003
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6600 - accuracy: 0.5004
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6614 - accuracy: 0.5003
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6637 - accuracy: 0.5002
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6615 - accuracy: 0.5003
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6615 - accuracy: 0.5003
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6607 - accuracy: 0.5004
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6607 - accuracy: 0.5004
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6600 - accuracy: 0.5004
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6600 - accuracy: 0.5004
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6644 - accuracy: 0.5001
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6644 - accuracy: 0.5001
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6630 - accuracy: 0.5002
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6652 - accuracy: 0.5001
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6652 - accuracy: 0.5001
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6601 - accuracy: 0.5004
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6623 - accuracy: 0.5003
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6623 - accuracy: 0.5003
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6645 - accuracy: 0.5001
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6616 - accuracy: 0.5003
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6631 - accuracy: 0.5002
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6595 - accuracy: 0.5005
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6631 - accuracy: 0.5002
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6652 - accuracy: 0.5001
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6617 - accuracy: 0.5003
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6680 - accuracy: 0.4999
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6701 - accuracy: 0.4998
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6729 - accuracy: 0.4996
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6756 - accuracy: 0.4994
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6763 - accuracy: 0.4994
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6749 - accuracy: 0.4995
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6770 - accuracy: 0.4993 
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6763 - accuracy: 0.4994
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6762 - accuracy: 0.4994
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6824 - accuracy: 0.4990
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6810 - accuracy: 0.4991
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6823 - accuracy: 0.4990
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6822 - accuracy: 0.4990
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6815 - accuracy: 0.4990
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6795 - accuracy: 0.4992
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6781 - accuracy: 0.4993
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6754 - accuracy: 0.4994
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6720 - accuracy: 0.4996
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6706 - accuracy: 0.4997
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6686 - accuracy: 0.4999
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6713 - accuracy: 0.4997
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6706 - accuracy: 0.4997
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6693 - accuracy: 0.4998
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6706 - accuracy: 0.4997
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6713 - accuracy: 0.4997
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6719 - accuracy: 0.4997
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6733 - accuracy: 0.4996
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6719 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6686 - accuracy: 0.4999
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6686 - accuracy: 0.4999
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6679 - accuracy: 0.4999
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6719 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6679 - accuracy: 0.4999
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6698 - accuracy: 0.4998
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6742 - accuracy: 0.4995
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6723 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24288/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24352/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24384/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24416/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24448/25000 [============================>.] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
24480/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24512/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24832/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24864/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24896/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24928/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 108s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

