
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
 40%|████      | 2/5 [00:53<01:20, 26.85s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4251942720838592, 'embedding_size_factor': 0.6937722766575765, 'layers.choice': 2, 'learning_rate': 0.005520536397840177, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 5.110617749631381e-08} and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb6b\tB\xfd&X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe63a\xea\xe3\xb4DX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?v\x9c\xb3\xb4\x8f\xc6HX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>ko\xfa\xc3\xdb\x82hu.' and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb6b\tB\xfd&X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe63a\xea\xe3\xb4DX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?v\x9c\xb3\xb4\x8f\xc6HX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>ko\xfa\xc3\xdb\x82hu.' and reward: 0.3828
 60%|██████    | 3/5 [01:48<01:10, 35.12s/it] 60%|██████    | 3/5 [01:48<01:12, 36.04s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.402749300611096, 'embedding_size_factor': 1.0292853351042983, 'layers.choice': 2, 'learning_rate': 0.009685234716943884, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.0019307384070559803} and reward: 0.3604
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xc6\xa5\x00\xa7#OX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0w\xf3\xe6Hg\xe2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x83\xd5\xda2\xe8("X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?_\xa2\x1a\x940\x1c\xdfu.' and reward: 0.3604
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xc6\xa5\x00\xa7#OX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0w\xf3\xe6Hg\xe2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x83\xd5\xda2\xe8("X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?_\xa2\x1a\x940\x1c\xdfu.' and reward: 0.3604
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 163.86565780639648
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -46.56s of remaining time.
Ensemble size: 36
Ensemble weights: 
[0.61111111 0.25       0.13888889]
	0.3918	 = Validation accuracy score
	1.14s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 167.74s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f005b7a0b00> 

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
 [ 0.07942402  0.06717196 -0.09890649  0.04234299 -0.02882526 -0.01312891]
 [ 0.03966634 -0.0536695  -0.05102541  0.15221655  0.18283337 -0.11312602]
 [ 0.22584631 -0.00917186  0.1053516  -0.19885987  0.3065044   0.14157973]
 [ 0.15665577  0.004524    0.38366252  0.31271967 -0.19842473 -0.07647846]
 [-0.20276454 -0.11533626  0.13527362  0.22670142  0.33027402  0.25539336]
 [ 0.32784376  0.18693598  0.45738888 -0.03499339 -0.11044271  0.2216896 ]
 [ 0.08892661  0.25937837  0.66794759  0.8494451   0.3178739  -0.01363785]
 [-0.15299143 -0.13464801  0.29172984 -0.26484007  0.29469606  0.07128908]
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
{'loss': 0.6742635741829872, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 10:17:08.510052: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.553550012409687, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 10:17:09.700554: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3022848/17464789 [====>.........................] - ETA: 0s
11173888/17464789 [==================>...........] - ETA: 0s
16154624/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 10:17:21.798469: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 10:17:21.802963: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-13 10:17:21.803126: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5555e66f0730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 10:17:21.803140: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:41 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:51 - loss: 6.4687 - accuracy: 0.5781
   96/25000 [..............................] - ETA: 2:17 - loss: 6.7083 - accuracy: 0.5625
  128/25000 [..............................] - ETA: 1:59 - loss: 7.1875 - accuracy: 0.5312
  160/25000 [..............................] - ETA: 1:48 - loss: 7.1875 - accuracy: 0.5312
  192/25000 [..............................] - ETA: 1:39 - loss: 6.9479 - accuracy: 0.5469
  224/25000 [..............................] - ETA: 1:34 - loss: 6.9136 - accuracy: 0.5491
  256/25000 [..............................] - ETA: 1:31 - loss: 7.1276 - accuracy: 0.5352
  288/25000 [..............................] - ETA: 1:28 - loss: 6.9213 - accuracy: 0.5486
  320/25000 [..............................] - ETA: 1:26 - loss: 6.9479 - accuracy: 0.5469
  352/25000 [..............................] - ETA: 1:25 - loss: 6.9697 - accuracy: 0.5455
  384/25000 [..............................] - ETA: 1:23 - loss: 7.1076 - accuracy: 0.5365
  416/25000 [..............................] - ETA: 1:21 - loss: 7.1506 - accuracy: 0.5337
  448/25000 [..............................] - ETA: 1:20 - loss: 7.2559 - accuracy: 0.5268
  480/25000 [..............................] - ETA: 1:18 - loss: 7.1236 - accuracy: 0.5354
  512/25000 [..............................] - ETA: 1:17 - loss: 7.1575 - accuracy: 0.5332
  544/25000 [..............................] - ETA: 1:16 - loss: 7.1593 - accuracy: 0.5331
  576/25000 [..............................] - ETA: 1:15 - loss: 7.2407 - accuracy: 0.5278
  608/25000 [..............................] - ETA: 1:15 - loss: 7.2631 - accuracy: 0.5263
  640/25000 [..............................] - ETA: 1:14 - loss: 7.3072 - accuracy: 0.5234
  672/25000 [..............................] - ETA: 1:13 - loss: 7.4156 - accuracy: 0.5164
  704/25000 [..............................] - ETA: 1:12 - loss: 7.4706 - accuracy: 0.5128
  736/25000 [..............................] - ETA: 1:12 - loss: 7.5625 - accuracy: 0.5068
  768/25000 [..............................] - ETA: 1:12 - loss: 7.5468 - accuracy: 0.5078
  800/25000 [..............................] - ETA: 1:11 - loss: 7.4750 - accuracy: 0.5125
  832/25000 [..............................] - ETA: 1:11 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:11 - loss: 7.3472 - accuracy: 0.5208
  896/25000 [>.............................] - ETA: 1:11 - loss: 7.3586 - accuracy: 0.5201
  928/25000 [>.............................] - ETA: 1:10 - loss: 7.4353 - accuracy: 0.5151
  960/25000 [>.............................] - ETA: 1:10 - loss: 7.4750 - accuracy: 0.5125
  992/25000 [>.............................] - ETA: 1:10 - loss: 7.5584 - accuracy: 0.5071
 1024/25000 [>.............................] - ETA: 1:09 - loss: 7.5618 - accuracy: 0.5068
 1056/25000 [>.............................] - ETA: 1:09 - loss: 7.5505 - accuracy: 0.5076
 1088/25000 [>.............................] - ETA: 1:09 - loss: 7.5116 - accuracy: 0.5101
 1120/25000 [>.............................] - ETA: 1:09 - loss: 7.5160 - accuracy: 0.5098
 1152/25000 [>.............................] - ETA: 1:08 - loss: 7.4936 - accuracy: 0.5113
 1184/25000 [>.............................] - ETA: 1:08 - loss: 7.5371 - accuracy: 0.5084
 1216/25000 [>.............................] - ETA: 1:08 - loss: 7.5910 - accuracy: 0.5049
 1248/25000 [>.............................] - ETA: 1:08 - loss: 7.5315 - accuracy: 0.5088
 1280/25000 [>.............................] - ETA: 1:07 - loss: 7.5468 - accuracy: 0.5078
 1312/25000 [>.............................] - ETA: 1:07 - loss: 7.5381 - accuracy: 0.5084
 1344/25000 [>.............................] - ETA: 1:07 - loss: 7.5297 - accuracy: 0.5089
 1376/25000 [>.............................] - ETA: 1:07 - loss: 7.5440 - accuracy: 0.5080
 1408/25000 [>.............................] - ETA: 1:07 - loss: 7.5142 - accuracy: 0.5099
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.4963 - accuracy: 0.5111
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.5104 - accuracy: 0.5102
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.4525 - accuracy: 0.5140
 1536/25000 [>.............................] - ETA: 1:06 - loss: 7.4969 - accuracy: 0.5111
 1568/25000 [>.............................] - ETA: 1:06 - loss: 7.5395 - accuracy: 0.5083
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.5037 - accuracy: 0.5106
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.4693 - accuracy: 0.5129
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.4731 - accuracy: 0.5126
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.5129 - accuracy: 0.5100
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.4803 - accuracy: 0.5122
 1760/25000 [=>............................] - ETA: 1:04 - loss: 7.4488 - accuracy: 0.5142
 1792/25000 [=>............................] - ETA: 1:04 - loss: 7.4185 - accuracy: 0.5162
 1824/25000 [=>............................] - ETA: 1:04 - loss: 7.4481 - accuracy: 0.5143
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.4683 - accuracy: 0.5129
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.4555 - accuracy: 0.5138
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.4750 - accuracy: 0.5125
 1952/25000 [=>............................] - ETA: 1:03 - loss: 7.5095 - accuracy: 0.5102
 1984/25000 [=>............................] - ETA: 1:03 - loss: 7.5121 - accuracy: 0.5101
 2016/25000 [=>............................] - ETA: 1:04 - loss: 7.4917 - accuracy: 0.5114
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.4420 - accuracy: 0.5146
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.4455 - accuracy: 0.5144
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.4561 - accuracy: 0.5137
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.4235 - accuracy: 0.5159
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.4341 - accuracy: 0.5152
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.4513 - accuracy: 0.5140
 2240/25000 [=>............................] - ETA: 1:03 - loss: 7.4339 - accuracy: 0.5152
 2272/25000 [=>............................] - ETA: 1:03 - loss: 7.4574 - accuracy: 0.5136
 2304/25000 [=>............................] - ETA: 1:03 - loss: 7.4537 - accuracy: 0.5139
 2336/25000 [=>............................] - ETA: 1:03 - loss: 7.4894 - accuracy: 0.5116
 2368/25000 [=>............................] - ETA: 1:03 - loss: 7.4788 - accuracy: 0.5122
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.4877 - accuracy: 0.5117
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.5027 - accuracy: 0.5107
 2464/25000 [=>............................] - ETA: 1:02 - loss: 7.5235 - accuracy: 0.5093
 2496/25000 [=>............................] - ETA: 1:02 - loss: 7.5499 - accuracy: 0.5076
 2528/25000 [==>...........................] - ETA: 1:02 - loss: 7.5574 - accuracy: 0.5071
 2560/25000 [==>...........................] - ETA: 1:02 - loss: 7.5348 - accuracy: 0.5086
 2592/25000 [==>...........................] - ETA: 1:02 - loss: 7.5601 - accuracy: 0.5069
 2624/25000 [==>...........................] - ETA: 1:02 - loss: 7.5848 - accuracy: 0.5053
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.5858 - accuracy: 0.5053
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.5754 - accuracy: 0.5060
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.5539 - accuracy: 0.5074
 2752/25000 [==>...........................] - ETA: 1:01 - loss: 7.5552 - accuracy: 0.5073
 2784/25000 [==>...........................] - ETA: 1:01 - loss: 7.5730 - accuracy: 0.5061
 2816/25000 [==>...........................] - ETA: 1:01 - loss: 7.5686 - accuracy: 0.5064
 2848/25000 [==>...........................] - ETA: 1:01 - loss: 7.5805 - accuracy: 0.5056
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.5974 - accuracy: 0.5045
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.5876 - accuracy: 0.5052
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.5885 - accuracy: 0.5051
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.5996 - accuracy: 0.5044
 3008/25000 [==>...........................] - ETA: 1:00 - loss: 7.5953 - accuracy: 0.5047
 3040/25000 [==>...........................] - ETA: 1:00 - loss: 7.5910 - accuracy: 0.5049
 3072/25000 [==>...........................] - ETA: 1:00 - loss: 7.5668 - accuracy: 0.5065
 3104/25000 [==>...........................] - ETA: 1:00 - loss: 7.5481 - accuracy: 0.5077
 3136/25000 [==>...........................] - ETA: 1:00 - loss: 7.5493 - accuracy: 0.5077
 3168/25000 [==>...........................] - ETA: 1:00 - loss: 7.5359 - accuracy: 0.5085
 3200/25000 [==>...........................] - ETA: 1:00 - loss: 7.5229 - accuracy: 0.5094
 3232/25000 [==>...........................] - ETA: 1:00 - loss: 7.5195 - accuracy: 0.5096
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.5257 - accuracy: 0.5092 
 3296/25000 [==>...........................] - ETA: 59s - loss: 7.5317 - accuracy: 0.5088
 3328/25000 [==>...........................] - ETA: 59s - loss: 7.5468 - accuracy: 0.5078
 3360/25000 [===>..........................] - ETA: 59s - loss: 7.5434 - accuracy: 0.5080
 3392/25000 [===>..........................] - ETA: 59s - loss: 7.5355 - accuracy: 0.5085
 3424/25000 [===>..........................] - ETA: 59s - loss: 7.5278 - accuracy: 0.5091
 3456/25000 [===>..........................] - ETA: 59s - loss: 7.5202 - accuracy: 0.5095
 3488/25000 [===>..........................] - ETA: 59s - loss: 7.5259 - accuracy: 0.5092
 3520/25000 [===>..........................] - ETA: 59s - loss: 7.5142 - accuracy: 0.5099
 3552/25000 [===>..........................] - ETA: 59s - loss: 7.5198 - accuracy: 0.5096
 3584/25000 [===>..........................] - ETA: 59s - loss: 7.5212 - accuracy: 0.5095
 3616/25000 [===>..........................] - ETA: 58s - loss: 7.5224 - accuracy: 0.5094
 3648/25000 [===>..........................] - ETA: 58s - loss: 7.5027 - accuracy: 0.5107
 3680/25000 [===>..........................] - ETA: 58s - loss: 7.5083 - accuracy: 0.5103
 3712/25000 [===>..........................] - ETA: 58s - loss: 7.5097 - accuracy: 0.5102
 3744/25000 [===>..........................] - ETA: 58s - loss: 7.4987 - accuracy: 0.5110
 3776/25000 [===>..........................] - ETA: 58s - loss: 7.4879 - accuracy: 0.5117
 3808/25000 [===>..........................] - ETA: 58s - loss: 7.4774 - accuracy: 0.5123
 3840/25000 [===>..........................] - ETA: 58s - loss: 7.4789 - accuracy: 0.5122
 3872/25000 [===>..........................] - ETA: 58s - loss: 7.4765 - accuracy: 0.5124
 3904/25000 [===>..........................] - ETA: 58s - loss: 7.4702 - accuracy: 0.5128
 3936/25000 [===>..........................] - ETA: 58s - loss: 7.4602 - accuracy: 0.5135
 3968/25000 [===>..........................] - ETA: 57s - loss: 7.4541 - accuracy: 0.5139
 4000/25000 [===>..........................] - ETA: 57s - loss: 7.4558 - accuracy: 0.5138
 4032/25000 [===>..........................] - ETA: 57s - loss: 7.4461 - accuracy: 0.5144
 4064/25000 [===>..........................] - ETA: 57s - loss: 7.4553 - accuracy: 0.5138
 4096/25000 [===>..........................] - ETA: 57s - loss: 7.4757 - accuracy: 0.5125
 4128/25000 [===>..........................] - ETA: 57s - loss: 7.4958 - accuracy: 0.5111
 4160/25000 [===>..........................] - ETA: 57s - loss: 7.5044 - accuracy: 0.5106
 4192/25000 [====>.........................] - ETA: 57s - loss: 7.5130 - accuracy: 0.5100
 4224/25000 [====>.........................] - ETA: 57s - loss: 7.5214 - accuracy: 0.5095
 4256/25000 [====>.........................] - ETA: 57s - loss: 7.5333 - accuracy: 0.5087
 4288/25000 [====>.........................] - ETA: 57s - loss: 7.5486 - accuracy: 0.5077
 4320/25000 [====>.........................] - ETA: 57s - loss: 7.5424 - accuracy: 0.5081
 4352/25000 [====>.........................] - ETA: 57s - loss: 7.5398 - accuracy: 0.5083
 4384/25000 [====>.........................] - ETA: 57s - loss: 7.5407 - accuracy: 0.5082
 4416/25000 [====>.........................] - ETA: 56s - loss: 7.5416 - accuracy: 0.5082
 4448/25000 [====>.........................] - ETA: 56s - loss: 7.5494 - accuracy: 0.5076
 4480/25000 [====>.........................] - ETA: 56s - loss: 7.5366 - accuracy: 0.5085
 4512/25000 [====>.........................] - ETA: 56s - loss: 7.5273 - accuracy: 0.5091
 4544/25000 [====>.........................] - ETA: 56s - loss: 7.5215 - accuracy: 0.5095
 4576/25000 [====>.........................] - ETA: 56s - loss: 7.5259 - accuracy: 0.5092
 4608/25000 [====>.........................] - ETA: 56s - loss: 7.5335 - accuracy: 0.5087
 4640/25000 [====>.........................] - ETA: 56s - loss: 7.5444 - accuracy: 0.5080
 4672/25000 [====>.........................] - ETA: 56s - loss: 7.5583 - accuracy: 0.5071
 4704/25000 [====>.........................] - ETA: 55s - loss: 7.5656 - accuracy: 0.5066
 4736/25000 [====>.........................] - ETA: 55s - loss: 7.5695 - accuracy: 0.5063
 4768/25000 [====>.........................] - ETA: 55s - loss: 7.5734 - accuracy: 0.5061
 4800/25000 [====>.........................] - ETA: 55s - loss: 7.5708 - accuracy: 0.5063
 4832/25000 [====>.........................] - ETA: 55s - loss: 7.5619 - accuracy: 0.5068
 4864/25000 [====>.........................] - ETA: 55s - loss: 7.5563 - accuracy: 0.5072
 4896/25000 [====>.........................] - ETA: 55s - loss: 7.5633 - accuracy: 0.5067
 4928/25000 [====>.........................] - ETA: 55s - loss: 7.5795 - accuracy: 0.5057
 4960/25000 [====>.........................] - ETA: 55s - loss: 7.5801 - accuracy: 0.5056
 4992/25000 [====>.........................] - ETA: 55s - loss: 7.5929 - accuracy: 0.5048
 5024/25000 [=====>........................] - ETA: 55s - loss: 7.5873 - accuracy: 0.5052
 5056/25000 [=====>........................] - ETA: 55s - loss: 7.5969 - accuracy: 0.5045
 5088/25000 [=====>........................] - ETA: 54s - loss: 7.5853 - accuracy: 0.5053
 5120/25000 [=====>........................] - ETA: 54s - loss: 7.5918 - accuracy: 0.5049
 5152/25000 [=====>........................] - ETA: 54s - loss: 7.6011 - accuracy: 0.5043
 5184/25000 [=====>........................] - ETA: 54s - loss: 7.5986 - accuracy: 0.5044
 5216/25000 [=====>........................] - ETA: 54s - loss: 7.5961 - accuracy: 0.5046
 5248/25000 [=====>........................] - ETA: 54s - loss: 7.5936 - accuracy: 0.5048
 5280/25000 [=====>........................] - ETA: 54s - loss: 7.5766 - accuracy: 0.5059
 5312/25000 [=====>........................] - ETA: 54s - loss: 7.5743 - accuracy: 0.5060
 5344/25000 [=====>........................] - ETA: 54s - loss: 7.5891 - accuracy: 0.5051
 5376/25000 [=====>........................] - ETA: 54s - loss: 7.5839 - accuracy: 0.5054
 5408/25000 [=====>........................] - ETA: 54s - loss: 7.5872 - accuracy: 0.5052
 5440/25000 [=====>........................] - ETA: 54s - loss: 7.5764 - accuracy: 0.5059
 5472/25000 [=====>........................] - ETA: 54s - loss: 7.5713 - accuracy: 0.5062
 5504/25000 [=====>........................] - ETA: 53s - loss: 7.5691 - accuracy: 0.5064
 5536/25000 [=====>........................] - ETA: 53s - loss: 7.5641 - accuracy: 0.5067
 5568/25000 [=====>........................] - ETA: 53s - loss: 7.5757 - accuracy: 0.5059
 5600/25000 [=====>........................] - ETA: 53s - loss: 7.5735 - accuracy: 0.5061
 5632/25000 [=====>........................] - ETA: 53s - loss: 7.5795 - accuracy: 0.5057
 5664/25000 [=====>........................] - ETA: 53s - loss: 7.5827 - accuracy: 0.5055
 5696/25000 [=====>........................] - ETA: 53s - loss: 7.5805 - accuracy: 0.5056
 5728/25000 [=====>........................] - ETA: 53s - loss: 7.5783 - accuracy: 0.5058
 5760/25000 [=====>........................] - ETA: 53s - loss: 7.5814 - accuracy: 0.5056
 5792/25000 [=====>........................] - ETA: 52s - loss: 7.5766 - accuracy: 0.5059
 5824/25000 [=====>........................] - ETA: 52s - loss: 7.5797 - accuracy: 0.5057
 5856/25000 [======>.......................] - ETA: 52s - loss: 7.5802 - accuracy: 0.5056
 5888/25000 [======>.......................] - ETA: 52s - loss: 7.5885 - accuracy: 0.5051
 5920/25000 [======>.......................] - ETA: 52s - loss: 7.5863 - accuracy: 0.5052
 5952/25000 [======>.......................] - ETA: 52s - loss: 7.5919 - accuracy: 0.5049
 5984/25000 [======>.......................] - ETA: 52s - loss: 7.5846 - accuracy: 0.5053
 6016/25000 [======>.......................] - ETA: 52s - loss: 7.5749 - accuracy: 0.5060
 6048/25000 [======>.......................] - ETA: 52s - loss: 7.5728 - accuracy: 0.5061
 6080/25000 [======>.......................] - ETA: 52s - loss: 7.5683 - accuracy: 0.5064
 6112/25000 [======>.......................] - ETA: 52s - loss: 7.5738 - accuracy: 0.5061
 6144/25000 [======>.......................] - ETA: 51s - loss: 7.5718 - accuracy: 0.5062
 6176/25000 [======>.......................] - ETA: 51s - loss: 7.5673 - accuracy: 0.5065
 6208/25000 [======>.......................] - ETA: 51s - loss: 7.5752 - accuracy: 0.5060
 6240/25000 [======>.......................] - ETA: 51s - loss: 7.5904 - accuracy: 0.5050
 6272/25000 [======>.......................] - ETA: 51s - loss: 7.5933 - accuracy: 0.5048
 6304/25000 [======>.......................] - ETA: 51s - loss: 7.6107 - accuracy: 0.5036
 6336/25000 [======>.......................] - ETA: 51s - loss: 7.5989 - accuracy: 0.5044
 6368/25000 [======>.......................] - ETA: 51s - loss: 7.5896 - accuracy: 0.5050
 6400/25000 [======>.......................] - ETA: 51s - loss: 7.5852 - accuracy: 0.5053
 6432/25000 [======>.......................] - ETA: 51s - loss: 7.5999 - accuracy: 0.5044
 6464/25000 [======>.......................] - ETA: 51s - loss: 7.5931 - accuracy: 0.5048
 6496/25000 [======>.......................] - ETA: 50s - loss: 7.5934 - accuracy: 0.5048
 6528/25000 [======>.......................] - ETA: 50s - loss: 7.5868 - accuracy: 0.5052
 6560/25000 [======>.......................] - ETA: 50s - loss: 7.5848 - accuracy: 0.5053
 6592/25000 [======>.......................] - ETA: 50s - loss: 7.5899 - accuracy: 0.5050
 6624/25000 [======>.......................] - ETA: 50s - loss: 7.5925 - accuracy: 0.5048
 6656/25000 [======>.......................] - ETA: 50s - loss: 7.5837 - accuracy: 0.5054
 6688/25000 [=======>......................] - ETA: 50s - loss: 7.5887 - accuracy: 0.5051
 6720/25000 [=======>......................] - ETA: 50s - loss: 7.5799 - accuracy: 0.5057
 6752/25000 [=======>......................] - ETA: 50s - loss: 7.5871 - accuracy: 0.5052
 6784/25000 [=======>......................] - ETA: 50s - loss: 7.5875 - accuracy: 0.5052
 6816/25000 [=======>......................] - ETA: 50s - loss: 7.5946 - accuracy: 0.5047
 6848/25000 [=======>......................] - ETA: 50s - loss: 7.5905 - accuracy: 0.5050
 6880/25000 [=======>......................] - ETA: 50s - loss: 7.5931 - accuracy: 0.5048
 6912/25000 [=======>......................] - ETA: 49s - loss: 7.5934 - accuracy: 0.5048
 6944/25000 [=======>......................] - ETA: 49s - loss: 7.5938 - accuracy: 0.5048
 6976/25000 [=======>......................] - ETA: 49s - loss: 7.5831 - accuracy: 0.5054
 7008/25000 [=======>......................] - ETA: 49s - loss: 7.5813 - accuracy: 0.5056
 7040/25000 [=======>......................] - ETA: 49s - loss: 7.5839 - accuracy: 0.5054
 7072/25000 [=======>......................] - ETA: 49s - loss: 7.5886 - accuracy: 0.5051
 7104/25000 [=======>......................] - ETA: 49s - loss: 7.5846 - accuracy: 0.5053
 7136/25000 [=======>......................] - ETA: 49s - loss: 7.5957 - accuracy: 0.5046
 7168/25000 [=======>......................] - ETA: 49s - loss: 7.6003 - accuracy: 0.5043
 7200/25000 [=======>......................] - ETA: 49s - loss: 7.6091 - accuracy: 0.5038
 7232/25000 [=======>......................] - ETA: 49s - loss: 7.6073 - accuracy: 0.5039
 7264/25000 [=======>......................] - ETA: 49s - loss: 7.6138 - accuracy: 0.5034
 7296/25000 [=======>......................] - ETA: 48s - loss: 7.6099 - accuracy: 0.5037
 7328/25000 [=======>......................] - ETA: 48s - loss: 7.6018 - accuracy: 0.5042
 7360/25000 [=======>......................] - ETA: 48s - loss: 7.6000 - accuracy: 0.5043
 7392/25000 [=======>......................] - ETA: 48s - loss: 7.5982 - accuracy: 0.5045
 7424/25000 [=======>......................] - ETA: 48s - loss: 7.6067 - accuracy: 0.5039
 7456/25000 [=======>......................] - ETA: 48s - loss: 7.6008 - accuracy: 0.5043
 7488/25000 [=======>......................] - ETA: 48s - loss: 7.6031 - accuracy: 0.5041
 7520/25000 [========>.....................] - ETA: 48s - loss: 7.6054 - accuracy: 0.5040
 7552/25000 [========>.....................] - ETA: 48s - loss: 7.6057 - accuracy: 0.5040
 7584/25000 [========>.....................] - ETA: 48s - loss: 7.6080 - accuracy: 0.5038
 7616/25000 [========>.....................] - ETA: 48s - loss: 7.6062 - accuracy: 0.5039
 7648/25000 [========>.....................] - ETA: 47s - loss: 7.6105 - accuracy: 0.5037
 7680/25000 [========>.....................] - ETA: 47s - loss: 7.6167 - accuracy: 0.5033
 7712/25000 [========>.....................] - ETA: 47s - loss: 7.6090 - accuracy: 0.5038
 7744/25000 [========>.....................] - ETA: 47s - loss: 7.5993 - accuracy: 0.5044
 7776/25000 [========>.....................] - ETA: 47s - loss: 7.6035 - accuracy: 0.5041
 7808/25000 [========>.....................] - ETA: 47s - loss: 7.6097 - accuracy: 0.5037
 7840/25000 [========>.....................] - ETA: 47s - loss: 7.6099 - accuracy: 0.5037
 7872/25000 [========>.....................] - ETA: 47s - loss: 7.6101 - accuracy: 0.5037
 7904/25000 [========>.....................] - ETA: 47s - loss: 7.6181 - accuracy: 0.5032
 7936/25000 [========>.....................] - ETA: 47s - loss: 7.6241 - accuracy: 0.5028
 7968/25000 [========>.....................] - ETA: 47s - loss: 7.6281 - accuracy: 0.5025
 8000/25000 [========>.....................] - ETA: 46s - loss: 7.6206 - accuracy: 0.5030
 8032/25000 [========>.....................] - ETA: 46s - loss: 7.6189 - accuracy: 0.5031
 8064/25000 [========>.....................] - ETA: 46s - loss: 7.6210 - accuracy: 0.5030
 8096/25000 [========>.....................] - ETA: 46s - loss: 7.6287 - accuracy: 0.5025
 8128/25000 [========>.....................] - ETA: 46s - loss: 7.6232 - accuracy: 0.5028
 8160/25000 [========>.....................] - ETA: 46s - loss: 7.6290 - accuracy: 0.5025
 8192/25000 [========>.....................] - ETA: 46s - loss: 7.6273 - accuracy: 0.5026
 8224/25000 [========>.....................] - ETA: 46s - loss: 7.6181 - accuracy: 0.5032
 8256/25000 [========>.....................] - ETA: 46s - loss: 7.6220 - accuracy: 0.5029
 8288/25000 [========>.....................] - ETA: 46s - loss: 7.6296 - accuracy: 0.5024
 8320/25000 [========>.....................] - ETA: 46s - loss: 7.6316 - accuracy: 0.5023
 8352/25000 [=========>....................] - ETA: 46s - loss: 7.6354 - accuracy: 0.5020
 8384/25000 [=========>....................] - ETA: 45s - loss: 7.6410 - accuracy: 0.5017
 8416/25000 [=========>....................] - ETA: 45s - loss: 7.6320 - accuracy: 0.5023
 8448/25000 [=========>....................] - ETA: 45s - loss: 7.6303 - accuracy: 0.5024
 8480/25000 [=========>....................] - ETA: 45s - loss: 7.6341 - accuracy: 0.5021
 8512/25000 [=========>....................] - ETA: 45s - loss: 7.6306 - accuracy: 0.5023
 8544/25000 [=========>....................] - ETA: 45s - loss: 7.6200 - accuracy: 0.5030
 8576/25000 [=========>....................] - ETA: 45s - loss: 7.6237 - accuracy: 0.5028
 8608/25000 [=========>....................] - ETA: 45s - loss: 7.6167 - accuracy: 0.5033
 8640/25000 [=========>....................] - ETA: 45s - loss: 7.6152 - accuracy: 0.5034
 8672/25000 [=========>....................] - ETA: 45s - loss: 7.6189 - accuracy: 0.5031
 8704/25000 [=========>....................] - ETA: 45s - loss: 7.6120 - accuracy: 0.5036
 8736/25000 [=========>....................] - ETA: 45s - loss: 7.6122 - accuracy: 0.5035
 8768/25000 [=========>....................] - ETA: 44s - loss: 7.6177 - accuracy: 0.5032
 8800/25000 [=========>....................] - ETA: 44s - loss: 7.6196 - accuracy: 0.5031
 8832/25000 [=========>....................] - ETA: 44s - loss: 7.6215 - accuracy: 0.5029
 8864/25000 [=========>....................] - ETA: 44s - loss: 7.6113 - accuracy: 0.5036
 8896/25000 [=========>....................] - ETA: 44s - loss: 7.6080 - accuracy: 0.5038
 8928/25000 [=========>....................] - ETA: 44s - loss: 7.6082 - accuracy: 0.5038
 8960/25000 [=========>....................] - ETA: 44s - loss: 7.6084 - accuracy: 0.5038
 8992/25000 [=========>....................] - ETA: 44s - loss: 7.6001 - accuracy: 0.5043
 9024/25000 [=========>....................] - ETA: 44s - loss: 7.6038 - accuracy: 0.5041
 9056/25000 [=========>....................] - ETA: 44s - loss: 7.6006 - accuracy: 0.5043
 9088/25000 [=========>....................] - ETA: 44s - loss: 7.6025 - accuracy: 0.5042
 9120/25000 [=========>....................] - ETA: 43s - loss: 7.6061 - accuracy: 0.5039
 9152/25000 [=========>....................] - ETA: 43s - loss: 7.6097 - accuracy: 0.5037
 9184/25000 [==========>...................] - ETA: 43s - loss: 7.6115 - accuracy: 0.5036
 9216/25000 [==========>...................] - ETA: 43s - loss: 7.6134 - accuracy: 0.5035
 9248/25000 [==========>...................] - ETA: 43s - loss: 7.6152 - accuracy: 0.5034
 9280/25000 [==========>...................] - ETA: 43s - loss: 7.6171 - accuracy: 0.5032
 9312/25000 [==========>...................] - ETA: 43s - loss: 7.6172 - accuracy: 0.5032
 9344/25000 [==========>...................] - ETA: 43s - loss: 7.6125 - accuracy: 0.5035
 9376/25000 [==========>...................] - ETA: 43s - loss: 7.6143 - accuracy: 0.5034
 9408/25000 [==========>...................] - ETA: 43s - loss: 7.6145 - accuracy: 0.5034
 9440/25000 [==========>...................] - ETA: 43s - loss: 7.6195 - accuracy: 0.5031
 9472/25000 [==========>...................] - ETA: 43s - loss: 7.6213 - accuracy: 0.5030
 9504/25000 [==========>...................] - ETA: 42s - loss: 7.6198 - accuracy: 0.5031
 9536/25000 [==========>...................] - ETA: 42s - loss: 7.6152 - accuracy: 0.5034
 9568/25000 [==========>...................] - ETA: 42s - loss: 7.6169 - accuracy: 0.5032
 9600/25000 [==========>...................] - ETA: 42s - loss: 7.6171 - accuracy: 0.5032
 9632/25000 [==========>...................] - ETA: 42s - loss: 7.6205 - accuracy: 0.5030
 9664/25000 [==========>...................] - ETA: 42s - loss: 7.6190 - accuracy: 0.5031
 9696/25000 [==========>...................] - ETA: 42s - loss: 7.6192 - accuracy: 0.5031
 9728/25000 [==========>...................] - ETA: 42s - loss: 7.6241 - accuracy: 0.5028
 9760/25000 [==========>...................] - ETA: 42s - loss: 7.6195 - accuracy: 0.5031
 9792/25000 [==========>...................] - ETA: 42s - loss: 7.6212 - accuracy: 0.5030
 9824/25000 [==========>...................] - ETA: 42s - loss: 7.6214 - accuracy: 0.5030
 9856/25000 [==========>...................] - ETA: 42s - loss: 7.6246 - accuracy: 0.5027
 9888/25000 [==========>...................] - ETA: 41s - loss: 7.6154 - accuracy: 0.5033
 9920/25000 [==========>...................] - ETA: 41s - loss: 7.6202 - accuracy: 0.5030
 9952/25000 [==========>...................] - ETA: 41s - loss: 7.6173 - accuracy: 0.5032
 9984/25000 [==========>...................] - ETA: 41s - loss: 7.6175 - accuracy: 0.5032
10016/25000 [===========>..................] - ETA: 41s - loss: 7.6130 - accuracy: 0.5035
10048/25000 [===========>..................] - ETA: 41s - loss: 7.6147 - accuracy: 0.5034
10080/25000 [===========>..................] - ETA: 41s - loss: 7.6195 - accuracy: 0.5031
10112/25000 [===========>..................] - ETA: 41s - loss: 7.6272 - accuracy: 0.5026
10144/25000 [===========>..................] - ETA: 41s - loss: 7.6273 - accuracy: 0.5026
10176/25000 [===========>..................] - ETA: 41s - loss: 7.6244 - accuracy: 0.5028
10208/25000 [===========>..................] - ETA: 41s - loss: 7.6261 - accuracy: 0.5026
10240/25000 [===========>..................] - ETA: 40s - loss: 7.6217 - accuracy: 0.5029
10272/25000 [===========>..................] - ETA: 40s - loss: 7.6278 - accuracy: 0.5025
10304/25000 [===========>..................] - ETA: 40s - loss: 7.6294 - accuracy: 0.5024
10336/25000 [===========>..................] - ETA: 40s - loss: 7.6236 - accuracy: 0.5028
10368/25000 [===========>..................] - ETA: 40s - loss: 7.6134 - accuracy: 0.5035
10400/25000 [===========>..................] - ETA: 40s - loss: 7.6150 - accuracy: 0.5034
10432/25000 [===========>..................] - ETA: 40s - loss: 7.6122 - accuracy: 0.5035
10464/25000 [===========>..................] - ETA: 40s - loss: 7.6168 - accuracy: 0.5032
10496/25000 [===========>..................] - ETA: 40s - loss: 7.6155 - accuracy: 0.5033
10528/25000 [===========>..................] - ETA: 40s - loss: 7.6171 - accuracy: 0.5032
10560/25000 [===========>..................] - ETA: 40s - loss: 7.6187 - accuracy: 0.5031
10592/25000 [===========>..................] - ETA: 40s - loss: 7.6217 - accuracy: 0.5029
10624/25000 [===========>..................] - ETA: 39s - loss: 7.6219 - accuracy: 0.5029
10656/25000 [===========>..................] - ETA: 39s - loss: 7.6206 - accuracy: 0.5030
10688/25000 [===========>..................] - ETA: 39s - loss: 7.6193 - accuracy: 0.5031
10720/25000 [===========>..................] - ETA: 39s - loss: 7.6194 - accuracy: 0.5031
10752/25000 [===========>..................] - ETA: 39s - loss: 7.6210 - accuracy: 0.5030
10784/25000 [===========>..................] - ETA: 39s - loss: 7.6240 - accuracy: 0.5028
10816/25000 [===========>..................] - ETA: 39s - loss: 7.6241 - accuracy: 0.5028
10848/25000 [============>.................] - ETA: 39s - loss: 7.6228 - accuracy: 0.5029
10880/25000 [============>.................] - ETA: 39s - loss: 7.6272 - accuracy: 0.5026
10912/25000 [============>.................] - ETA: 39s - loss: 7.6287 - accuracy: 0.5025
10944/25000 [============>.................] - ETA: 39s - loss: 7.6302 - accuracy: 0.5024
10976/25000 [============>.................] - ETA: 39s - loss: 7.6317 - accuracy: 0.5023
11008/25000 [============>.................] - ETA: 38s - loss: 7.6290 - accuracy: 0.5025
11040/25000 [============>.................] - ETA: 38s - loss: 7.6277 - accuracy: 0.5025
11072/25000 [============>.................] - ETA: 38s - loss: 7.6265 - accuracy: 0.5026
11104/25000 [============>.................] - ETA: 38s - loss: 7.6252 - accuracy: 0.5027
11136/25000 [============>.................] - ETA: 38s - loss: 7.6239 - accuracy: 0.5028
11168/25000 [============>.................] - ETA: 38s - loss: 7.6282 - accuracy: 0.5025
11200/25000 [============>.................] - ETA: 38s - loss: 7.6297 - accuracy: 0.5024
11232/25000 [============>.................] - ETA: 38s - loss: 7.6311 - accuracy: 0.5023
11264/25000 [============>.................] - ETA: 38s - loss: 7.6339 - accuracy: 0.5021
11296/25000 [============>.................] - ETA: 38s - loss: 7.6300 - accuracy: 0.5024
11328/25000 [============>.................] - ETA: 38s - loss: 7.6274 - accuracy: 0.5026
11360/25000 [============>.................] - ETA: 38s - loss: 7.6288 - accuracy: 0.5025
11392/25000 [============>.................] - ETA: 37s - loss: 7.6276 - accuracy: 0.5025
11424/25000 [============>.................] - ETA: 37s - loss: 7.6277 - accuracy: 0.5025
11456/25000 [============>.................] - ETA: 37s - loss: 7.6305 - accuracy: 0.5024
11488/25000 [============>.................] - ETA: 37s - loss: 7.6279 - accuracy: 0.5025
11520/25000 [============>.................] - ETA: 37s - loss: 7.6307 - accuracy: 0.5023
11552/25000 [============>.................] - ETA: 37s - loss: 7.6295 - accuracy: 0.5024
11584/25000 [============>.................] - ETA: 37s - loss: 7.6296 - accuracy: 0.5024
11616/25000 [============>.................] - ETA: 37s - loss: 7.6231 - accuracy: 0.5028
11648/25000 [============>.................] - ETA: 37s - loss: 7.6205 - accuracy: 0.5030
11680/25000 [=============>................] - ETA: 37s - loss: 7.6207 - accuracy: 0.5030
11712/25000 [=============>................] - ETA: 37s - loss: 7.6234 - accuracy: 0.5028
11744/25000 [=============>................] - ETA: 36s - loss: 7.6248 - accuracy: 0.5027
11776/25000 [=============>................] - ETA: 36s - loss: 7.6276 - accuracy: 0.5025
11808/25000 [=============>................] - ETA: 36s - loss: 7.6303 - accuracy: 0.5024
11840/25000 [=============>................] - ETA: 36s - loss: 7.6304 - accuracy: 0.5024
11872/25000 [=============>................] - ETA: 36s - loss: 7.6279 - accuracy: 0.5025
11904/25000 [=============>................] - ETA: 36s - loss: 7.6215 - accuracy: 0.5029
11936/25000 [=============>................] - ETA: 36s - loss: 7.6204 - accuracy: 0.5030
11968/25000 [=============>................] - ETA: 36s - loss: 7.6205 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 36s - loss: 7.6193 - accuracy: 0.5031
12032/25000 [=============>................] - ETA: 36s - loss: 7.6220 - accuracy: 0.5029
12064/25000 [=============>................] - ETA: 36s - loss: 7.6247 - accuracy: 0.5027
12096/25000 [=============>................] - ETA: 35s - loss: 7.6261 - accuracy: 0.5026
12128/25000 [=============>................] - ETA: 35s - loss: 7.6249 - accuracy: 0.5027
12160/25000 [=============>................] - ETA: 35s - loss: 7.6263 - accuracy: 0.5026
12192/25000 [=============>................] - ETA: 35s - loss: 7.6301 - accuracy: 0.5024
12224/25000 [=============>................] - ETA: 35s - loss: 7.6365 - accuracy: 0.5020
12256/25000 [=============>................] - ETA: 35s - loss: 7.6378 - accuracy: 0.5019
12288/25000 [=============>................] - ETA: 35s - loss: 7.6417 - accuracy: 0.5016
12320/25000 [=============>................] - ETA: 35s - loss: 7.6442 - accuracy: 0.5015
12352/25000 [=============>................] - ETA: 35s - loss: 7.6418 - accuracy: 0.5016
12384/25000 [=============>................] - ETA: 35s - loss: 7.6419 - accuracy: 0.5016
12416/25000 [=============>................] - ETA: 35s - loss: 7.6469 - accuracy: 0.5013
12448/25000 [=============>................] - ETA: 34s - loss: 7.6432 - accuracy: 0.5015
12480/25000 [=============>................] - ETA: 34s - loss: 7.6433 - accuracy: 0.5015
12512/25000 [==============>...............] - ETA: 34s - loss: 7.6421 - accuracy: 0.5016
12544/25000 [==============>...............] - ETA: 34s - loss: 7.6385 - accuracy: 0.5018
12576/25000 [==============>...............] - ETA: 34s - loss: 7.6398 - accuracy: 0.5017
12608/25000 [==============>...............] - ETA: 34s - loss: 7.6484 - accuracy: 0.5012
12640/25000 [==============>...............] - ETA: 34s - loss: 7.6533 - accuracy: 0.5009
12672/25000 [==============>...............] - ETA: 34s - loss: 7.6521 - accuracy: 0.5009
12704/25000 [==============>...............] - ETA: 34s - loss: 7.6509 - accuracy: 0.5010
12736/25000 [==============>...............] - ETA: 34s - loss: 7.6534 - accuracy: 0.5009
12768/25000 [==============>...............] - ETA: 34s - loss: 7.6510 - accuracy: 0.5010
12800/25000 [==============>...............] - ETA: 33s - loss: 7.6546 - accuracy: 0.5008
12832/25000 [==============>...............] - ETA: 33s - loss: 7.6559 - accuracy: 0.5007
12864/25000 [==============>...............] - ETA: 33s - loss: 7.6559 - accuracy: 0.5007
12896/25000 [==============>...............] - ETA: 33s - loss: 7.6571 - accuracy: 0.5006
12928/25000 [==============>...............] - ETA: 33s - loss: 7.6536 - accuracy: 0.5009
12960/25000 [==============>...............] - ETA: 33s - loss: 7.6560 - accuracy: 0.5007
12992/25000 [==============>...............] - ETA: 33s - loss: 7.6560 - accuracy: 0.5007
13024/25000 [==============>...............] - ETA: 33s - loss: 7.6560 - accuracy: 0.5007
13056/25000 [==============>...............] - ETA: 33s - loss: 7.6560 - accuracy: 0.5007
13088/25000 [==============>...............] - ETA: 33s - loss: 7.6514 - accuracy: 0.5010
13120/25000 [==============>...............] - ETA: 33s - loss: 7.6538 - accuracy: 0.5008
13152/25000 [==============>...............] - ETA: 32s - loss: 7.6573 - accuracy: 0.5006
13184/25000 [==============>...............] - ETA: 32s - loss: 7.6620 - accuracy: 0.5003
13216/25000 [==============>...............] - ETA: 32s - loss: 7.6608 - accuracy: 0.5004
13248/25000 [==============>...............] - ETA: 32s - loss: 7.6608 - accuracy: 0.5004
13280/25000 [==============>...............] - ETA: 32s - loss: 7.6620 - accuracy: 0.5003
13312/25000 [==============>...............] - ETA: 32s - loss: 7.6620 - accuracy: 0.5003
13344/25000 [===============>..............] - ETA: 32s - loss: 7.6632 - accuracy: 0.5002
13376/25000 [===============>..............] - ETA: 32s - loss: 7.6609 - accuracy: 0.5004
13408/25000 [===============>..............] - ETA: 32s - loss: 7.6598 - accuracy: 0.5004
13440/25000 [===============>..............] - ETA: 32s - loss: 7.6643 - accuracy: 0.5001
13472/25000 [===============>..............] - ETA: 32s - loss: 7.6655 - accuracy: 0.5001
13504/25000 [===============>..............] - ETA: 31s - loss: 7.6678 - accuracy: 0.4999
13536/25000 [===============>..............] - ETA: 31s - loss: 7.6700 - accuracy: 0.4998
13568/25000 [===============>..............] - ETA: 31s - loss: 7.6677 - accuracy: 0.4999
13600/25000 [===============>..............] - ETA: 31s - loss: 7.6677 - accuracy: 0.4999
13632/25000 [===============>..............] - ETA: 31s - loss: 7.6655 - accuracy: 0.5001
13664/25000 [===============>..............] - ETA: 31s - loss: 7.6621 - accuracy: 0.5003
13696/25000 [===============>..............] - ETA: 31s - loss: 7.6621 - accuracy: 0.5003
13728/25000 [===============>..............] - ETA: 31s - loss: 7.6622 - accuracy: 0.5003
13760/25000 [===============>..............] - ETA: 31s - loss: 7.6577 - accuracy: 0.5006
13792/25000 [===============>..............] - ETA: 31s - loss: 7.6577 - accuracy: 0.5006
13824/25000 [===============>..............] - ETA: 31s - loss: 7.6611 - accuracy: 0.5004
13856/25000 [===============>..............] - ETA: 30s - loss: 7.6633 - accuracy: 0.5002
13888/25000 [===============>..............] - ETA: 30s - loss: 7.6611 - accuracy: 0.5004
13920/25000 [===============>..............] - ETA: 30s - loss: 7.6600 - accuracy: 0.5004
13952/25000 [===============>..............] - ETA: 30s - loss: 7.6644 - accuracy: 0.5001
13984/25000 [===============>..............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
14016/25000 [===============>..............] - ETA: 30s - loss: 7.6699 - accuracy: 0.4998
14048/25000 [===============>..............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
14080/25000 [===============>..............] - ETA: 30s - loss: 7.6623 - accuracy: 0.5003
14112/25000 [===============>..............] - ETA: 30s - loss: 7.6601 - accuracy: 0.5004
14144/25000 [===============>..............] - ETA: 30s - loss: 7.6612 - accuracy: 0.5004
14176/25000 [================>.............] - ETA: 30s - loss: 7.6634 - accuracy: 0.5002
14208/25000 [================>.............] - ETA: 29s - loss: 7.6612 - accuracy: 0.5004
14240/25000 [================>.............] - ETA: 29s - loss: 7.6612 - accuracy: 0.5004
14272/25000 [================>.............] - ETA: 29s - loss: 7.6591 - accuracy: 0.5005
14304/25000 [================>.............] - ETA: 29s - loss: 7.6623 - accuracy: 0.5003
14336/25000 [================>.............] - ETA: 29s - loss: 7.6613 - accuracy: 0.5003
14368/25000 [================>.............] - ETA: 29s - loss: 7.6624 - accuracy: 0.5003
14400/25000 [================>.............] - ETA: 29s - loss: 7.6592 - accuracy: 0.5005
14432/25000 [================>.............] - ETA: 29s - loss: 7.6624 - accuracy: 0.5003
14464/25000 [================>.............] - ETA: 29s - loss: 7.6634 - accuracy: 0.5002
14496/25000 [================>.............] - ETA: 29s - loss: 7.6634 - accuracy: 0.5002
14528/25000 [================>.............] - ETA: 29s - loss: 7.6624 - accuracy: 0.5003
14560/25000 [================>.............] - ETA: 28s - loss: 7.6635 - accuracy: 0.5002
14592/25000 [================>.............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
14624/25000 [================>.............] - ETA: 28s - loss: 7.6708 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 28s - loss: 7.6719 - accuracy: 0.4997
14688/25000 [================>.............] - ETA: 28s - loss: 7.6708 - accuracy: 0.4997
14720/25000 [================>.............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
14752/25000 [================>.............] - ETA: 28s - loss: 7.6697 - accuracy: 0.4998
14784/25000 [================>.............] - ETA: 28s - loss: 7.6656 - accuracy: 0.5001
14816/25000 [================>.............] - ETA: 28s - loss: 7.6645 - accuracy: 0.5001
14848/25000 [================>.............] - ETA: 28s - loss: 7.6604 - accuracy: 0.5004
14880/25000 [================>.............] - ETA: 28s - loss: 7.6563 - accuracy: 0.5007
14912/25000 [================>.............] - ETA: 27s - loss: 7.6553 - accuracy: 0.5007
14944/25000 [================>.............] - ETA: 27s - loss: 7.6543 - accuracy: 0.5008
14976/25000 [================>.............] - ETA: 27s - loss: 7.6584 - accuracy: 0.5005
15008/25000 [=================>............] - ETA: 27s - loss: 7.6615 - accuracy: 0.5003
15040/25000 [=================>............] - ETA: 27s - loss: 7.6574 - accuracy: 0.5006
15072/25000 [=================>............] - ETA: 27s - loss: 7.6585 - accuracy: 0.5005
15104/25000 [=================>............] - ETA: 27s - loss: 7.6585 - accuracy: 0.5005
15136/25000 [=================>............] - ETA: 27s - loss: 7.6595 - accuracy: 0.5005
15168/25000 [=================>............] - ETA: 27s - loss: 7.6585 - accuracy: 0.5005
15200/25000 [=================>............] - ETA: 27s - loss: 7.6596 - accuracy: 0.5005
15232/25000 [=================>............] - ETA: 27s - loss: 7.6626 - accuracy: 0.5003
15264/25000 [=================>............] - ETA: 26s - loss: 7.6636 - accuracy: 0.5002
15296/25000 [=================>............] - ETA: 26s - loss: 7.6646 - accuracy: 0.5001
15328/25000 [=================>............] - ETA: 26s - loss: 7.6616 - accuracy: 0.5003
15360/25000 [=================>............] - ETA: 26s - loss: 7.6616 - accuracy: 0.5003
15392/25000 [=================>............] - ETA: 26s - loss: 7.6587 - accuracy: 0.5005
15424/25000 [=================>............] - ETA: 26s - loss: 7.6616 - accuracy: 0.5003
15456/25000 [=================>............] - ETA: 26s - loss: 7.6577 - accuracy: 0.5006
15488/25000 [=================>............] - ETA: 26s - loss: 7.6627 - accuracy: 0.5003
15520/25000 [=================>............] - ETA: 26s - loss: 7.6637 - accuracy: 0.5002
15552/25000 [=================>............] - ETA: 26s - loss: 7.6637 - accuracy: 0.5002
15584/25000 [=================>............] - ETA: 26s - loss: 7.6627 - accuracy: 0.5003
15616/25000 [=================>............] - ETA: 25s - loss: 7.6656 - accuracy: 0.5001
15648/25000 [=================>............] - ETA: 25s - loss: 7.6676 - accuracy: 0.4999
15680/25000 [=================>............] - ETA: 25s - loss: 7.6696 - accuracy: 0.4998
15712/25000 [=================>............] - ETA: 25s - loss: 7.6686 - accuracy: 0.4999
15744/25000 [=================>............] - ETA: 25s - loss: 7.6647 - accuracy: 0.5001
15776/25000 [=================>............] - ETA: 25s - loss: 7.6598 - accuracy: 0.5004
15808/25000 [=================>............] - ETA: 25s - loss: 7.6579 - accuracy: 0.5006
15840/25000 [==================>...........] - ETA: 25s - loss: 7.6569 - accuracy: 0.5006
15872/25000 [==================>...........] - ETA: 25s - loss: 7.6618 - accuracy: 0.5003
15904/25000 [==================>...........] - ETA: 25s - loss: 7.6579 - accuracy: 0.5006
15936/25000 [==================>...........] - ETA: 25s - loss: 7.6551 - accuracy: 0.5008
15968/25000 [==================>...........] - ETA: 24s - loss: 7.6541 - accuracy: 0.5008
16000/25000 [==================>...........] - ETA: 24s - loss: 7.6513 - accuracy: 0.5010
16032/25000 [==================>...........] - ETA: 24s - loss: 7.6532 - accuracy: 0.5009
16064/25000 [==================>...........] - ETA: 24s - loss: 7.6542 - accuracy: 0.5008
16096/25000 [==================>...........] - ETA: 24s - loss: 7.6561 - accuracy: 0.5007
16128/25000 [==================>...........] - ETA: 24s - loss: 7.6581 - accuracy: 0.5006
16160/25000 [==================>...........] - ETA: 24s - loss: 7.6600 - accuracy: 0.5004
16192/25000 [==================>...........] - ETA: 24s - loss: 7.6609 - accuracy: 0.5004
16224/25000 [==================>...........] - ETA: 24s - loss: 7.6638 - accuracy: 0.5002
16256/25000 [==================>...........] - ETA: 24s - loss: 7.6619 - accuracy: 0.5003
16288/25000 [==================>...........] - ETA: 24s - loss: 7.6647 - accuracy: 0.5001
16320/25000 [==================>...........] - ETA: 23s - loss: 7.6647 - accuracy: 0.5001
16352/25000 [==================>...........] - ETA: 23s - loss: 7.6647 - accuracy: 0.5001
16384/25000 [==================>...........] - ETA: 23s - loss: 7.6629 - accuracy: 0.5002
16416/25000 [==================>...........] - ETA: 23s - loss: 7.6619 - accuracy: 0.5003
16448/25000 [==================>...........] - ETA: 23s - loss: 7.6610 - accuracy: 0.5004
16480/25000 [==================>...........] - ETA: 23s - loss: 7.6610 - accuracy: 0.5004
16512/25000 [==================>...........] - ETA: 23s - loss: 7.6545 - accuracy: 0.5008
16544/25000 [==================>...........] - ETA: 23s - loss: 7.6592 - accuracy: 0.5005
16576/25000 [==================>...........] - ETA: 23s - loss: 7.6601 - accuracy: 0.5004
16608/25000 [==================>...........] - ETA: 23s - loss: 7.6583 - accuracy: 0.5005
16640/25000 [==================>...........] - ETA: 23s - loss: 7.6574 - accuracy: 0.5006
16672/25000 [===================>..........] - ETA: 22s - loss: 7.6574 - accuracy: 0.5006
16704/25000 [===================>..........] - ETA: 22s - loss: 7.6593 - accuracy: 0.5005
16736/25000 [===================>..........] - ETA: 22s - loss: 7.6575 - accuracy: 0.5006
16768/25000 [===================>..........] - ETA: 22s - loss: 7.6566 - accuracy: 0.5007
16800/25000 [===================>..........] - ETA: 22s - loss: 7.6566 - accuracy: 0.5007
16832/25000 [===================>..........] - ETA: 22s - loss: 7.6530 - accuracy: 0.5009
16864/25000 [===================>..........] - ETA: 22s - loss: 7.6539 - accuracy: 0.5008
16896/25000 [===================>..........] - ETA: 22s - loss: 7.6575 - accuracy: 0.5006
16928/25000 [===================>..........] - ETA: 22s - loss: 7.6557 - accuracy: 0.5007
16960/25000 [===================>..........] - ETA: 22s - loss: 7.6531 - accuracy: 0.5009
16992/25000 [===================>..........] - ETA: 22s - loss: 7.6549 - accuracy: 0.5008
17024/25000 [===================>..........] - ETA: 21s - loss: 7.6585 - accuracy: 0.5005
17056/25000 [===================>..........] - ETA: 21s - loss: 7.6585 - accuracy: 0.5005
17088/25000 [===================>..........] - ETA: 21s - loss: 7.6541 - accuracy: 0.5008
17120/25000 [===================>..........] - ETA: 21s - loss: 7.6496 - accuracy: 0.5011
17152/25000 [===================>..........] - ETA: 21s - loss: 7.6496 - accuracy: 0.5011
17184/25000 [===================>..........] - ETA: 21s - loss: 7.6488 - accuracy: 0.5012
17216/25000 [===================>..........] - ETA: 21s - loss: 7.6479 - accuracy: 0.5012
17248/25000 [===================>..........] - ETA: 21s - loss: 7.6453 - accuracy: 0.5014
17280/25000 [===================>..........] - ETA: 21s - loss: 7.6435 - accuracy: 0.5015
17312/25000 [===================>..........] - ETA: 21s - loss: 7.6418 - accuracy: 0.5016
17344/25000 [===================>..........] - ETA: 21s - loss: 7.6436 - accuracy: 0.5015
17376/25000 [===================>..........] - ETA: 20s - loss: 7.6428 - accuracy: 0.5016
17408/25000 [===================>..........] - ETA: 20s - loss: 7.6437 - accuracy: 0.5015
17440/25000 [===================>..........] - ETA: 20s - loss: 7.6411 - accuracy: 0.5017
17472/25000 [===================>..........] - ETA: 20s - loss: 7.6438 - accuracy: 0.5015
17504/25000 [====================>.........] - ETA: 20s - loss: 7.6421 - accuracy: 0.5016
17536/25000 [====================>.........] - ETA: 20s - loss: 7.6430 - accuracy: 0.5015
17568/25000 [====================>.........] - ETA: 20s - loss: 7.6448 - accuracy: 0.5014
17600/25000 [====================>.........] - ETA: 20s - loss: 7.6431 - accuracy: 0.5015
17632/25000 [====================>.........] - ETA: 20s - loss: 7.6388 - accuracy: 0.5018
17664/25000 [====================>.........] - ETA: 20s - loss: 7.6328 - accuracy: 0.5022
17696/25000 [====================>.........] - ETA: 20s - loss: 7.6294 - accuracy: 0.5024
17728/25000 [====================>.........] - ETA: 19s - loss: 7.6312 - accuracy: 0.5023
17760/25000 [====================>.........] - ETA: 19s - loss: 7.6269 - accuracy: 0.5026
17792/25000 [====================>.........] - ETA: 19s - loss: 7.6278 - accuracy: 0.5025
17824/25000 [====================>.........] - ETA: 19s - loss: 7.6288 - accuracy: 0.5025
17856/25000 [====================>.........] - ETA: 19s - loss: 7.6314 - accuracy: 0.5023
17888/25000 [====================>.........] - ETA: 19s - loss: 7.6272 - accuracy: 0.5026
17920/25000 [====================>.........] - ETA: 19s - loss: 7.6281 - accuracy: 0.5025
17952/25000 [====================>.........] - ETA: 19s - loss: 7.6256 - accuracy: 0.5027
17984/25000 [====================>.........] - ETA: 19s - loss: 7.6240 - accuracy: 0.5028
18016/25000 [====================>.........] - ETA: 19s - loss: 7.6215 - accuracy: 0.5029
18048/25000 [====================>.........] - ETA: 19s - loss: 7.6207 - accuracy: 0.5030
18080/25000 [====================>.........] - ETA: 18s - loss: 7.6174 - accuracy: 0.5032
18112/25000 [====================>.........] - ETA: 18s - loss: 7.6133 - accuracy: 0.5035
18144/25000 [====================>.........] - ETA: 18s - loss: 7.6083 - accuracy: 0.5038
18176/25000 [====================>.........] - ETA: 18s - loss: 7.6109 - accuracy: 0.5036
18208/25000 [====================>.........] - ETA: 18s - loss: 7.6110 - accuracy: 0.5036
18240/25000 [====================>.........] - ETA: 18s - loss: 7.6111 - accuracy: 0.5036
18272/25000 [====================>.........] - ETA: 18s - loss: 7.6087 - accuracy: 0.5038
18304/25000 [====================>.........] - ETA: 18s - loss: 7.6122 - accuracy: 0.5036
18336/25000 [=====================>........] - ETA: 18s - loss: 7.6139 - accuracy: 0.5034
18368/25000 [=====================>........] - ETA: 18s - loss: 7.6140 - accuracy: 0.5034
18400/25000 [=====================>........] - ETA: 18s - loss: 7.6175 - accuracy: 0.5032
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6159 - accuracy: 0.5033
18464/25000 [=====================>........] - ETA: 17s - loss: 7.6176 - accuracy: 0.5032
18496/25000 [=====================>........] - ETA: 17s - loss: 7.6169 - accuracy: 0.5032
18528/25000 [=====================>........] - ETA: 17s - loss: 7.6178 - accuracy: 0.5032
18560/25000 [=====================>........] - ETA: 17s - loss: 7.6179 - accuracy: 0.5032
18592/25000 [=====================>........] - ETA: 17s - loss: 7.6180 - accuracy: 0.5032
18624/25000 [=====================>........] - ETA: 17s - loss: 7.6180 - accuracy: 0.5032
18656/25000 [=====================>........] - ETA: 17s - loss: 7.6189 - accuracy: 0.5031
18688/25000 [=====================>........] - ETA: 17s - loss: 7.6182 - accuracy: 0.5032
18720/25000 [=====================>........] - ETA: 17s - loss: 7.6199 - accuracy: 0.5030
18752/25000 [=====================>........] - ETA: 17s - loss: 7.6216 - accuracy: 0.5029
18784/25000 [=====================>........] - ETA: 17s - loss: 7.6217 - accuracy: 0.5029
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6194 - accuracy: 0.5031
18848/25000 [=====================>........] - ETA: 16s - loss: 7.6202 - accuracy: 0.5030
18880/25000 [=====================>........] - ETA: 16s - loss: 7.6195 - accuracy: 0.5031
18912/25000 [=====================>........] - ETA: 16s - loss: 7.6196 - accuracy: 0.5031
18944/25000 [=====================>........] - ETA: 16s - loss: 7.6237 - accuracy: 0.5028
18976/25000 [=====================>........] - ETA: 16s - loss: 7.6230 - accuracy: 0.5028
19008/25000 [=====================>........] - ETA: 16s - loss: 7.6247 - accuracy: 0.5027
19040/25000 [=====================>........] - ETA: 16s - loss: 7.6264 - accuracy: 0.5026
19072/25000 [=====================>........] - ETA: 16s - loss: 7.6264 - accuracy: 0.5026
19104/25000 [=====================>........] - ETA: 16s - loss: 7.6233 - accuracy: 0.5028
19136/25000 [=====================>........] - ETA: 16s - loss: 7.6242 - accuracy: 0.5028
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6242 - accuracy: 0.5028
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6251 - accuracy: 0.5027
19232/25000 [======================>.......] - ETA: 15s - loss: 7.6276 - accuracy: 0.5025
19264/25000 [======================>.......] - ETA: 15s - loss: 7.6268 - accuracy: 0.5026
19296/25000 [======================>.......] - ETA: 15s - loss: 7.6277 - accuracy: 0.5025
19328/25000 [======================>.......] - ETA: 15s - loss: 7.6262 - accuracy: 0.5026
19360/25000 [======================>.......] - ETA: 15s - loss: 7.6254 - accuracy: 0.5027
19392/25000 [======================>.......] - ETA: 15s - loss: 7.6239 - accuracy: 0.5028
19424/25000 [======================>.......] - ETA: 15s - loss: 7.6264 - accuracy: 0.5026
19456/25000 [======================>.......] - ETA: 15s - loss: 7.6296 - accuracy: 0.5024
19488/25000 [======================>.......] - ETA: 15s - loss: 7.6273 - accuracy: 0.5026
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6226 - accuracy: 0.5029
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6258 - accuracy: 0.5027
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6251 - accuracy: 0.5027
19616/25000 [======================>.......] - ETA: 14s - loss: 7.6275 - accuracy: 0.5025
19648/25000 [======================>.......] - ETA: 14s - loss: 7.6268 - accuracy: 0.5026
19680/25000 [======================>.......] - ETA: 14s - loss: 7.6253 - accuracy: 0.5027
19712/25000 [======================>.......] - ETA: 14s - loss: 7.6246 - accuracy: 0.5027
19744/25000 [======================>.......] - ETA: 14s - loss: 7.6270 - accuracy: 0.5026
19776/25000 [======================>.......] - ETA: 14s - loss: 7.6294 - accuracy: 0.5024
19808/25000 [======================>.......] - ETA: 14s - loss: 7.6310 - accuracy: 0.5023
19840/25000 [======================>.......] - ETA: 14s - loss: 7.6280 - accuracy: 0.5025
19872/25000 [======================>.......] - ETA: 14s - loss: 7.6273 - accuracy: 0.5026
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6304 - accuracy: 0.5024
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6297 - accuracy: 0.5024
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6313 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 13s - loss: 7.6329 - accuracy: 0.5022
20032/25000 [=======================>......] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
20064/25000 [=======================>......] - ETA: 13s - loss: 7.6353 - accuracy: 0.5020
20096/25000 [=======================>......] - ETA: 13s - loss: 7.6376 - accuracy: 0.5019
20128/25000 [=======================>......] - ETA: 13s - loss: 7.6369 - accuracy: 0.5019
20160/25000 [=======================>......] - ETA: 13s - loss: 7.6347 - accuracy: 0.5021
20192/25000 [=======================>......] - ETA: 13s - loss: 7.6332 - accuracy: 0.5022
20224/25000 [=======================>......] - ETA: 13s - loss: 7.6348 - accuracy: 0.5021
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6310 - accuracy: 0.5023
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6319 - accuracy: 0.5023
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6327 - accuracy: 0.5022
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6335 - accuracy: 0.5022
20384/25000 [=======================>......] - ETA: 12s - loss: 7.6335 - accuracy: 0.5022
20416/25000 [=======================>......] - ETA: 12s - loss: 7.6313 - accuracy: 0.5023
20448/25000 [=======================>......] - ETA: 12s - loss: 7.6306 - accuracy: 0.5023
20480/25000 [=======================>......] - ETA: 12s - loss: 7.6299 - accuracy: 0.5024
20512/25000 [=======================>......] - ETA: 12s - loss: 7.6248 - accuracy: 0.5027
20544/25000 [=======================>......] - ETA: 12s - loss: 7.6241 - accuracy: 0.5028
20576/25000 [=======================>......] - ETA: 12s - loss: 7.6256 - accuracy: 0.5027
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6227 - accuracy: 0.5029
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6250 - accuracy: 0.5027
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6258 - accuracy: 0.5027
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6281 - accuracy: 0.5025
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6289 - accuracy: 0.5025
20768/25000 [=======================>......] - ETA: 11s - loss: 7.6312 - accuracy: 0.5023
20800/25000 [=======================>......] - ETA: 11s - loss: 7.6305 - accuracy: 0.5024
20832/25000 [=======================>......] - ETA: 11s - loss: 7.6306 - accuracy: 0.5024
20864/25000 [========================>.....] - ETA: 11s - loss: 7.6291 - accuracy: 0.5024
20896/25000 [========================>.....] - ETA: 11s - loss: 7.6248 - accuracy: 0.5027
20928/25000 [========================>.....] - ETA: 11s - loss: 7.6241 - accuracy: 0.5028
20960/25000 [========================>.....] - ETA: 11s - loss: 7.6264 - accuracy: 0.5026
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6301 - accuracy: 0.5024
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6287 - accuracy: 0.5025
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6331 - accuracy: 0.5022
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6303 - accuracy: 0.5024
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6310 - accuracy: 0.5023
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6296 - accuracy: 0.5024
21184/25000 [========================>.....] - ETA: 10s - loss: 7.6297 - accuracy: 0.5024
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6305 - accuracy: 0.5024
21248/25000 [========================>.....] - ETA: 10s - loss: 7.6284 - accuracy: 0.5025
21280/25000 [========================>.....] - ETA: 10s - loss: 7.6277 - accuracy: 0.5025
21312/25000 [========================>.....] - ETA: 10s - loss: 7.6278 - accuracy: 0.5025
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6257 - accuracy: 0.5027 
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6229 - accuracy: 0.5029
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6208 - accuracy: 0.5030
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6216 - accuracy: 0.5029
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6202 - accuracy: 0.5030
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6196 - accuracy: 0.5031
21536/25000 [========================>.....] - ETA: 9s - loss: 7.6211 - accuracy: 0.5030
21568/25000 [========================>.....] - ETA: 9s - loss: 7.6211 - accuracy: 0.5030
21600/25000 [========================>.....] - ETA: 9s - loss: 7.6212 - accuracy: 0.5030
21632/25000 [========================>.....] - ETA: 9s - loss: 7.6205 - accuracy: 0.5030
21664/25000 [========================>.....] - ETA: 9s - loss: 7.6234 - accuracy: 0.5028
21696/25000 [=========================>....] - ETA: 9s - loss: 7.6235 - accuracy: 0.5028
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6285 - accuracy: 0.5025
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6300 - accuracy: 0.5024
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6300 - accuracy: 0.5024
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6322 - accuracy: 0.5022
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6266 - accuracy: 0.5026
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6253 - accuracy: 0.5027
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6281 - accuracy: 0.5025
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6289 - accuracy: 0.5025
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6297 - accuracy: 0.5024
22016/25000 [=========================>....] - ETA: 8s - loss: 7.6304 - accuracy: 0.5024
22048/25000 [=========================>....] - ETA: 8s - loss: 7.6305 - accuracy: 0.5024
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6319 - accuracy: 0.5023
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6326 - accuracy: 0.5022
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6327 - accuracy: 0.5022
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6314 - accuracy: 0.5023
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6300 - accuracy: 0.5024
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6294 - accuracy: 0.5024
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6315 - accuracy: 0.5023
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6316 - accuracy: 0.5023
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6323 - accuracy: 0.5022
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6317 - accuracy: 0.5023
22400/25000 [=========================>....] - ETA: 7s - loss: 7.6338 - accuracy: 0.5021
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6304 - accuracy: 0.5024
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6291 - accuracy: 0.5024
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6332 - accuracy: 0.5022
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6339 - accuracy: 0.5021
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6367 - accuracy: 0.5020
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6395 - accuracy: 0.5018
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6395 - accuracy: 0.5018
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6409 - accuracy: 0.5017
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6416 - accuracy: 0.5016
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6416 - accuracy: 0.5016
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6424 - accuracy: 0.5016
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6444 - accuracy: 0.5014
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6444 - accuracy: 0.5014
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6472 - accuracy: 0.5013
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6486 - accuracy: 0.5012
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6500 - accuracy: 0.5011
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6520 - accuracy: 0.5010
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6494 - accuracy: 0.5011
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6494 - accuracy: 0.5011
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6481 - accuracy: 0.5012
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6508 - accuracy: 0.5010
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6555 - accuracy: 0.5007
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6561 - accuracy: 0.5007
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6542 - accuracy: 0.5008
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6562 - accuracy: 0.5007
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6575 - accuracy: 0.5006
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6562 - accuracy: 0.5007
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6575 - accuracy: 0.5006
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6549 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6589 - accuracy: 0.5005
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6557 - accuracy: 0.5007
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24192/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24224/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24256/25000 [============================>.] - ETA: 2s - loss: 7.6565 - accuracy: 0.5007
24288/25000 [============================>.] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
24320/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24352/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24416/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24448/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24480/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24544/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24576/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24832/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 80s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

