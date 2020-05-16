
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/be4e81fe281eae9822d779771f5b85f7e37f3171', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'be4e81fe281eae9822d779771f5b85f7e37f3171', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/be4e81fe281eae9822d779771f5b85f7e37f3171

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/be4e81fe281eae9822d779771f5b85f7e37f3171

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
	Data preprocessing and feature engineering runtime = 0.24s ...
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
 40%|████      | 2/5 [00:46<01:10, 23.34s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.22881304018048182, 'embedding_size_factor': 1.3194961912782177, 'layers.choice': 3, 'learning_rate': 0.000788590567337245, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.01626550595034007} and reward: 0.35
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcdI\xbe\xe6<\x9b\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x1c\xa8\t\xcb\xc8\xbaX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?I\xd7-Y+\xbf=X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x90\xa7\xe7\xa0p<\x08u.' and reward: 0.35
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcdI\xbe\xe6<\x9b\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x1c\xa8\t\xcb\xc8\xbaX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?I\xd7-Y+\xbf=X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x90\xa7\xe7\xa0p<\x08u.' and reward: 0.35
 60%|██████    | 3/5 [01:36<01:02, 31.24s/it] 60%|██████    | 3/5 [01:36<01:04, 32.11s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4356234062740046, 'embedding_size_factor': 0.8806491492903235, 'layers.choice': 1, 'learning_rate': 0.0010270708044957456, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.007422576118893505} and reward: 0.3684
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xe1@\xfe\xd4j\x05X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec.G\x1f\xeex\x1aX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?P\xd3\xd8\xe1\x05\xe8\x9fX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?~g"\x9a\xeci\xeeu.' and reward: 0.3684
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xe1@\xfe\xd4j\x05X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec.G\x1f\xeex\x1aX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?P\xd3\xd8\xe1\x05\xe8\x9fX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?~g"\x9a\xeci\xeeu.' and reward: 0.3684
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 187.51256680488586
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -70.04s of remaining time.
Ensemble size: 73
Ensemble weights: 
[0.53424658 0.26027397 0.20547945]
	0.3918	 = Validation accuracy score
	0.96s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 191.04s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f6fba35eba8> 

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
 [ 0.03961803  0.05699047  0.00143914  0.04804053 -0.01824197  0.1856797 ]
 [ 0.17195857  0.05155836  0.05356225  0.07731479 -0.062207   -0.007947  ]
 [ 0.04474952 -0.04672777  0.03179231  0.29696012  0.13381179  0.18149792]
 [ 0.03530026 -0.01948578  0.03919059  0.09250525  0.00691826 -0.01649241]
 [-0.17281994  0.05959371 -0.01525548  0.2975671   0.06159377 -0.14217964]
 [ 0.51750982  0.55501467 -0.15097675  0.6711666   0.33937323  0.39899665]
 [ 0.0736548   0.25019485 -0.42119727  0.75673681  0.36809748 -0.48640203]
 [ 0.26264033  0.00335633 -0.64265728  0.20211942  0.54563594  0.32267794]
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
{'loss': 0.5746228396892548, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 00:38:49.533412: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.42364418506622314, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 00:38:50.555666: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3342336/17464789 [====>.........................] - ETA: 0s
13443072/17464789 [======================>.......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 00:39:00.808833: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-16 00:39:00.812609: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-16 00:39:00.812764: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fe88b45ce0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:39:00.812778: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:10 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 2:35 - loss: 8.6249 - accuracy: 0.4375
   96/25000 [..............................] - ETA: 2:03 - loss: 8.4652 - accuracy: 0.4479
  128/25000 [..............................] - ETA: 1:47 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:37 - loss: 7.5708 - accuracy: 0.5063
  192/25000 [..............................] - ETA: 1:29 - loss: 7.8263 - accuracy: 0.4896
  224/25000 [..............................] - ETA: 1:25 - loss: 8.2142 - accuracy: 0.4643
  256/25000 [..............................] - ETA: 1:22 - loss: 8.2656 - accuracy: 0.4609
  288/25000 [..............................] - ETA: 1:20 - loss: 8.4120 - accuracy: 0.4514
  320/25000 [..............................] - ETA: 1:18 - loss: 8.1458 - accuracy: 0.4688
  352/25000 [..............................] - ETA: 1:16 - loss: 8.1458 - accuracy: 0.4688
  384/25000 [..............................] - ETA: 1:14 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:13 - loss: 7.9983 - accuracy: 0.4784
  448/25000 [..............................] - ETA: 1:13 - loss: 7.9062 - accuracy: 0.4844
  480/25000 [..............................] - ETA: 1:12 - loss: 7.8583 - accuracy: 0.4875
  512/25000 [..............................] - ETA: 1:11 - loss: 7.7265 - accuracy: 0.4961
  544/25000 [..............................] - ETA: 1:10 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:09 - loss: 7.6134 - accuracy: 0.5035
  608/25000 [..............................] - ETA: 1:09 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:08 - loss: 7.4750 - accuracy: 0.5125
  672/25000 [..............................] - ETA: 1:08 - loss: 7.4613 - accuracy: 0.5134
  704/25000 [..............................] - ETA: 1:07 - loss: 7.5142 - accuracy: 0.5099
  736/25000 [..............................] - ETA: 1:07 - loss: 7.4166 - accuracy: 0.5163
  768/25000 [..............................] - ETA: 1:06 - loss: 7.4869 - accuracy: 0.5117
  800/25000 [..............................] - ETA: 1:06 - loss: 7.4366 - accuracy: 0.5150
  832/25000 [..............................] - ETA: 1:05 - loss: 7.5192 - accuracy: 0.5096
  864/25000 [>.............................] - ETA: 1:05 - loss: 7.4892 - accuracy: 0.5116
  896/25000 [>.............................] - ETA: 1:04 - loss: 7.4784 - accuracy: 0.5123
  928/25000 [>.............................] - ETA: 1:04 - loss: 7.4518 - accuracy: 0.5140
  960/25000 [>.............................] - ETA: 1:04 - loss: 7.4750 - accuracy: 0.5125
  992/25000 [>.............................] - ETA: 1:04 - loss: 7.4193 - accuracy: 0.5161
 1024/25000 [>.............................] - ETA: 1:03 - loss: 7.3671 - accuracy: 0.5195
 1056/25000 [>.............................] - ETA: 1:03 - loss: 7.3907 - accuracy: 0.5180
 1088/25000 [>.............................] - ETA: 1:03 - loss: 7.3848 - accuracy: 0.5184
 1120/25000 [>.............................] - ETA: 1:02 - loss: 7.3381 - accuracy: 0.5214
 1152/25000 [>.............................] - ETA: 1:02 - loss: 7.3472 - accuracy: 0.5208
 1184/25000 [>.............................] - ETA: 1:03 - loss: 7.3040 - accuracy: 0.5236
 1216/25000 [>.............................] - ETA: 1:03 - loss: 7.2631 - accuracy: 0.5263
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.2612 - accuracy: 0.5264
 1280/25000 [>.............................] - ETA: 1:04 - loss: 7.2833 - accuracy: 0.5250
 1312/25000 [>.............................] - ETA: 1:04 - loss: 7.2693 - accuracy: 0.5259
 1344/25000 [>.............................] - ETA: 1:05 - loss: 7.2673 - accuracy: 0.5260
 1376/25000 [>.............................] - ETA: 1:05 - loss: 7.2989 - accuracy: 0.5240
 1408/25000 [>.............................] - ETA: 1:06 - loss: 7.3072 - accuracy: 0.5234
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.3365 - accuracy: 0.5215
 1472/25000 [>.............................] - ETA: 1:07 - loss: 7.3541 - accuracy: 0.5204
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.3506 - accuracy: 0.5206
 1536/25000 [>.............................] - ETA: 1:06 - loss: 7.3472 - accuracy: 0.5208
 1568/25000 [>.............................] - ETA: 1:06 - loss: 7.2950 - accuracy: 0.5242
 1600/25000 [>.............................] - ETA: 1:06 - loss: 7.2833 - accuracy: 0.5250
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.2720 - accuracy: 0.5257
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.2335 - accuracy: 0.5282
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.2327 - accuracy: 0.5283
 1728/25000 [=>............................] - ETA: 1:04 - loss: 7.2141 - accuracy: 0.5295
 1760/25000 [=>............................] - ETA: 1:04 - loss: 7.1962 - accuracy: 0.5307
 1792/25000 [=>............................] - ETA: 1:04 - loss: 7.2302 - accuracy: 0.5285
 1824/25000 [=>............................] - ETA: 1:03 - loss: 7.2883 - accuracy: 0.5247
 1856/25000 [=>............................] - ETA: 1:03 - loss: 7.2453 - accuracy: 0.5275
 1888/25000 [=>............................] - ETA: 1:03 - loss: 7.2281 - accuracy: 0.5286
 1920/25000 [=>............................] - ETA: 1:02 - loss: 7.2833 - accuracy: 0.5250
 1952/25000 [=>............................] - ETA: 1:02 - loss: 7.2896 - accuracy: 0.5246
 1984/25000 [=>............................] - ETA: 1:02 - loss: 7.2261 - accuracy: 0.5287
 2016/25000 [=>............................] - ETA: 1:02 - loss: 7.2559 - accuracy: 0.5268
 2048/25000 [=>............................] - ETA: 1:01 - loss: 7.2623 - accuracy: 0.5264
 2080/25000 [=>............................] - ETA: 1:01 - loss: 7.2685 - accuracy: 0.5260
 2112/25000 [=>............................] - ETA: 1:01 - loss: 7.2673 - accuracy: 0.5260
 2144/25000 [=>............................] - ETA: 1:01 - loss: 7.3019 - accuracy: 0.5238
 2176/25000 [=>............................] - ETA: 1:01 - loss: 7.3284 - accuracy: 0.5221
 2208/25000 [=>............................] - ETA: 1:00 - loss: 7.3402 - accuracy: 0.5213
 2240/25000 [=>............................] - ETA: 1:00 - loss: 7.3586 - accuracy: 0.5201
 2272/25000 [=>............................] - ETA: 1:00 - loss: 7.3427 - accuracy: 0.5211
 2304/25000 [=>............................] - ETA: 1:00 - loss: 7.3339 - accuracy: 0.5217
 2336/25000 [=>............................] - ETA: 1:00 - loss: 7.3516 - accuracy: 0.5205
 2368/25000 [=>............................] - ETA: 59s - loss: 7.3817 - accuracy: 0.5186 
 2400/25000 [=>............................] - ETA: 59s - loss: 7.3919 - accuracy: 0.5179
 2432/25000 [=>............................] - ETA: 59s - loss: 7.4018 - accuracy: 0.5173
 2464/25000 [=>............................] - ETA: 59s - loss: 7.4239 - accuracy: 0.5158
 2496/25000 [=>............................] - ETA: 59s - loss: 7.4148 - accuracy: 0.5164
 2528/25000 [==>...........................] - ETA: 59s - loss: 7.4361 - accuracy: 0.5150
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.4151 - accuracy: 0.5164
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.4122 - accuracy: 0.5166
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.4095 - accuracy: 0.5168
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.4126 - accuracy: 0.5166
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.4270 - accuracy: 0.5156
 2720/25000 [==>...........................] - ETA: 58s - loss: 7.4242 - accuracy: 0.5158
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.4493 - accuracy: 0.5142
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.4518 - accuracy: 0.5140
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.4379 - accuracy: 0.5149
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.4566 - accuracy: 0.5137
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.4377 - accuracy: 0.5149
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.4402 - accuracy: 0.5148
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.4687 - accuracy: 0.5129
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.4605 - accuracy: 0.5134
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.4780 - accuracy: 0.5123
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.4750 - accuracy: 0.5125
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.4770 - accuracy: 0.5124
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.4740 - accuracy: 0.5126
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.4808 - accuracy: 0.5121
 3168/25000 [==>...........................] - ETA: 55s - loss: 7.4682 - accuracy: 0.5129
 3200/25000 [==>...........................] - ETA: 55s - loss: 7.4606 - accuracy: 0.5134
 3232/25000 [==>...........................] - ETA: 55s - loss: 7.4484 - accuracy: 0.5142
 3264/25000 [==>...........................] - ETA: 55s - loss: 7.4646 - accuracy: 0.5132
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.4712 - accuracy: 0.5127
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.4961 - accuracy: 0.5111
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.5023 - accuracy: 0.5107
 3392/25000 [===>..........................] - ETA: 54s - loss: 7.4903 - accuracy: 0.5115
 3424/25000 [===>..........................] - ETA: 54s - loss: 7.4964 - accuracy: 0.5111
 3456/25000 [===>..........................] - ETA: 54s - loss: 7.4936 - accuracy: 0.5113
 3488/25000 [===>..........................] - ETA: 54s - loss: 7.4996 - accuracy: 0.5109
 3520/25000 [===>..........................] - ETA: 54s - loss: 7.4750 - accuracy: 0.5125
 3552/25000 [===>..........................] - ETA: 54s - loss: 7.4637 - accuracy: 0.5132
 3584/25000 [===>..........................] - ETA: 54s - loss: 7.4570 - accuracy: 0.5137
 3616/25000 [===>..........................] - ETA: 53s - loss: 7.4716 - accuracy: 0.5127
 3648/25000 [===>..........................] - ETA: 53s - loss: 7.4817 - accuracy: 0.5121
 3680/25000 [===>..........................] - ETA: 53s - loss: 7.4833 - accuracy: 0.5120
 3712/25000 [===>..........................] - ETA: 53s - loss: 7.4807 - accuracy: 0.5121
 3744/25000 [===>..........................] - ETA: 53s - loss: 7.4741 - accuracy: 0.5126
 3776/25000 [===>..........................] - ETA: 53s - loss: 7.4798 - accuracy: 0.5122
 3808/25000 [===>..........................] - ETA: 53s - loss: 7.4733 - accuracy: 0.5126
 3840/25000 [===>..........................] - ETA: 53s - loss: 7.4869 - accuracy: 0.5117
 3872/25000 [===>..........................] - ETA: 52s - loss: 7.4845 - accuracy: 0.5119
 3904/25000 [===>..........................] - ETA: 52s - loss: 7.4860 - accuracy: 0.5118
 3936/25000 [===>..........................] - ETA: 52s - loss: 7.4835 - accuracy: 0.5119
 3968/25000 [===>..........................] - ETA: 52s - loss: 7.4889 - accuracy: 0.5116
 4000/25000 [===>..........................] - ETA: 52s - loss: 7.4903 - accuracy: 0.5115
 4032/25000 [===>..........................] - ETA: 52s - loss: 7.5031 - accuracy: 0.5107
 4064/25000 [===>..........................] - ETA: 52s - loss: 7.4931 - accuracy: 0.5113
 4096/25000 [===>..........................] - ETA: 52s - loss: 7.4982 - accuracy: 0.5110
 4128/25000 [===>..........................] - ETA: 52s - loss: 7.5143 - accuracy: 0.5099
 4160/25000 [===>..........................] - ETA: 51s - loss: 7.5118 - accuracy: 0.5101
 4192/25000 [====>.........................] - ETA: 51s - loss: 7.5130 - accuracy: 0.5100
 4224/25000 [====>.........................] - ETA: 51s - loss: 7.5178 - accuracy: 0.5097
 4256/25000 [====>.........................] - ETA: 51s - loss: 7.5261 - accuracy: 0.5092
 4288/25000 [====>.........................] - ETA: 51s - loss: 7.5379 - accuracy: 0.5084
 4320/25000 [====>.........................] - ETA: 51s - loss: 7.5353 - accuracy: 0.5086
 4352/25000 [====>.........................] - ETA: 51s - loss: 7.5327 - accuracy: 0.5087
 4384/25000 [====>.........................] - ETA: 51s - loss: 7.5232 - accuracy: 0.5094
 4416/25000 [====>.........................] - ETA: 51s - loss: 7.5277 - accuracy: 0.5091
 4448/25000 [====>.........................] - ETA: 51s - loss: 7.5218 - accuracy: 0.5094
 4480/25000 [====>.........................] - ETA: 51s - loss: 7.5366 - accuracy: 0.5085
 4512/25000 [====>.........................] - ETA: 50s - loss: 7.5443 - accuracy: 0.5080
 4544/25000 [====>.........................] - ETA: 50s - loss: 7.5418 - accuracy: 0.5081
 4576/25000 [====>.........................] - ETA: 50s - loss: 7.5560 - accuracy: 0.5072
 4608/25000 [====>.........................] - ETA: 50s - loss: 7.5635 - accuracy: 0.5067
 4640/25000 [====>.........................] - ETA: 50s - loss: 7.5642 - accuracy: 0.5067
 4672/25000 [====>.........................] - ETA: 50s - loss: 7.5616 - accuracy: 0.5068
 4704/25000 [====>.........................] - ETA: 50s - loss: 7.5656 - accuracy: 0.5066
 4736/25000 [====>.........................] - ETA: 50s - loss: 7.5598 - accuracy: 0.5070
 4768/25000 [====>.........................] - ETA: 50s - loss: 7.5734 - accuracy: 0.5061
 4800/25000 [====>.........................] - ETA: 50s - loss: 7.5772 - accuracy: 0.5058
 4832/25000 [====>.........................] - ETA: 49s - loss: 7.5936 - accuracy: 0.5048
 4864/25000 [====>.........................] - ETA: 49s - loss: 7.6036 - accuracy: 0.5041
 4896/25000 [====>.........................] - ETA: 49s - loss: 7.5946 - accuracy: 0.5047
 4928/25000 [====>.........................] - ETA: 49s - loss: 7.5982 - accuracy: 0.5045
 4960/25000 [====>.........................] - ETA: 49s - loss: 7.5955 - accuracy: 0.5046
 4992/25000 [====>.........................] - ETA: 49s - loss: 7.5990 - accuracy: 0.5044
 5024/25000 [=====>........................] - ETA: 49s - loss: 7.5995 - accuracy: 0.5044
 5056/25000 [=====>........................] - ETA: 49s - loss: 7.6029 - accuracy: 0.5042
 5088/25000 [=====>........................] - ETA: 49s - loss: 7.6033 - accuracy: 0.5041
 5120/25000 [=====>........................] - ETA: 49s - loss: 7.6007 - accuracy: 0.5043
 5152/25000 [=====>........................] - ETA: 48s - loss: 7.6011 - accuracy: 0.5043
 5184/25000 [=====>........................] - ETA: 48s - loss: 7.5956 - accuracy: 0.5046
 5216/25000 [=====>........................] - ETA: 48s - loss: 7.5902 - accuracy: 0.5050
 5248/25000 [=====>........................] - ETA: 48s - loss: 7.5936 - accuracy: 0.5048
 5280/25000 [=====>........................] - ETA: 48s - loss: 7.5911 - accuracy: 0.5049
 5312/25000 [=====>........................] - ETA: 48s - loss: 7.5916 - accuracy: 0.5049
 5344/25000 [=====>........................] - ETA: 48s - loss: 7.6035 - accuracy: 0.5041
 5376/25000 [=====>........................] - ETA: 48s - loss: 7.6096 - accuracy: 0.5037
 5408/25000 [=====>........................] - ETA: 48s - loss: 7.5957 - accuracy: 0.5046
 5440/25000 [=====>........................] - ETA: 48s - loss: 7.5990 - accuracy: 0.5044
 5472/25000 [=====>........................] - ETA: 48s - loss: 7.6078 - accuracy: 0.5038
 5504/25000 [=====>........................] - ETA: 47s - loss: 7.6193 - accuracy: 0.5031
 5536/25000 [=====>........................] - ETA: 47s - loss: 7.6223 - accuracy: 0.5029
 5568/25000 [=====>........................] - ETA: 47s - loss: 7.6281 - accuracy: 0.5025
 5600/25000 [=====>........................] - ETA: 47s - loss: 7.6475 - accuracy: 0.5013
 5632/25000 [=====>........................] - ETA: 47s - loss: 7.6448 - accuracy: 0.5014
 5664/25000 [=====>........................] - ETA: 47s - loss: 7.6450 - accuracy: 0.5014
 5696/25000 [=====>........................] - ETA: 47s - loss: 7.6397 - accuracy: 0.5018
 5728/25000 [=====>........................] - ETA: 47s - loss: 7.6372 - accuracy: 0.5019
 5760/25000 [=====>........................] - ETA: 47s - loss: 7.6586 - accuracy: 0.5005
 5792/25000 [=====>........................] - ETA: 47s - loss: 7.6587 - accuracy: 0.5005
 5824/25000 [=====>........................] - ETA: 47s - loss: 7.6456 - accuracy: 0.5014
 5856/25000 [======>.......................] - ETA: 46s - loss: 7.6483 - accuracy: 0.5012
 5888/25000 [======>.......................] - ETA: 46s - loss: 7.6510 - accuracy: 0.5010
 5920/25000 [======>.......................] - ETA: 46s - loss: 7.6563 - accuracy: 0.5007
 5952/25000 [======>.......................] - ETA: 46s - loss: 7.6537 - accuracy: 0.5008
 5984/25000 [======>.......................] - ETA: 46s - loss: 7.6512 - accuracy: 0.5010
 6016/25000 [======>.......................] - ETA: 46s - loss: 7.6309 - accuracy: 0.5023
 6048/25000 [======>.......................] - ETA: 46s - loss: 7.6286 - accuracy: 0.5025
 6080/25000 [======>.......................] - ETA: 46s - loss: 7.6288 - accuracy: 0.5025
 6112/25000 [======>.......................] - ETA: 46s - loss: 7.6465 - accuracy: 0.5013
 6144/25000 [======>.......................] - ETA: 46s - loss: 7.6541 - accuracy: 0.5008
 6176/25000 [======>.......................] - ETA: 46s - loss: 7.6492 - accuracy: 0.5011
 6208/25000 [======>.......................] - ETA: 45s - loss: 7.6493 - accuracy: 0.5011
 6240/25000 [======>.......................] - ETA: 45s - loss: 7.6371 - accuracy: 0.5019
 6272/25000 [======>.......................] - ETA: 45s - loss: 7.6446 - accuracy: 0.5014
 6304/25000 [======>.......................] - ETA: 45s - loss: 7.6618 - accuracy: 0.5003
 6336/25000 [======>.......................] - ETA: 45s - loss: 7.6569 - accuracy: 0.5006
 6368/25000 [======>.......................] - ETA: 45s - loss: 7.6594 - accuracy: 0.5005
 6400/25000 [======>.......................] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
 6432/25000 [======>.......................] - ETA: 45s - loss: 7.6690 - accuracy: 0.4998
 6464/25000 [======>.......................] - ETA: 45s - loss: 7.6737 - accuracy: 0.4995
 6496/25000 [======>.......................] - ETA: 45s - loss: 7.6784 - accuracy: 0.4992
 6528/25000 [======>.......................] - ETA: 45s - loss: 7.6784 - accuracy: 0.4992
 6560/25000 [======>.......................] - ETA: 44s - loss: 7.6853 - accuracy: 0.4988
 6592/25000 [======>.......................] - ETA: 44s - loss: 7.6876 - accuracy: 0.4986
 6624/25000 [======>.......................] - ETA: 44s - loss: 7.6898 - accuracy: 0.4985
 6656/25000 [======>.......................] - ETA: 44s - loss: 7.6804 - accuracy: 0.4991
 6688/25000 [=======>......................] - ETA: 44s - loss: 7.6735 - accuracy: 0.4996
 6720/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 6752/25000 [=======>......................] - ETA: 44s - loss: 7.6689 - accuracy: 0.4999
 6784/25000 [=======>......................] - ETA: 44s - loss: 7.6689 - accuracy: 0.4999
 6816/25000 [=======>......................] - ETA: 44s - loss: 7.6576 - accuracy: 0.5006
 6848/25000 [=======>......................] - ETA: 44s - loss: 7.6599 - accuracy: 0.5004
 6880/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 6912/25000 [=======>......................] - ETA: 44s - loss: 7.6577 - accuracy: 0.5006
 6944/25000 [=======>......................] - ETA: 44s - loss: 7.6556 - accuracy: 0.5007
 6976/25000 [=======>......................] - ETA: 43s - loss: 7.6512 - accuracy: 0.5010
 7008/25000 [=======>......................] - ETA: 43s - loss: 7.6579 - accuracy: 0.5006
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.6644 - accuracy: 0.5001
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7104/25000 [=======>......................] - ETA: 43s - loss: 7.6601 - accuracy: 0.5004
 7136/25000 [=======>......................] - ETA: 43s - loss: 7.6688 - accuracy: 0.4999
 7168/25000 [=======>......................] - ETA: 43s - loss: 7.6709 - accuracy: 0.4997
 7200/25000 [=======>......................] - ETA: 43s - loss: 7.6730 - accuracy: 0.4996
 7232/25000 [=======>......................] - ETA: 43s - loss: 7.6730 - accuracy: 0.4996
 7264/25000 [=======>......................] - ETA: 43s - loss: 7.6687 - accuracy: 0.4999
 7296/25000 [=======>......................] - ETA: 42s - loss: 7.6708 - accuracy: 0.4997
 7328/25000 [=======>......................] - ETA: 42s - loss: 7.6729 - accuracy: 0.4996
 7360/25000 [=======>......................] - ETA: 42s - loss: 7.6645 - accuracy: 0.5001
 7392/25000 [=======>......................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.6584 - accuracy: 0.5005
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.6625 - accuracy: 0.5003
 7488/25000 [=======>......................] - ETA: 42s - loss: 7.6584 - accuracy: 0.5005
 7520/25000 [========>.....................] - ETA: 42s - loss: 7.6544 - accuracy: 0.5008
 7552/25000 [========>.....................] - ETA: 42s - loss: 7.6524 - accuracy: 0.5009
 7584/25000 [========>.....................] - ETA: 42s - loss: 7.6504 - accuracy: 0.5011
 7616/25000 [========>.....................] - ETA: 42s - loss: 7.6425 - accuracy: 0.5016
 7648/25000 [========>.....................] - ETA: 41s - loss: 7.6446 - accuracy: 0.5014
 7680/25000 [========>.....................] - ETA: 41s - loss: 7.6387 - accuracy: 0.5018
 7712/25000 [========>.....................] - ETA: 41s - loss: 7.6428 - accuracy: 0.5016
 7744/25000 [========>.....................] - ETA: 41s - loss: 7.6409 - accuracy: 0.5017
 7776/25000 [========>.....................] - ETA: 41s - loss: 7.6410 - accuracy: 0.5017
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.6450 - accuracy: 0.5014
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.6373 - accuracy: 0.5019
 7872/25000 [========>.....................] - ETA: 41s - loss: 7.6355 - accuracy: 0.5020
 7904/25000 [========>.....................] - ETA: 41s - loss: 7.6414 - accuracy: 0.5016
 7936/25000 [========>.....................] - ETA: 41s - loss: 7.6454 - accuracy: 0.5014
 7968/25000 [========>.....................] - ETA: 41s - loss: 7.6512 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 41s - loss: 7.6532 - accuracy: 0.5009
 8032/25000 [========>.....................] - ETA: 41s - loss: 7.6533 - accuracy: 0.5009
 8064/25000 [========>.....................] - ETA: 41s - loss: 7.6495 - accuracy: 0.5011
 8096/25000 [========>.....................] - ETA: 40s - loss: 7.6534 - accuracy: 0.5009
 8128/25000 [========>.....................] - ETA: 40s - loss: 7.6515 - accuracy: 0.5010
 8160/25000 [========>.....................] - ETA: 40s - loss: 7.6422 - accuracy: 0.5016
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6442 - accuracy: 0.5015
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6424 - accuracy: 0.5016
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6332 - accuracy: 0.5022
 8288/25000 [========>.....................] - ETA: 40s - loss: 7.6315 - accuracy: 0.5023
 8320/25000 [========>.....................] - ETA: 40s - loss: 7.6242 - accuracy: 0.5028
 8352/25000 [=========>....................] - ETA: 40s - loss: 7.6281 - accuracy: 0.5025
 8384/25000 [=========>....................] - ETA: 40s - loss: 7.6246 - accuracy: 0.5027
 8416/25000 [=========>....................] - ETA: 40s - loss: 7.6320 - accuracy: 0.5023
 8448/25000 [=========>....................] - ETA: 39s - loss: 7.6376 - accuracy: 0.5019
 8480/25000 [=========>....................] - ETA: 39s - loss: 7.6449 - accuracy: 0.5014
 8512/25000 [=========>....................] - ETA: 39s - loss: 7.6432 - accuracy: 0.5015
 8544/25000 [=========>....................] - ETA: 39s - loss: 7.6505 - accuracy: 0.5011
 8576/25000 [=========>....................] - ETA: 39s - loss: 7.6505 - accuracy: 0.5010
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6470 - accuracy: 0.5013
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6382 - accuracy: 0.5019
 8672/25000 [=========>....................] - ETA: 39s - loss: 7.6242 - accuracy: 0.5028
 8704/25000 [=========>....................] - ETA: 39s - loss: 7.6226 - accuracy: 0.5029
 8736/25000 [=========>....................] - ETA: 39s - loss: 7.6263 - accuracy: 0.5026
 8768/25000 [=========>....................] - ETA: 39s - loss: 7.6246 - accuracy: 0.5027
 8800/25000 [=========>....................] - ETA: 39s - loss: 7.6231 - accuracy: 0.5028
 8832/25000 [=========>....................] - ETA: 38s - loss: 7.6163 - accuracy: 0.5033
 8864/25000 [=========>....................] - ETA: 38s - loss: 7.6113 - accuracy: 0.5036
 8896/25000 [=========>....................] - ETA: 38s - loss: 7.6149 - accuracy: 0.5034
 8928/25000 [=========>....................] - ETA: 38s - loss: 7.6082 - accuracy: 0.5038
 8960/25000 [=========>....................] - ETA: 38s - loss: 7.6101 - accuracy: 0.5037
 8992/25000 [=========>....................] - ETA: 38s - loss: 7.6018 - accuracy: 0.5042
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6054 - accuracy: 0.5040
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6124 - accuracy: 0.5035
 9088/25000 [=========>....................] - ETA: 38s - loss: 7.6093 - accuracy: 0.5037
 9120/25000 [=========>....................] - ETA: 38s - loss: 7.6162 - accuracy: 0.5033
 9152/25000 [=========>....................] - ETA: 38s - loss: 7.6147 - accuracy: 0.5034
 9184/25000 [==========>...................] - ETA: 38s - loss: 7.6132 - accuracy: 0.5035
 9216/25000 [==========>...................] - ETA: 37s - loss: 7.6117 - accuracy: 0.5036
 9248/25000 [==========>...................] - ETA: 37s - loss: 7.6152 - accuracy: 0.5034
 9280/25000 [==========>...................] - ETA: 37s - loss: 7.6253 - accuracy: 0.5027
 9312/25000 [==========>...................] - ETA: 37s - loss: 7.6255 - accuracy: 0.5027
 9344/25000 [==========>...................] - ETA: 37s - loss: 7.6272 - accuracy: 0.5026
 9376/25000 [==========>...................] - ETA: 37s - loss: 7.6208 - accuracy: 0.5030
 9408/25000 [==========>...................] - ETA: 37s - loss: 7.6259 - accuracy: 0.5027
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6228 - accuracy: 0.5029
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6245 - accuracy: 0.5027
 9504/25000 [==========>...................] - ETA: 37s - loss: 7.6134 - accuracy: 0.5035
 9536/25000 [==========>...................] - ETA: 37s - loss: 7.6119 - accuracy: 0.5036
 9568/25000 [==========>...................] - ETA: 37s - loss: 7.6121 - accuracy: 0.5036
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6123 - accuracy: 0.5035
 9632/25000 [==========>...................] - ETA: 36s - loss: 7.6077 - accuracy: 0.5038
 9664/25000 [==========>...................] - ETA: 36s - loss: 7.6016 - accuracy: 0.5042
 9696/25000 [==========>...................] - ETA: 36s - loss: 7.6034 - accuracy: 0.5041
 9728/25000 [==========>...................] - ETA: 36s - loss: 7.6020 - accuracy: 0.5042
 9760/25000 [==========>...................] - ETA: 36s - loss: 7.6038 - accuracy: 0.5041
 9792/25000 [==========>...................] - ETA: 36s - loss: 7.5993 - accuracy: 0.5044
 9824/25000 [==========>...................] - ETA: 36s - loss: 7.5995 - accuracy: 0.5044
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6028 - accuracy: 0.5042
 9888/25000 [==========>...................] - ETA: 36s - loss: 7.6030 - accuracy: 0.5041
 9920/25000 [==========>...................] - ETA: 36s - loss: 7.6048 - accuracy: 0.5040
 9952/25000 [==========>...................] - ETA: 36s - loss: 7.6081 - accuracy: 0.5038
 9984/25000 [==========>...................] - ETA: 36s - loss: 7.6144 - accuracy: 0.5034
10016/25000 [===========>..................] - ETA: 36s - loss: 7.6069 - accuracy: 0.5039
10048/25000 [===========>..................] - ETA: 35s - loss: 7.6117 - accuracy: 0.5036
10080/25000 [===========>..................] - ETA: 35s - loss: 7.6149 - accuracy: 0.5034
10112/25000 [===========>..................] - ETA: 35s - loss: 7.6135 - accuracy: 0.5035
10144/25000 [===========>..................] - ETA: 35s - loss: 7.6167 - accuracy: 0.5033
10176/25000 [===========>..................] - ETA: 35s - loss: 7.6169 - accuracy: 0.5032
10208/25000 [===========>..................] - ETA: 35s - loss: 7.6140 - accuracy: 0.5034
10240/25000 [===========>..................] - ETA: 35s - loss: 7.6082 - accuracy: 0.5038
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6099 - accuracy: 0.5037
10304/25000 [===========>..................] - ETA: 35s - loss: 7.6056 - accuracy: 0.5040
10336/25000 [===========>..................] - ETA: 35s - loss: 7.6088 - accuracy: 0.5038
10368/25000 [===========>..................] - ETA: 35s - loss: 7.6089 - accuracy: 0.5038
10400/25000 [===========>..................] - ETA: 35s - loss: 7.6121 - accuracy: 0.5036
10432/25000 [===========>..................] - ETA: 34s - loss: 7.6049 - accuracy: 0.5040
10464/25000 [===========>..................] - ETA: 34s - loss: 7.6153 - accuracy: 0.5033
10496/25000 [===========>..................] - ETA: 34s - loss: 7.6199 - accuracy: 0.5030
10528/25000 [===========>..................] - ETA: 34s - loss: 7.6229 - accuracy: 0.5028
10560/25000 [===========>..................] - ETA: 34s - loss: 7.6245 - accuracy: 0.5027
10592/25000 [===========>..................] - ETA: 34s - loss: 7.6188 - accuracy: 0.5031
10624/25000 [===========>..................] - ETA: 34s - loss: 7.6204 - accuracy: 0.5030
10656/25000 [===========>..................] - ETA: 34s - loss: 7.6148 - accuracy: 0.5034
10688/25000 [===========>..................] - ETA: 34s - loss: 7.6193 - accuracy: 0.5031
10720/25000 [===========>..................] - ETA: 34s - loss: 7.6194 - accuracy: 0.5031
10752/25000 [===========>..................] - ETA: 34s - loss: 7.6224 - accuracy: 0.5029
10784/25000 [===========>..................] - ETA: 34s - loss: 7.6240 - accuracy: 0.5028
10816/25000 [===========>..................] - ETA: 34s - loss: 7.6269 - accuracy: 0.5026
10848/25000 [============>.................] - ETA: 34s - loss: 7.6256 - accuracy: 0.5027
10880/25000 [============>.................] - ETA: 33s - loss: 7.6300 - accuracy: 0.5024
10912/25000 [============>.................] - ETA: 33s - loss: 7.6343 - accuracy: 0.5021
10944/25000 [============>.................] - ETA: 33s - loss: 7.6288 - accuracy: 0.5025
10976/25000 [============>.................] - ETA: 33s - loss: 7.6331 - accuracy: 0.5022
11008/25000 [============>.................] - ETA: 33s - loss: 7.6374 - accuracy: 0.5019
11040/25000 [============>.................] - ETA: 33s - loss: 7.6361 - accuracy: 0.5020
11072/25000 [============>.................] - ETA: 33s - loss: 7.6445 - accuracy: 0.5014
11104/25000 [============>.................] - ETA: 33s - loss: 7.6542 - accuracy: 0.5008
11136/25000 [============>.................] - ETA: 33s - loss: 7.6556 - accuracy: 0.5007
11168/25000 [============>.................] - ETA: 33s - loss: 7.6543 - accuracy: 0.5008
11200/25000 [============>.................] - ETA: 33s - loss: 7.6598 - accuracy: 0.5004
11232/25000 [============>.................] - ETA: 33s - loss: 7.6530 - accuracy: 0.5009
11264/25000 [============>.................] - ETA: 33s - loss: 7.6530 - accuracy: 0.5009
11296/25000 [============>.................] - ETA: 32s - loss: 7.6476 - accuracy: 0.5012
11328/25000 [============>.................] - ETA: 32s - loss: 7.6490 - accuracy: 0.5011
11360/25000 [============>.................] - ETA: 32s - loss: 7.6410 - accuracy: 0.5017
11392/25000 [============>.................] - ETA: 32s - loss: 7.6397 - accuracy: 0.5018
11424/25000 [============>.................] - ETA: 32s - loss: 7.6357 - accuracy: 0.5020
11456/25000 [============>.................] - ETA: 32s - loss: 7.6385 - accuracy: 0.5018
11488/25000 [============>.................] - ETA: 32s - loss: 7.6373 - accuracy: 0.5019
11520/25000 [============>.................] - ETA: 32s - loss: 7.6413 - accuracy: 0.5016
11552/25000 [============>.................] - ETA: 32s - loss: 7.6454 - accuracy: 0.5014
11584/25000 [============>.................] - ETA: 32s - loss: 7.6428 - accuracy: 0.5016
11616/25000 [============>.................] - ETA: 32s - loss: 7.6402 - accuracy: 0.5017
11648/25000 [============>.................] - ETA: 32s - loss: 7.6377 - accuracy: 0.5019
11680/25000 [=============>................] - ETA: 32s - loss: 7.6351 - accuracy: 0.5021
11712/25000 [=============>................] - ETA: 31s - loss: 7.6339 - accuracy: 0.5021
11744/25000 [=============>................] - ETA: 31s - loss: 7.6327 - accuracy: 0.5022
11776/25000 [=============>................] - ETA: 31s - loss: 7.6315 - accuracy: 0.5023
11808/25000 [=============>................] - ETA: 31s - loss: 7.6264 - accuracy: 0.5026
11840/25000 [=============>................] - ETA: 31s - loss: 7.6304 - accuracy: 0.5024
11872/25000 [=============>................] - ETA: 31s - loss: 7.6292 - accuracy: 0.5024
11904/25000 [=============>................] - ETA: 31s - loss: 7.6306 - accuracy: 0.5024
11936/25000 [=============>................] - ETA: 31s - loss: 7.6384 - accuracy: 0.5018
11968/25000 [=============>................] - ETA: 31s - loss: 7.6333 - accuracy: 0.5022
12000/25000 [=============>................] - ETA: 31s - loss: 7.6308 - accuracy: 0.5023
12032/25000 [=============>................] - ETA: 31s - loss: 7.6322 - accuracy: 0.5022
12064/25000 [=============>................] - ETA: 31s - loss: 7.6348 - accuracy: 0.5021
12096/25000 [=============>................] - ETA: 31s - loss: 7.6324 - accuracy: 0.5022
12128/25000 [=============>................] - ETA: 31s - loss: 7.6375 - accuracy: 0.5019
12160/25000 [=============>................] - ETA: 30s - loss: 7.6351 - accuracy: 0.5021
12192/25000 [=============>................] - ETA: 30s - loss: 7.6415 - accuracy: 0.5016
12224/25000 [=============>................] - ETA: 30s - loss: 7.6440 - accuracy: 0.5015
12256/25000 [=============>................] - ETA: 30s - loss: 7.6378 - accuracy: 0.5019
12288/25000 [=============>................] - ETA: 30s - loss: 7.6429 - accuracy: 0.5015
12320/25000 [=============>................] - ETA: 30s - loss: 7.6492 - accuracy: 0.5011
12352/25000 [=============>................] - ETA: 30s - loss: 7.6505 - accuracy: 0.5011
12384/25000 [=============>................] - ETA: 30s - loss: 7.6456 - accuracy: 0.5014
12416/25000 [=============>................] - ETA: 30s - loss: 7.6395 - accuracy: 0.5018
12448/25000 [=============>................] - ETA: 30s - loss: 7.6420 - accuracy: 0.5016
12480/25000 [=============>................] - ETA: 30s - loss: 7.6519 - accuracy: 0.5010
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6556 - accuracy: 0.5007
12544/25000 [==============>...............] - ETA: 29s - loss: 7.6568 - accuracy: 0.5006
12576/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12608/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12640/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12672/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12704/25000 [==============>...............] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6594 - accuracy: 0.5005
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6594 - accuracy: 0.5005
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6631 - accuracy: 0.5002
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12960/25000 [==============>...............] - ETA: 28s - loss: 7.6654 - accuracy: 0.5001
12992/25000 [==============>...............] - ETA: 28s - loss: 7.6690 - accuracy: 0.4998
13024/25000 [==============>...............] - ETA: 28s - loss: 7.6678 - accuracy: 0.4999
13056/25000 [==============>...............] - ETA: 28s - loss: 7.6654 - accuracy: 0.5001
13088/25000 [==============>...............] - ETA: 28s - loss: 7.6654 - accuracy: 0.5001
13120/25000 [==============>...............] - ETA: 28s - loss: 7.6631 - accuracy: 0.5002
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6655 - accuracy: 0.5001
13184/25000 [==============>...............] - ETA: 28s - loss: 7.6678 - accuracy: 0.4999
13216/25000 [==============>...............] - ETA: 28s - loss: 7.6689 - accuracy: 0.4998
13248/25000 [==============>...............] - ETA: 28s - loss: 7.6689 - accuracy: 0.4998
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6689 - accuracy: 0.4998
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6735 - accuracy: 0.4995
13344/25000 [===============>..............] - ETA: 27s - loss: 7.6712 - accuracy: 0.4997
13376/25000 [===============>..............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
13408/25000 [===============>..............] - ETA: 27s - loss: 7.6620 - accuracy: 0.5003
13440/25000 [===============>..............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
13472/25000 [===============>..............] - ETA: 27s - loss: 7.6678 - accuracy: 0.4999
13504/25000 [===============>..............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
13536/25000 [===============>..............] - ETA: 27s - loss: 7.6678 - accuracy: 0.4999
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6689 - accuracy: 0.4999
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6723 - accuracy: 0.4996
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6689 - accuracy: 0.4999
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6644 - accuracy: 0.5001
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6689 - accuracy: 0.4999
13728/25000 [===============>..............] - ETA: 26s - loss: 7.6610 - accuracy: 0.5004
13760/25000 [===============>..............] - ETA: 26s - loss: 7.6633 - accuracy: 0.5002
13792/25000 [===============>..............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
13824/25000 [===============>..............] - ETA: 26s - loss: 7.6699 - accuracy: 0.4998
13856/25000 [===============>..............] - ETA: 26s - loss: 7.6733 - accuracy: 0.4996
13888/25000 [===============>..............] - ETA: 26s - loss: 7.6710 - accuracy: 0.4997
13920/25000 [===============>..............] - ETA: 26s - loss: 7.6688 - accuracy: 0.4999
13952/25000 [===============>..............] - ETA: 26s - loss: 7.6710 - accuracy: 0.4997
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6699 - accuracy: 0.4998
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6699 - accuracy: 0.4998
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6688 - accuracy: 0.4999
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6634 - accuracy: 0.5002
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6644 - accuracy: 0.5001
14144/25000 [===============>..............] - ETA: 25s - loss: 7.6677 - accuracy: 0.4999
14176/25000 [================>.............] - ETA: 25s - loss: 7.6709 - accuracy: 0.4997
14208/25000 [================>.............] - ETA: 25s - loss: 7.6742 - accuracy: 0.4995
14240/25000 [================>.............] - ETA: 25s - loss: 7.6752 - accuracy: 0.4994
14272/25000 [================>.............] - ETA: 25s - loss: 7.6784 - accuracy: 0.4992
14304/25000 [================>.............] - ETA: 25s - loss: 7.6752 - accuracy: 0.4994
14336/25000 [================>.............] - ETA: 25s - loss: 7.6741 - accuracy: 0.4995
14368/25000 [================>.............] - ETA: 25s - loss: 7.6741 - accuracy: 0.4995
14400/25000 [================>.............] - ETA: 25s - loss: 7.6698 - accuracy: 0.4998
14432/25000 [================>.............] - ETA: 25s - loss: 7.6687 - accuracy: 0.4999
14464/25000 [================>.............] - ETA: 25s - loss: 7.6656 - accuracy: 0.5001
14496/25000 [================>.............] - ETA: 25s - loss: 7.6677 - accuracy: 0.4999
14528/25000 [================>.............] - ETA: 24s - loss: 7.6635 - accuracy: 0.5002
14560/25000 [================>.............] - ETA: 24s - loss: 7.6708 - accuracy: 0.4997
14592/25000 [================>.............] - ETA: 24s - loss: 7.6708 - accuracy: 0.4997
14624/25000 [================>.............] - ETA: 24s - loss: 7.6740 - accuracy: 0.4995
14656/25000 [================>.............] - ETA: 24s - loss: 7.6771 - accuracy: 0.4993
14688/25000 [================>.............] - ETA: 24s - loss: 7.6760 - accuracy: 0.4994
14720/25000 [================>.............] - ETA: 24s - loss: 7.6760 - accuracy: 0.4994
14752/25000 [================>.............] - ETA: 24s - loss: 7.6791 - accuracy: 0.4992
14784/25000 [================>.............] - ETA: 24s - loss: 7.6801 - accuracy: 0.4991
14816/25000 [================>.............] - ETA: 24s - loss: 7.6790 - accuracy: 0.4992
14848/25000 [================>.............] - ETA: 24s - loss: 7.6780 - accuracy: 0.4993
14880/25000 [================>.............] - ETA: 24s - loss: 7.6790 - accuracy: 0.4992
14912/25000 [================>.............] - ETA: 24s - loss: 7.6800 - accuracy: 0.4991
14944/25000 [================>.............] - ETA: 23s - loss: 7.6779 - accuracy: 0.4993
14976/25000 [================>.............] - ETA: 23s - loss: 7.6748 - accuracy: 0.4995
15008/25000 [=================>............] - ETA: 23s - loss: 7.6779 - accuracy: 0.4993
15040/25000 [=================>............] - ETA: 23s - loss: 7.6799 - accuracy: 0.4991
15072/25000 [=================>............] - ETA: 23s - loss: 7.6778 - accuracy: 0.4993
15104/25000 [=================>............] - ETA: 23s - loss: 7.6798 - accuracy: 0.4991
15136/25000 [=================>............] - ETA: 23s - loss: 7.6778 - accuracy: 0.4993
15168/25000 [=================>............] - ETA: 23s - loss: 7.6747 - accuracy: 0.4995
15200/25000 [=================>............] - ETA: 23s - loss: 7.6737 - accuracy: 0.4995
15232/25000 [=================>............] - ETA: 23s - loss: 7.6757 - accuracy: 0.4994
15264/25000 [=================>............] - ETA: 23s - loss: 7.6737 - accuracy: 0.4995
15296/25000 [=================>............] - ETA: 23s - loss: 7.6706 - accuracy: 0.4997
15328/25000 [=================>............] - ETA: 22s - loss: 7.6686 - accuracy: 0.4999
15360/25000 [=================>............] - ETA: 22s - loss: 7.6686 - accuracy: 0.4999
15392/25000 [=================>............] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15424/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15456/25000 [=================>............] - ETA: 22s - loss: 7.6627 - accuracy: 0.5003
15488/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15520/25000 [=================>............] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15552/25000 [=================>............] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
15584/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15616/25000 [=================>............] - ETA: 22s - loss: 7.6686 - accuracy: 0.4999
15648/25000 [=================>............] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15680/25000 [=================>............] - ETA: 22s - loss: 7.6686 - accuracy: 0.4999
15712/25000 [=================>............] - ETA: 22s - loss: 7.6637 - accuracy: 0.5002
15744/25000 [=================>............] - ETA: 21s - loss: 7.6569 - accuracy: 0.5006
15776/25000 [=================>............] - ETA: 21s - loss: 7.6540 - accuracy: 0.5008
15808/25000 [=================>............] - ETA: 21s - loss: 7.6521 - accuracy: 0.5009
15840/25000 [==================>...........] - ETA: 21s - loss: 7.6540 - accuracy: 0.5008
15872/25000 [==================>...........] - ETA: 21s - loss: 7.6550 - accuracy: 0.5008
15904/25000 [==================>...........] - ETA: 21s - loss: 7.6541 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6560 - accuracy: 0.5007
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6541 - accuracy: 0.5008
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6542 - accuracy: 0.5008
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6523 - accuracy: 0.5009
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6513 - accuracy: 0.5010
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6504 - accuracy: 0.5011
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6505 - accuracy: 0.5011
16160/25000 [==================>...........] - ETA: 20s - loss: 7.6524 - accuracy: 0.5009
16192/25000 [==================>...........] - ETA: 20s - loss: 7.6534 - accuracy: 0.5009
16224/25000 [==================>...........] - ETA: 20s - loss: 7.6543 - accuracy: 0.5008
16256/25000 [==================>...........] - ETA: 20s - loss: 7.6553 - accuracy: 0.5007
16288/25000 [==================>...........] - ETA: 20s - loss: 7.6525 - accuracy: 0.5009
16320/25000 [==================>...........] - ETA: 20s - loss: 7.6553 - accuracy: 0.5007
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6554 - accuracy: 0.5007
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6582 - accuracy: 0.5005
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6582 - accuracy: 0.5005
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6582 - accuracy: 0.5005
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6564 - accuracy: 0.5007
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6583 - accuracy: 0.5005
16544/25000 [==================>...........] - ETA: 19s - loss: 7.6611 - accuracy: 0.5004
16576/25000 [==================>...........] - ETA: 19s - loss: 7.6601 - accuracy: 0.5004
16608/25000 [==================>...........] - ETA: 19s - loss: 7.6592 - accuracy: 0.5005
16640/25000 [==================>...........] - ETA: 19s - loss: 7.6546 - accuracy: 0.5008
16672/25000 [===================>..........] - ETA: 19s - loss: 7.6501 - accuracy: 0.5011
16704/25000 [===================>..........] - ETA: 19s - loss: 7.6510 - accuracy: 0.5010
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6520 - accuracy: 0.5010
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6538 - accuracy: 0.5008
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6529 - accuracy: 0.5009
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6530 - accuracy: 0.5009
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6521 - accuracy: 0.5009
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6494 - accuracy: 0.5011
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6539 - accuracy: 0.5008
16960/25000 [===================>..........] - ETA: 18s - loss: 7.6503 - accuracy: 0.5011
16992/25000 [===================>..........] - ETA: 18s - loss: 7.6540 - accuracy: 0.5008
17024/25000 [===================>..........] - ETA: 18s - loss: 7.6558 - accuracy: 0.5007
17056/25000 [===================>..........] - ETA: 18s - loss: 7.6558 - accuracy: 0.5007
17088/25000 [===================>..........] - ETA: 18s - loss: 7.6594 - accuracy: 0.5005
17120/25000 [===================>..........] - ETA: 18s - loss: 7.6586 - accuracy: 0.5005
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6595 - accuracy: 0.5005
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6639 - accuracy: 0.5002
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6613 - accuracy: 0.5003
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6568 - accuracy: 0.5006
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6595 - accuracy: 0.5005
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6595 - accuracy: 0.5005
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6595 - accuracy: 0.5005
17376/25000 [===================>..........] - ETA: 17s - loss: 7.6578 - accuracy: 0.5006
17408/25000 [===================>..........] - ETA: 17s - loss: 7.6596 - accuracy: 0.5005
17440/25000 [===================>..........] - ETA: 17s - loss: 7.6569 - accuracy: 0.5006
17472/25000 [===================>..........] - ETA: 17s - loss: 7.6578 - accuracy: 0.5006
17504/25000 [====================>.........] - ETA: 17s - loss: 7.6579 - accuracy: 0.5006
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6553 - accuracy: 0.5007
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6579 - accuracy: 0.5006
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6588 - accuracy: 0.5005
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6536 - accuracy: 0.5009
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6536 - accuracy: 0.5008
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6528 - accuracy: 0.5009
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6528 - accuracy: 0.5009
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6537 - accuracy: 0.5008
17792/25000 [====================>.........] - ETA: 16s - loss: 7.6563 - accuracy: 0.5007
17824/25000 [====================>.........] - ETA: 16s - loss: 7.6615 - accuracy: 0.5003
17856/25000 [====================>.........] - ETA: 16s - loss: 7.6615 - accuracy: 0.5003
17888/25000 [====================>.........] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
17920/25000 [====================>.........] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6607 - accuracy: 0.5004
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6598 - accuracy: 0.5004
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6615 - accuracy: 0.5003
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6624 - accuracy: 0.5003
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6632 - accuracy: 0.5002
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6599 - accuracy: 0.5004
18208/25000 [====================>.........] - ETA: 15s - loss: 7.6616 - accuracy: 0.5003
18240/25000 [====================>.........] - ETA: 15s - loss: 7.6641 - accuracy: 0.5002
18272/25000 [====================>.........] - ETA: 15s - loss: 7.6641 - accuracy: 0.5002
18304/25000 [====================>.........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18336/25000 [=====================>........] - ETA: 15s - loss: 7.6624 - accuracy: 0.5003
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6599 - accuracy: 0.5004
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6616 - accuracy: 0.5003
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6666 - accuracy: 0.5000
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6716 - accuracy: 0.4997
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6749 - accuracy: 0.4995
18624/25000 [=====================>........] - ETA: 14s - loss: 7.6740 - accuracy: 0.4995
18656/25000 [=====================>........] - ETA: 14s - loss: 7.6773 - accuracy: 0.4993
18688/25000 [=====================>........] - ETA: 14s - loss: 7.6773 - accuracy: 0.4993
18720/25000 [=====================>........] - ETA: 14s - loss: 7.6781 - accuracy: 0.4993
18752/25000 [=====================>........] - ETA: 14s - loss: 7.6772 - accuracy: 0.4993
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6764 - accuracy: 0.4994
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6788 - accuracy: 0.4992
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6804 - accuracy: 0.4991
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6821 - accuracy: 0.4990
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6877 - accuracy: 0.4986
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6901 - accuracy: 0.4985
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6892 - accuracy: 0.4985
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6908 - accuracy: 0.4984
19040/25000 [=====================>........] - ETA: 13s - loss: 7.6892 - accuracy: 0.4985
19072/25000 [=====================>........] - ETA: 13s - loss: 7.6883 - accuracy: 0.4986
19104/25000 [=====================>........] - ETA: 13s - loss: 7.6851 - accuracy: 0.4988
19136/25000 [=====================>........] - ETA: 13s - loss: 7.6834 - accuracy: 0.4989
19168/25000 [======================>.......] - ETA: 13s - loss: 7.6794 - accuracy: 0.4992
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6770 - accuracy: 0.4993
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6778 - accuracy: 0.4993
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6754 - accuracy: 0.4994
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6738 - accuracy: 0.4995
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6730 - accuracy: 0.4996
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6729 - accuracy: 0.4996
19456/25000 [======================>.......] - ETA: 12s - loss: 7.6729 - accuracy: 0.4996
19488/25000 [======================>.......] - ETA: 12s - loss: 7.6753 - accuracy: 0.4994
19520/25000 [======================>.......] - ETA: 12s - loss: 7.6776 - accuracy: 0.4993
19552/25000 [======================>.......] - ETA: 12s - loss: 7.6745 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6737 - accuracy: 0.4995
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6768 - accuracy: 0.4993
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6729 - accuracy: 0.4996
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6736 - accuracy: 0.4995
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6736 - accuracy: 0.4995
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6736 - accuracy: 0.4995
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6775 - accuracy: 0.4993
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6790 - accuracy: 0.4992
19872/25000 [======================>.......] - ETA: 11s - loss: 7.6774 - accuracy: 0.4993
19904/25000 [======================>.......] - ETA: 11s - loss: 7.6751 - accuracy: 0.4994
19936/25000 [======================>.......] - ETA: 11s - loss: 7.6735 - accuracy: 0.4995
19968/25000 [======================>.......] - ETA: 11s - loss: 7.6743 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6758 - accuracy: 0.4994
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6720 - accuracy: 0.4997
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6720 - accuracy: 0.4997
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6720 - accuracy: 0.4997
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6712 - accuracy: 0.4997
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6719 - accuracy: 0.4997
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6719 - accuracy: 0.4997
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6719 - accuracy: 0.4997
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6734 - accuracy: 0.4996
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6742 - accuracy: 0.4995
20320/25000 [=======================>......] - ETA: 10s - loss: 7.6764 - accuracy: 0.4994
20352/25000 [=======================>......] - ETA: 10s - loss: 7.6726 - accuracy: 0.4996
20384/25000 [=======================>......] - ETA: 10s - loss: 7.6719 - accuracy: 0.4997
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6726 - accuracy: 0.4996
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6726 - accuracy: 0.4996
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6719 - accuracy: 0.4997
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6733 - accuracy: 0.4996
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6741 - accuracy: 0.4995
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6733 - accuracy: 0.4996
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6755 - accuracy: 0.4994
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6748 - accuracy: 0.4995
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6763 - accuracy: 0.4994
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6755 - accuracy: 0.4994
20736/25000 [=======================>......] - ETA: 9s - loss: 7.6733 - accuracy: 0.4996 
20768/25000 [=======================>......] - ETA: 9s - loss: 7.6740 - accuracy: 0.4995
20800/25000 [=======================>......] - ETA: 9s - loss: 7.6733 - accuracy: 0.4996
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6725 - accuracy: 0.4996
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6740 - accuracy: 0.4995
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6740 - accuracy: 0.4995
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6739 - accuracy: 0.4995
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6739 - accuracy: 0.4995
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6710 - accuracy: 0.4997
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6695 - accuracy: 0.4998
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6695 - accuracy: 0.4998
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6688 - accuracy: 0.4999
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6644 - accuracy: 0.5001
21152/25000 [========================>.....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6681 - accuracy: 0.4999
21216/25000 [========================>.....] - ETA: 8s - loss: 7.6681 - accuracy: 0.4999
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6695 - accuracy: 0.4998
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6695 - accuracy: 0.4998
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6723 - accuracy: 0.4996
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6765 - accuracy: 0.4994
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6765 - accuracy: 0.4994
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6765 - accuracy: 0.4994
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6786 - accuracy: 0.4992
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6793 - accuracy: 0.4992
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6771 - accuracy: 0.4993
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6799 - accuracy: 0.4991
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6792 - accuracy: 0.4992
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6757 - accuracy: 0.4994
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6701 - accuracy: 0.4998
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6687 - accuracy: 0.4999
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6652 - accuracy: 0.5001
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6680 - accuracy: 0.4999
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6659 - accuracy: 0.5000
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6645 - accuracy: 0.5001
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6646 - accuracy: 0.5001
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6646 - accuracy: 0.5001
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6646 - accuracy: 0.5001
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6618 - accuracy: 0.5003
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6543 - accuracy: 0.5008
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6502 - accuracy: 0.5011
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6523 - accuracy: 0.5009
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6550 - accuracy: 0.5008
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6551 - accuracy: 0.5008
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6537 - accuracy: 0.5008
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6517 - accuracy: 0.5010
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6517 - accuracy: 0.5010
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6545 - accuracy: 0.5008
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6498 - accuracy: 0.5011
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6458 - accuracy: 0.5014
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6485 - accuracy: 0.5012
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6492 - accuracy: 0.5011
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6486 - accuracy: 0.5012
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6466 - accuracy: 0.5013
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6446 - accuracy: 0.5014
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6460 - accuracy: 0.5013
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6447 - accuracy: 0.5014
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6447 - accuracy: 0.5014
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6447 - accuracy: 0.5014
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6461 - accuracy: 0.5013
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6461 - accuracy: 0.5013
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6468 - accuracy: 0.5013
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6482 - accuracy: 0.5012
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6475 - accuracy: 0.5012
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6502 - accuracy: 0.5011
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6509 - accuracy: 0.5010
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6496 - accuracy: 0.5011
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6496 - accuracy: 0.5011
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6509 - accuracy: 0.5010
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6529 - accuracy: 0.5009
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6536 - accuracy: 0.5009
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6543 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6517 - accuracy: 0.5010
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6524 - accuracy: 0.5009
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6517 - accuracy: 0.5010
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6530 - accuracy: 0.5009
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6544 - accuracy: 0.5008
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6544 - accuracy: 0.5008
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6550 - accuracy: 0.5008
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6563 - accuracy: 0.5007
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6507 - accuracy: 0.5010
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
24192/25000 [============================>.] - ETA: 1s - loss: 7.6552 - accuracy: 0.5007
24224/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24256/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24288/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24320/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24352/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24416/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24448/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24480/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24512/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24544/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24608/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24736/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 68s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
