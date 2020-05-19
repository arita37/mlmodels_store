
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '73f54da32a5da4768415eb9105ad096255137679', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/73f54da32a5da4768415eb9105ad096255137679

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
 40%|████      | 2/5 [00:46<01:09, 23.01s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4644691663731165, 'embedding_size_factor': 1.0021230433594444, 'layers.choice': 0, 'learning_rate': 0.00013712208497639443, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.0482199048857772e-08} and reward: 0.2886
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\xb9\xdc\xe1\xe4\xaa\xb2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x08\xb2,\x1c\xbf\xb5X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?!\xf9\r\xbd\xb6~\x93X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>F\x82\xa6]\xd7\xfb\x02u.' and reward: 0.2886
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\xb9\xdc\xe1\xe4\xaa\xb2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x08\xb2,\x1c\xbf\xb5X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?!\xf9\r\xbd\xb6~\x93X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>F\x82\xa6]\xd7\xfb\x02u.' and reward: 0.2886
 60%|██████    | 3/5 [01:32<01:00, 30.20s/it] 60%|██████    | 3/5 [01:32<01:01, 30.99s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.0552949265605079, 'embedding_size_factor': 1.1842489664038167, 'layers.choice': 0, 'learning_rate': 0.004687170553733235, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.0224248600630015e-09} and reward: 0.3886
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xacO\x9d\xdal\x98\xcaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\xf2\xaf\x0bPkLX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?s2\xda\xc3\xd3.\xf5X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>)\xf6a\xef'2\xfeu." and reward: 0.3886
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xacO\x9d\xdal\x98\xcaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\xf2\xaf\x0bPkLX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?s2\xda\xc3\xd3.\xf5X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>)\xf6a\xef'2\xfeu." and reward: 0.3886
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 140.4719295501709
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 2, 'dropout_prob': 0.0552949265605079, 'embedding_size_factor': 1.1842489664038167, 'layers.choice': 0, 'learning_rate': 0.004687170553733235, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.0224248600630015e-09}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -22.76s of remaining time.
Ensemble size: 2
Ensemble weights: 
[0.5 0.  0.5]
	0.3968	 = Validation accuracy score
	0.92s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 143.71s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fb392c709b0> 

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
 [-0.01053608  0.01049841 -0.01774422  0.09437051  0.03408168  0.0256343 ]
 [ 0.01819142 -0.0338898   0.17819858  0.03811937 -0.05765348 -0.02933612]
 [-0.11088631  0.21633807  0.00076339 -0.31575471  0.30231324  0.24792883]
 [-0.27477399 -0.12365396 -0.15978721  0.13587993  0.14743854 -0.20207027]
 [-0.18991597 -0.15073919 -0.00990246 -0.14863974  0.21375278 -0.01549906]
 [ 0.03294404 -0.00965468  0.41956773 -0.17777511  0.25041106  0.30205363]
 [ 0.58841288 -0.42462453 -0.31395662  0.36158592  0.38720927 -0.76013994]
 [ 0.24422365  0.0837936  -0.13649771  0.56620055  0.36348227 -0.3570317 ]
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
{'loss': 0.5677342042326927, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 00:23:54.501414: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.521515216678381, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 00:23:55.584202: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2342912/17464789 [===>..........................] - ETA: 0s
 9428992/17464789 [===============>..............] - ETA: 0s
16310272/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-19 00:24:06.350137: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:24:06.354317: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 00:24:06.354509: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555b7c468e40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 00:24:06.354526: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:04 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 2:42 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:15 - loss: 7.0277 - accuracy: 0.5417
  128/25000 [..............................] - ETA: 2:01 - loss: 7.9062 - accuracy: 0.4844
  160/25000 [..............................] - ETA: 1:52 - loss: 7.5708 - accuracy: 0.5063
  192/25000 [..............................] - ETA: 1:46 - loss: 7.5069 - accuracy: 0.5104
  224/25000 [..............................] - ETA: 1:43 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:40 - loss: 7.9661 - accuracy: 0.4805
  288/25000 [..............................] - ETA: 1:38 - loss: 7.7731 - accuracy: 0.4931
  320/25000 [..............................] - ETA: 1:36 - loss: 7.7625 - accuracy: 0.4938
  352/25000 [..............................] - ETA: 1:35 - loss: 7.6666 - accuracy: 0.5000
  384/25000 [..............................] - ETA: 1:33 - loss: 7.4270 - accuracy: 0.5156
  416/25000 [..............................] - ETA: 1:32 - loss: 7.4455 - accuracy: 0.5144
  448/25000 [..............................] - ETA: 1:31 - loss: 7.3586 - accuracy: 0.5201
  480/25000 [..............................] - ETA: 1:30 - loss: 7.5069 - accuracy: 0.5104
  512/25000 [..............................] - ETA: 1:29 - loss: 7.5768 - accuracy: 0.5059
  544/25000 [..............................] - ETA: 1:28 - loss: 7.6102 - accuracy: 0.5037
  576/25000 [..............................] - ETA: 1:28 - loss: 7.6134 - accuracy: 0.5035
  608/25000 [..............................] - ETA: 1:27 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:26 - loss: 7.5947 - accuracy: 0.5047
  672/25000 [..............................] - ETA: 1:27 - loss: 7.5069 - accuracy: 0.5104
  704/25000 [..............................] - ETA: 1:26 - loss: 7.6231 - accuracy: 0.5028
  736/25000 [..............................] - ETA: 1:26 - loss: 7.6458 - accuracy: 0.5014
  768/25000 [..............................] - ETA: 1:26 - loss: 7.5668 - accuracy: 0.5065
  800/25000 [..............................] - ETA: 1:26 - loss: 7.5516 - accuracy: 0.5075
  832/25000 [..............................] - ETA: 1:25 - loss: 7.4823 - accuracy: 0.5120
  864/25000 [>.............................] - ETA: 1:25 - loss: 7.4359 - accuracy: 0.5150
  896/25000 [>.............................] - ETA: 1:25 - loss: 7.3415 - accuracy: 0.5212
  928/25000 [>.............................] - ETA: 1:24 - loss: 7.3362 - accuracy: 0.5216
  960/25000 [>.............................] - ETA: 1:24 - loss: 7.3312 - accuracy: 0.5219
  992/25000 [>.............................] - ETA: 1:24 - loss: 7.3729 - accuracy: 0.5192
 1024/25000 [>.............................] - ETA: 1:23 - loss: 7.4270 - accuracy: 0.5156
 1056/25000 [>.............................] - ETA: 1:23 - loss: 7.3472 - accuracy: 0.5208
 1088/25000 [>.............................] - ETA: 1:23 - loss: 7.3989 - accuracy: 0.5175
 1120/25000 [>.............................] - ETA: 1:22 - loss: 7.3654 - accuracy: 0.5196
 1152/25000 [>.............................] - ETA: 1:22 - loss: 7.4004 - accuracy: 0.5174
 1184/25000 [>.............................] - ETA: 1:22 - loss: 7.4465 - accuracy: 0.5144
 1216/25000 [>.............................] - ETA: 1:22 - loss: 7.4523 - accuracy: 0.5140
 1248/25000 [>.............................] - ETA: 1:21 - loss: 7.4455 - accuracy: 0.5144
 1280/25000 [>.............................] - ETA: 1:21 - loss: 7.4031 - accuracy: 0.5172
 1312/25000 [>.............................] - ETA: 1:21 - loss: 7.4212 - accuracy: 0.5160
 1344/25000 [>.............................] - ETA: 1:21 - loss: 7.4270 - accuracy: 0.5156
 1376/25000 [>.............................] - ETA: 1:20 - loss: 7.4772 - accuracy: 0.5124
 1408/25000 [>.............................] - ETA: 1:20 - loss: 7.4815 - accuracy: 0.5121
 1440/25000 [>.............................] - ETA: 1:20 - loss: 7.4537 - accuracy: 0.5139
 1472/25000 [>.............................] - ETA: 1:20 - loss: 7.4791 - accuracy: 0.5122
 1504/25000 [>.............................] - ETA: 1:20 - loss: 7.5443 - accuracy: 0.5080
 1536/25000 [>.............................] - ETA: 1:20 - loss: 7.5069 - accuracy: 0.5104
 1568/25000 [>.............................] - ETA: 1:19 - loss: 7.5004 - accuracy: 0.5108
 1600/25000 [>.............................] - ETA: 1:19 - loss: 7.4845 - accuracy: 0.5119
 1632/25000 [>.............................] - ETA: 1:19 - loss: 7.4787 - accuracy: 0.5123
 1664/25000 [>.............................] - ETA: 1:19 - loss: 7.4915 - accuracy: 0.5114
 1696/25000 [=>............................] - ETA: 1:19 - loss: 7.4858 - accuracy: 0.5118
 1728/25000 [=>............................] - ETA: 1:19 - loss: 7.5158 - accuracy: 0.5098
 1760/25000 [=>............................] - ETA: 1:18 - loss: 7.5447 - accuracy: 0.5080
 1792/25000 [=>............................] - ETA: 1:18 - loss: 7.5639 - accuracy: 0.5067
 1824/25000 [=>............................] - ETA: 1:18 - loss: 7.5657 - accuracy: 0.5066
 1856/25000 [=>............................] - ETA: 1:18 - loss: 7.5675 - accuracy: 0.5065
 1888/25000 [=>............................] - ETA: 1:18 - loss: 7.5367 - accuracy: 0.5085
 1920/25000 [=>............................] - ETA: 1:18 - loss: 7.5309 - accuracy: 0.5089
 1952/25000 [=>............................] - ETA: 1:18 - loss: 7.5252 - accuracy: 0.5092
 1984/25000 [=>............................] - ETA: 1:18 - loss: 7.4734 - accuracy: 0.5126
 2016/25000 [=>............................] - ETA: 1:18 - loss: 7.4613 - accuracy: 0.5134
 2048/25000 [=>............................] - ETA: 1:17 - loss: 7.5094 - accuracy: 0.5103
 2080/25000 [=>............................] - ETA: 1:17 - loss: 7.5044 - accuracy: 0.5106
 2112/25000 [=>............................] - ETA: 1:17 - loss: 7.5287 - accuracy: 0.5090
 2144/25000 [=>............................] - ETA: 1:17 - loss: 7.5379 - accuracy: 0.5084
 2176/25000 [=>............................] - ETA: 1:17 - loss: 7.5468 - accuracy: 0.5078
 2208/25000 [=>............................] - ETA: 1:16 - loss: 7.5208 - accuracy: 0.5095
 2240/25000 [=>............................] - ETA: 1:16 - loss: 7.5434 - accuracy: 0.5080
 2272/25000 [=>............................] - ETA: 1:16 - loss: 7.5721 - accuracy: 0.5062
 2304/25000 [=>............................] - ETA: 1:16 - loss: 7.5734 - accuracy: 0.5061
 2336/25000 [=>............................] - ETA: 1:16 - loss: 7.5682 - accuracy: 0.5064
 2368/25000 [=>............................] - ETA: 1:15 - loss: 7.5630 - accuracy: 0.5068
 2400/25000 [=>............................] - ETA: 1:15 - loss: 7.5580 - accuracy: 0.5071
 2432/25000 [=>............................] - ETA: 1:15 - loss: 7.5720 - accuracy: 0.5062
 2464/25000 [=>............................] - ETA: 1:15 - loss: 7.5919 - accuracy: 0.5049
 2496/25000 [=>............................] - ETA: 1:15 - loss: 7.5745 - accuracy: 0.5060
 2528/25000 [==>...........................] - ETA: 1:15 - loss: 7.5453 - accuracy: 0.5079
 2560/25000 [==>...........................] - ETA: 1:15 - loss: 7.5528 - accuracy: 0.5074
 2592/25000 [==>...........................] - ETA: 1:15 - loss: 7.5424 - accuracy: 0.5081
 2624/25000 [==>...........................] - ETA: 1:15 - loss: 7.5381 - accuracy: 0.5084
 2656/25000 [==>...........................] - ETA: 1:15 - loss: 7.5396 - accuracy: 0.5083
 2688/25000 [==>...........................] - ETA: 1:15 - loss: 7.5354 - accuracy: 0.5086
 2720/25000 [==>...........................] - ETA: 1:15 - loss: 7.5370 - accuracy: 0.5085
 2752/25000 [==>...........................] - ETA: 1:14 - loss: 7.5050 - accuracy: 0.5105
 2784/25000 [==>...........................] - ETA: 1:14 - loss: 7.5234 - accuracy: 0.5093
 2816/25000 [==>...........................] - ETA: 1:14 - loss: 7.5033 - accuracy: 0.5107
 2848/25000 [==>...........................] - ETA: 1:14 - loss: 7.4997 - accuracy: 0.5109
 2880/25000 [==>...........................] - ETA: 1:14 - loss: 7.5069 - accuracy: 0.5104
 2912/25000 [==>...........................] - ETA: 1:14 - loss: 7.4823 - accuracy: 0.5120
 2944/25000 [==>...........................] - ETA: 1:14 - loss: 7.4531 - accuracy: 0.5139
 2976/25000 [==>...........................] - ETA: 1:14 - loss: 7.4245 - accuracy: 0.5158
 3008/25000 [==>...........................] - ETA: 1:13 - loss: 7.4066 - accuracy: 0.5170
 3040/25000 [==>...........................] - ETA: 1:13 - loss: 7.4195 - accuracy: 0.5161
 3072/25000 [==>...........................] - ETA: 1:13 - loss: 7.4270 - accuracy: 0.5156
 3104/25000 [==>...........................] - ETA: 1:13 - loss: 7.4542 - accuracy: 0.5139
 3136/25000 [==>...........................] - ETA: 1:13 - loss: 7.4515 - accuracy: 0.5140
 3168/25000 [==>...........................] - ETA: 1:13 - loss: 7.4295 - accuracy: 0.5155
 3200/25000 [==>...........................] - ETA: 1:13 - loss: 7.4222 - accuracy: 0.5159
 3232/25000 [==>...........................] - ETA: 1:12 - loss: 7.4104 - accuracy: 0.5167
 3264/25000 [==>...........................] - ETA: 1:12 - loss: 7.4129 - accuracy: 0.5165
 3296/25000 [==>...........................] - ETA: 1:12 - loss: 7.4387 - accuracy: 0.5149
 3328/25000 [==>...........................] - ETA: 1:12 - loss: 7.4409 - accuracy: 0.5147
 3360/25000 [===>..........................] - ETA: 1:12 - loss: 7.4476 - accuracy: 0.5143
 3392/25000 [===>..........................] - ETA: 1:12 - loss: 7.4406 - accuracy: 0.5147
 3424/25000 [===>..........................] - ETA: 1:12 - loss: 7.4382 - accuracy: 0.5149
 3456/25000 [===>..........................] - ETA: 1:12 - loss: 7.4403 - accuracy: 0.5148
 3488/25000 [===>..........................] - ETA: 1:12 - loss: 7.4556 - accuracy: 0.5138
 3520/25000 [===>..........................] - ETA: 1:11 - loss: 7.4793 - accuracy: 0.5122
 3552/25000 [===>..........................] - ETA: 1:11 - loss: 7.4810 - accuracy: 0.5121
 3584/25000 [===>..........................] - ETA: 1:11 - loss: 7.4741 - accuracy: 0.5126
 3616/25000 [===>..........................] - ETA: 1:11 - loss: 7.4758 - accuracy: 0.5124
 3648/25000 [===>..........................] - ETA: 1:11 - loss: 7.4733 - accuracy: 0.5126
 3680/25000 [===>..........................] - ETA: 1:11 - loss: 7.4833 - accuracy: 0.5120
 3712/25000 [===>..........................] - ETA: 1:11 - loss: 7.4890 - accuracy: 0.5116
 3744/25000 [===>..........................] - ETA: 1:11 - loss: 7.4823 - accuracy: 0.5120
 3776/25000 [===>..........................] - ETA: 1:10 - loss: 7.4758 - accuracy: 0.5124
 3808/25000 [===>..........................] - ETA: 1:10 - loss: 7.4653 - accuracy: 0.5131
 3840/25000 [===>..........................] - ETA: 1:10 - loss: 7.4750 - accuracy: 0.5125
 3872/25000 [===>..........................] - ETA: 1:10 - loss: 7.4845 - accuracy: 0.5119
 3904/25000 [===>..........................] - ETA: 1:10 - loss: 7.4742 - accuracy: 0.5126
 3936/25000 [===>..........................] - ETA: 1:10 - loss: 7.4757 - accuracy: 0.5124
 3968/25000 [===>..........................] - ETA: 1:10 - loss: 7.4889 - accuracy: 0.5116
 4000/25000 [===>..........................] - ETA: 1:09 - loss: 7.4903 - accuracy: 0.5115
 4032/25000 [===>..........................] - ETA: 1:09 - loss: 7.4841 - accuracy: 0.5119
 4064/25000 [===>..........................] - ETA: 1:09 - loss: 7.4742 - accuracy: 0.5125
 4096/25000 [===>..........................] - ETA: 1:09 - loss: 7.4645 - accuracy: 0.5132
 4128/25000 [===>..........................] - ETA: 1:09 - loss: 7.4846 - accuracy: 0.5119
 4160/25000 [===>..........................] - ETA: 1:09 - loss: 7.4897 - accuracy: 0.5115
 4192/25000 [====>.........................] - ETA: 1:09 - loss: 7.4764 - accuracy: 0.5124
 4224/25000 [====>.........................] - ETA: 1:09 - loss: 7.4670 - accuracy: 0.5130
 4256/25000 [====>.........................] - ETA: 1:09 - loss: 7.4685 - accuracy: 0.5129
 4288/25000 [====>.........................] - ETA: 1:08 - loss: 7.4807 - accuracy: 0.5121
 4320/25000 [====>.........................] - ETA: 1:08 - loss: 7.4892 - accuracy: 0.5116
 4352/25000 [====>.........................] - ETA: 1:08 - loss: 7.4834 - accuracy: 0.5119
 4384/25000 [====>.........................] - ETA: 1:08 - loss: 7.4778 - accuracy: 0.5123
 4416/25000 [====>.........................] - ETA: 1:08 - loss: 7.4826 - accuracy: 0.5120
 4448/25000 [====>.........................] - ETA: 1:08 - loss: 7.4943 - accuracy: 0.5112
 4480/25000 [====>.........................] - ETA: 1:08 - loss: 7.4921 - accuracy: 0.5114
 4512/25000 [====>.........................] - ETA: 1:08 - loss: 7.4899 - accuracy: 0.5115
 4544/25000 [====>.........................] - ETA: 1:07 - loss: 7.4777 - accuracy: 0.5123
 4576/25000 [====>.........................] - ETA: 1:07 - loss: 7.4756 - accuracy: 0.5125
 4608/25000 [====>.........................] - ETA: 1:07 - loss: 7.4836 - accuracy: 0.5119
 4640/25000 [====>.........................] - ETA: 1:07 - loss: 7.4948 - accuracy: 0.5112
 4672/25000 [====>.........................] - ETA: 1:07 - loss: 7.4828 - accuracy: 0.5120
 4704/25000 [====>.........................] - ETA: 1:07 - loss: 7.4873 - accuracy: 0.5117
 4736/25000 [====>.........................] - ETA: 1:07 - loss: 7.4886 - accuracy: 0.5116
 4768/25000 [====>.........................] - ETA: 1:07 - loss: 7.4801 - accuracy: 0.5122
 4800/25000 [====>.........................] - ETA: 1:07 - loss: 7.4750 - accuracy: 0.5125
 4832/25000 [====>.........................] - ETA: 1:07 - loss: 7.4762 - accuracy: 0.5124
 4864/25000 [====>.........................] - ETA: 1:06 - loss: 7.4932 - accuracy: 0.5113
 4896/25000 [====>.........................] - ETA: 1:06 - loss: 7.4944 - accuracy: 0.5112
 4928/25000 [====>.........................] - ETA: 1:06 - loss: 7.5017 - accuracy: 0.5108
 4960/25000 [====>.........................] - ETA: 1:06 - loss: 7.4904 - accuracy: 0.5115
 4992/25000 [====>.........................] - ETA: 1:06 - loss: 7.4977 - accuracy: 0.5110
 5024/25000 [=====>........................] - ETA: 1:06 - loss: 7.5201 - accuracy: 0.5096
 5056/25000 [=====>........................] - ETA: 1:06 - loss: 7.5271 - accuracy: 0.5091
 5088/25000 [=====>........................] - ETA: 1:06 - loss: 7.5431 - accuracy: 0.5081
 5120/25000 [=====>........................] - ETA: 1:05 - loss: 7.5289 - accuracy: 0.5090
 5152/25000 [=====>........................] - ETA: 1:05 - loss: 7.5386 - accuracy: 0.5083
 5184/25000 [=====>........................] - ETA: 1:05 - loss: 7.5542 - accuracy: 0.5073
 5216/25000 [=====>........................] - ETA: 1:05 - loss: 7.5579 - accuracy: 0.5071
 5248/25000 [=====>........................] - ETA: 1:05 - loss: 7.5527 - accuracy: 0.5074
 5280/25000 [=====>........................] - ETA: 1:05 - loss: 7.5563 - accuracy: 0.5072
 5312/25000 [=====>........................] - ETA: 1:05 - loss: 7.5425 - accuracy: 0.5081
 5344/25000 [=====>........................] - ETA: 1:05 - loss: 7.5605 - accuracy: 0.5069
 5376/25000 [=====>........................] - ETA: 1:05 - loss: 7.5782 - accuracy: 0.5058
 5408/25000 [=====>........................] - ETA: 1:05 - loss: 7.5702 - accuracy: 0.5063
 5440/25000 [=====>........................] - ETA: 1:04 - loss: 7.5708 - accuracy: 0.5063
 5472/25000 [=====>........................] - ETA: 1:04 - loss: 7.5685 - accuracy: 0.5064
 5504/25000 [=====>........................] - ETA: 1:04 - loss: 7.5886 - accuracy: 0.5051
 5536/25000 [=====>........................] - ETA: 1:04 - loss: 7.5835 - accuracy: 0.5054
 5568/25000 [=====>........................] - ETA: 1:04 - loss: 7.5785 - accuracy: 0.5057
 5600/25000 [=====>........................] - ETA: 1:04 - loss: 7.5763 - accuracy: 0.5059
 5632/25000 [=====>........................] - ETA: 1:04 - loss: 7.5768 - accuracy: 0.5059
 5664/25000 [=====>........................] - ETA: 1:04 - loss: 7.5637 - accuracy: 0.5067
 5696/25000 [=====>........................] - ETA: 1:04 - loss: 7.5589 - accuracy: 0.5070
 5728/25000 [=====>........................] - ETA: 1:04 - loss: 7.5622 - accuracy: 0.5068
 5760/25000 [=====>........................] - ETA: 1:03 - loss: 7.5548 - accuracy: 0.5073
 5792/25000 [=====>........................] - ETA: 1:03 - loss: 7.5554 - accuracy: 0.5073
 5824/25000 [=====>........................] - ETA: 1:03 - loss: 7.5402 - accuracy: 0.5082
 5856/25000 [======>.......................] - ETA: 1:03 - loss: 7.5278 - accuracy: 0.5091
 5888/25000 [======>.......................] - ETA: 1:03 - loss: 7.5364 - accuracy: 0.5085
 5920/25000 [======>.......................] - ETA: 1:03 - loss: 7.5371 - accuracy: 0.5084
 5952/25000 [======>.......................] - ETA: 1:03 - loss: 7.5404 - accuracy: 0.5082
 5984/25000 [======>.......................] - ETA: 1:03 - loss: 7.5411 - accuracy: 0.5082
 6016/25000 [======>.......................] - ETA: 1:03 - loss: 7.5443 - accuracy: 0.5080
 6048/25000 [======>.......................] - ETA: 1:02 - loss: 7.5601 - accuracy: 0.5069
 6080/25000 [======>.......................] - ETA: 1:02 - loss: 7.5607 - accuracy: 0.5069
 6112/25000 [======>.......................] - ETA: 1:02 - loss: 7.5562 - accuracy: 0.5072
 6144/25000 [======>.......................] - ETA: 1:02 - loss: 7.5593 - accuracy: 0.5070
 6176/25000 [======>.......................] - ETA: 1:02 - loss: 7.5723 - accuracy: 0.5062
 6208/25000 [======>.......................] - ETA: 1:02 - loss: 7.5752 - accuracy: 0.5060
 6240/25000 [======>.......................] - ETA: 1:02 - loss: 7.5782 - accuracy: 0.5058
 6272/25000 [======>.......................] - ETA: 1:02 - loss: 7.5835 - accuracy: 0.5054
 6304/25000 [======>.......................] - ETA: 1:02 - loss: 7.5815 - accuracy: 0.5056
 6336/25000 [======>.......................] - ETA: 1:01 - loss: 7.5819 - accuracy: 0.5055
 6368/25000 [======>.......................] - ETA: 1:01 - loss: 7.5848 - accuracy: 0.5053
 6400/25000 [======>.......................] - ETA: 1:01 - loss: 7.5900 - accuracy: 0.5050
 6432/25000 [======>.......................] - ETA: 1:01 - loss: 7.5927 - accuracy: 0.5048
 6464/25000 [======>.......................] - ETA: 1:01 - loss: 7.5955 - accuracy: 0.5046
 6496/25000 [======>.......................] - ETA: 1:01 - loss: 7.5958 - accuracy: 0.5046
 6528/25000 [======>.......................] - ETA: 1:01 - loss: 7.6079 - accuracy: 0.5038
 6560/25000 [======>.......................] - ETA: 1:01 - loss: 7.6129 - accuracy: 0.5035
 6592/25000 [======>.......................] - ETA: 1:01 - loss: 7.6108 - accuracy: 0.5036
 6624/25000 [======>.......................] - ETA: 1:01 - loss: 7.6087 - accuracy: 0.5038
 6656/25000 [======>.......................] - ETA: 1:00 - loss: 7.5998 - accuracy: 0.5044
 6688/25000 [=======>......................] - ETA: 1:00 - loss: 7.6047 - accuracy: 0.5040
 6720/25000 [=======>......................] - ETA: 1:00 - loss: 7.6119 - accuracy: 0.5036
 6752/25000 [=======>......................] - ETA: 1:00 - loss: 7.6121 - accuracy: 0.5036
 6784/25000 [=======>......................] - ETA: 1:00 - loss: 7.6192 - accuracy: 0.5031
 6816/25000 [=======>......................] - ETA: 1:00 - loss: 7.6239 - accuracy: 0.5028
 6848/25000 [=======>......................] - ETA: 1:00 - loss: 7.6263 - accuracy: 0.5026
 6880/25000 [=======>......................] - ETA: 1:00 - loss: 7.6220 - accuracy: 0.5029
 6912/25000 [=======>......................] - ETA: 1:00 - loss: 7.6267 - accuracy: 0.5026
 6944/25000 [=======>......................] - ETA: 59s - loss: 7.6202 - accuracy: 0.5030 
 6976/25000 [=======>......................] - ETA: 59s - loss: 7.6249 - accuracy: 0.5027
 7008/25000 [=======>......................] - ETA: 59s - loss: 7.6229 - accuracy: 0.5029
 7040/25000 [=======>......................] - ETA: 59s - loss: 7.6231 - accuracy: 0.5028
 7072/25000 [=======>......................] - ETA: 59s - loss: 7.6233 - accuracy: 0.5028
 7104/25000 [=======>......................] - ETA: 59s - loss: 7.6278 - accuracy: 0.5025
 7136/25000 [=======>......................] - ETA: 59s - loss: 7.6258 - accuracy: 0.5027
 7168/25000 [=======>......................] - ETA: 59s - loss: 7.6367 - accuracy: 0.5020
 7200/25000 [=======>......................] - ETA: 59s - loss: 7.6240 - accuracy: 0.5028
 7232/25000 [=======>......................] - ETA: 58s - loss: 7.6115 - accuracy: 0.5036
 7264/25000 [=======>......................] - ETA: 58s - loss: 7.6244 - accuracy: 0.5028
 7296/25000 [=======>......................] - ETA: 58s - loss: 7.6351 - accuracy: 0.5021
 7328/25000 [=======>......................] - ETA: 58s - loss: 7.6352 - accuracy: 0.5020
 7360/25000 [=======>......................] - ETA: 58s - loss: 7.6312 - accuracy: 0.5023
 7392/25000 [=======>......................] - ETA: 58s - loss: 7.6376 - accuracy: 0.5019
 7424/25000 [=======>......................] - ETA: 58s - loss: 7.6460 - accuracy: 0.5013
 7456/25000 [=======>......................] - ETA: 58s - loss: 7.6481 - accuracy: 0.5012
 7488/25000 [=======>......................] - ETA: 58s - loss: 7.6441 - accuracy: 0.5015
 7520/25000 [========>.....................] - ETA: 57s - loss: 7.6462 - accuracy: 0.5013
 7552/25000 [========>.....................] - ETA: 57s - loss: 7.6463 - accuracy: 0.5013
 7584/25000 [========>.....................] - ETA: 57s - loss: 7.6444 - accuracy: 0.5015
 7616/25000 [========>.....................] - ETA: 57s - loss: 7.6485 - accuracy: 0.5012
 7648/25000 [========>.....................] - ETA: 57s - loss: 7.6526 - accuracy: 0.5009
 7680/25000 [========>.....................] - ETA: 57s - loss: 7.6506 - accuracy: 0.5010
 7712/25000 [========>.....................] - ETA: 57s - loss: 7.6587 - accuracy: 0.5005
 7744/25000 [========>.....................] - ETA: 57s - loss: 7.6567 - accuracy: 0.5006
 7776/25000 [========>.....................] - ETA: 56s - loss: 7.6508 - accuracy: 0.5010
 7808/25000 [========>.....................] - ETA: 56s - loss: 7.6470 - accuracy: 0.5013
 7840/25000 [========>.....................] - ETA: 56s - loss: 7.6451 - accuracy: 0.5014
 7872/25000 [========>.....................] - ETA: 56s - loss: 7.6432 - accuracy: 0.5015
 7904/25000 [========>.....................] - ETA: 56s - loss: 7.6414 - accuracy: 0.5016
 7936/25000 [========>.....................] - ETA: 56s - loss: 7.6454 - accuracy: 0.5014
 7968/25000 [========>.....................] - ETA: 56s - loss: 7.6416 - accuracy: 0.5016
 8000/25000 [========>.....................] - ETA: 56s - loss: 7.6398 - accuracy: 0.5017
 8032/25000 [========>.....................] - ETA: 56s - loss: 7.6437 - accuracy: 0.5015
 8064/25000 [========>.....................] - ETA: 56s - loss: 7.6438 - accuracy: 0.5015
 8096/25000 [========>.....................] - ETA: 55s - loss: 7.6496 - accuracy: 0.5011
 8128/25000 [========>.....................] - ETA: 55s - loss: 7.6553 - accuracy: 0.5007
 8160/25000 [========>.....................] - ETA: 55s - loss: 7.6497 - accuracy: 0.5011
 8192/25000 [========>.....................] - ETA: 55s - loss: 7.6442 - accuracy: 0.5015
 8224/25000 [========>.....................] - ETA: 55s - loss: 7.6517 - accuracy: 0.5010
 8256/25000 [========>.....................] - ETA: 55s - loss: 7.6425 - accuracy: 0.5016
 8288/25000 [========>.....................] - ETA: 55s - loss: 7.6444 - accuracy: 0.5014
 8320/25000 [========>.....................] - ETA: 55s - loss: 7.6371 - accuracy: 0.5019
 8352/25000 [=========>....................] - ETA: 55s - loss: 7.6372 - accuracy: 0.5019
 8384/25000 [=========>....................] - ETA: 54s - loss: 7.6392 - accuracy: 0.5018
 8416/25000 [=========>....................] - ETA: 54s - loss: 7.6356 - accuracy: 0.5020
 8448/25000 [=========>....................] - ETA: 54s - loss: 7.6394 - accuracy: 0.5018
 8480/25000 [=========>....................] - ETA: 54s - loss: 7.6431 - accuracy: 0.5015
 8512/25000 [=========>....................] - ETA: 54s - loss: 7.6396 - accuracy: 0.5018
 8544/25000 [=========>....................] - ETA: 54s - loss: 7.6469 - accuracy: 0.5013
 8576/25000 [=========>....................] - ETA: 54s - loss: 7.6487 - accuracy: 0.5012
 8608/25000 [=========>....................] - ETA: 54s - loss: 7.6488 - accuracy: 0.5012
 8640/25000 [=========>....................] - ETA: 54s - loss: 7.6471 - accuracy: 0.5013
 8672/25000 [=========>....................] - ETA: 54s - loss: 7.6489 - accuracy: 0.5012
 8704/25000 [=========>....................] - ETA: 53s - loss: 7.6578 - accuracy: 0.5006
 8736/25000 [=========>....................] - ETA: 53s - loss: 7.6473 - accuracy: 0.5013
 8768/25000 [=========>....................] - ETA: 53s - loss: 7.6404 - accuracy: 0.5017
 8800/25000 [=========>....................] - ETA: 53s - loss: 7.6335 - accuracy: 0.5022
 8832/25000 [=========>....................] - ETA: 53s - loss: 7.6354 - accuracy: 0.5020
 8864/25000 [=========>....................] - ETA: 53s - loss: 7.6389 - accuracy: 0.5018
 8896/25000 [=========>....................] - ETA: 53s - loss: 7.6339 - accuracy: 0.5021
 8928/25000 [=========>....................] - ETA: 53s - loss: 7.6340 - accuracy: 0.5021
 8960/25000 [=========>....................] - ETA: 52s - loss: 7.6273 - accuracy: 0.5026
 8992/25000 [=========>....................] - ETA: 52s - loss: 7.6240 - accuracy: 0.5028
 9024/25000 [=========>....................] - ETA: 52s - loss: 7.6224 - accuracy: 0.5029
 9056/25000 [=========>....................] - ETA: 52s - loss: 7.6158 - accuracy: 0.5033
 9088/25000 [=========>....................] - ETA: 52s - loss: 7.6126 - accuracy: 0.5035
 9120/25000 [=========>....................] - ETA: 52s - loss: 7.6179 - accuracy: 0.5032
 9152/25000 [=========>....................] - ETA: 52s - loss: 7.6097 - accuracy: 0.5037
 9184/25000 [==========>...................] - ETA: 52s - loss: 7.6082 - accuracy: 0.5038
 9216/25000 [==========>...................] - ETA: 52s - loss: 7.6101 - accuracy: 0.5037
 9248/25000 [==========>...................] - ETA: 51s - loss: 7.6102 - accuracy: 0.5037
 9280/25000 [==========>...................] - ETA: 51s - loss: 7.6104 - accuracy: 0.5037
 9312/25000 [==========>...................] - ETA: 51s - loss: 7.6139 - accuracy: 0.5034
 9344/25000 [==========>...................] - ETA: 51s - loss: 7.6075 - accuracy: 0.5039
 9376/25000 [==========>...................] - ETA: 51s - loss: 7.6127 - accuracy: 0.5035
 9408/25000 [==========>...................] - ETA: 51s - loss: 7.6194 - accuracy: 0.5031
 9440/25000 [==========>...................] - ETA: 51s - loss: 7.6260 - accuracy: 0.5026
 9472/25000 [==========>...................] - ETA: 51s - loss: 7.6229 - accuracy: 0.5029
 9504/25000 [==========>...................] - ETA: 51s - loss: 7.6214 - accuracy: 0.5029
 9536/25000 [==========>...................] - ETA: 51s - loss: 7.6168 - accuracy: 0.5033
 9568/25000 [==========>...................] - ETA: 50s - loss: 7.6121 - accuracy: 0.5036
 9600/25000 [==========>...................] - ETA: 50s - loss: 7.6123 - accuracy: 0.5035
 9632/25000 [==========>...................] - ETA: 50s - loss: 7.6189 - accuracy: 0.5031
 9664/25000 [==========>...................] - ETA: 50s - loss: 7.6190 - accuracy: 0.5031
 9696/25000 [==========>...................] - ETA: 50s - loss: 7.6160 - accuracy: 0.5033
 9728/25000 [==========>...................] - ETA: 50s - loss: 7.6225 - accuracy: 0.5029
 9760/25000 [==========>...................] - ETA: 50s - loss: 7.6163 - accuracy: 0.5033
 9792/25000 [==========>...................] - ETA: 50s - loss: 7.6275 - accuracy: 0.5026
 9824/25000 [==========>...................] - ETA: 50s - loss: 7.6260 - accuracy: 0.5026
 9856/25000 [==========>...................] - ETA: 49s - loss: 7.6231 - accuracy: 0.5028
 9888/25000 [==========>...................] - ETA: 49s - loss: 7.6279 - accuracy: 0.5025
 9920/25000 [==========>...................] - ETA: 49s - loss: 7.6295 - accuracy: 0.5024
 9952/25000 [==========>...................] - ETA: 49s - loss: 7.6373 - accuracy: 0.5019
 9984/25000 [==========>...................] - ETA: 49s - loss: 7.6359 - accuracy: 0.5020
10016/25000 [===========>..................] - ETA: 49s - loss: 7.6421 - accuracy: 0.5016
10048/25000 [===========>..................] - ETA: 49s - loss: 7.6468 - accuracy: 0.5013
10080/25000 [===========>..................] - ETA: 49s - loss: 7.6423 - accuracy: 0.5016
10112/25000 [===========>..................] - ETA: 49s - loss: 7.6393 - accuracy: 0.5018
10144/25000 [===========>..................] - ETA: 48s - loss: 7.6409 - accuracy: 0.5017
10176/25000 [===========>..................] - ETA: 48s - loss: 7.6440 - accuracy: 0.5015
10208/25000 [===========>..................] - ETA: 48s - loss: 7.6411 - accuracy: 0.5017
10240/25000 [===========>..................] - ETA: 48s - loss: 7.6457 - accuracy: 0.5014
10272/25000 [===========>..................] - ETA: 48s - loss: 7.6472 - accuracy: 0.5013
10304/25000 [===========>..................] - ETA: 48s - loss: 7.6488 - accuracy: 0.5012
10336/25000 [===========>..................] - ETA: 48s - loss: 7.6444 - accuracy: 0.5015
10368/25000 [===========>..................] - ETA: 48s - loss: 7.6504 - accuracy: 0.5011
10400/25000 [===========>..................] - ETA: 48s - loss: 7.6519 - accuracy: 0.5010
10432/25000 [===========>..................] - ETA: 47s - loss: 7.6460 - accuracy: 0.5013
10464/25000 [===========>..................] - ETA: 47s - loss: 7.6534 - accuracy: 0.5009
10496/25000 [===========>..................] - ETA: 47s - loss: 7.6476 - accuracy: 0.5012
10528/25000 [===========>..................] - ETA: 47s - loss: 7.6506 - accuracy: 0.5010
10560/25000 [===========>..................] - ETA: 47s - loss: 7.6536 - accuracy: 0.5009
10592/25000 [===========>..................] - ETA: 47s - loss: 7.6478 - accuracy: 0.5012
10624/25000 [===========>..................] - ETA: 47s - loss: 7.6435 - accuracy: 0.5015
10656/25000 [===========>..................] - ETA: 47s - loss: 7.6407 - accuracy: 0.5017
10688/25000 [===========>..................] - ETA: 47s - loss: 7.6379 - accuracy: 0.5019
10720/25000 [===========>..................] - ETA: 46s - loss: 7.6323 - accuracy: 0.5022
10752/25000 [===========>..................] - ETA: 46s - loss: 7.6267 - accuracy: 0.5026
10784/25000 [===========>..................] - ETA: 46s - loss: 7.6311 - accuracy: 0.5023
10816/25000 [===========>..................] - ETA: 46s - loss: 7.6241 - accuracy: 0.5028
10848/25000 [============>.................] - ETA: 46s - loss: 7.6214 - accuracy: 0.5029
10880/25000 [============>.................] - ETA: 46s - loss: 7.6229 - accuracy: 0.5028
10912/25000 [============>.................] - ETA: 46s - loss: 7.6245 - accuracy: 0.5027
10944/25000 [============>.................] - ETA: 46s - loss: 7.6218 - accuracy: 0.5029
10976/25000 [============>.................] - ETA: 46s - loss: 7.6233 - accuracy: 0.5028
11008/25000 [============>.................] - ETA: 46s - loss: 7.6179 - accuracy: 0.5032
11040/25000 [============>.................] - ETA: 45s - loss: 7.6194 - accuracy: 0.5031
11072/25000 [============>.................] - ETA: 45s - loss: 7.6168 - accuracy: 0.5033
11104/25000 [============>.................] - ETA: 45s - loss: 7.6141 - accuracy: 0.5034
11136/25000 [============>.................] - ETA: 45s - loss: 7.6171 - accuracy: 0.5032
11168/25000 [============>.................] - ETA: 45s - loss: 7.6144 - accuracy: 0.5034
11200/25000 [============>.................] - ETA: 45s - loss: 7.6173 - accuracy: 0.5032
11232/25000 [============>.................] - ETA: 45s - loss: 7.6175 - accuracy: 0.5032
11264/25000 [============>.................] - ETA: 45s - loss: 7.6203 - accuracy: 0.5030
11296/25000 [============>.................] - ETA: 45s - loss: 7.6178 - accuracy: 0.5032
11328/25000 [============>.................] - ETA: 44s - loss: 7.6152 - accuracy: 0.5034
11360/25000 [============>.................] - ETA: 44s - loss: 7.6126 - accuracy: 0.5035
11392/25000 [============>.................] - ETA: 44s - loss: 7.6155 - accuracy: 0.5033
11424/25000 [============>.................] - ETA: 44s - loss: 7.6183 - accuracy: 0.5032
11456/25000 [============>.................] - ETA: 44s - loss: 7.6211 - accuracy: 0.5030
11488/25000 [============>.................] - ETA: 44s - loss: 7.6212 - accuracy: 0.5030
11520/25000 [============>.................] - ETA: 44s - loss: 7.6227 - accuracy: 0.5029
11552/25000 [============>.................] - ETA: 44s - loss: 7.6281 - accuracy: 0.5025
11584/25000 [============>.................] - ETA: 44s - loss: 7.6335 - accuracy: 0.5022
11616/25000 [============>.................] - ETA: 44s - loss: 7.6336 - accuracy: 0.5022
11648/25000 [============>.................] - ETA: 43s - loss: 7.6350 - accuracy: 0.5021
11680/25000 [=============>................] - ETA: 43s - loss: 7.6364 - accuracy: 0.5020
11712/25000 [=============>................] - ETA: 43s - loss: 7.6417 - accuracy: 0.5016
11744/25000 [=============>................] - ETA: 43s - loss: 7.6431 - accuracy: 0.5015
11776/25000 [=============>................] - ETA: 43s - loss: 7.6406 - accuracy: 0.5017
11808/25000 [=============>................] - ETA: 43s - loss: 7.6355 - accuracy: 0.5020
11840/25000 [=============>................] - ETA: 43s - loss: 7.6368 - accuracy: 0.5019
11872/25000 [=============>................] - ETA: 43s - loss: 7.6305 - accuracy: 0.5024
11904/25000 [=============>................] - ETA: 43s - loss: 7.6318 - accuracy: 0.5023
11936/25000 [=============>................] - ETA: 43s - loss: 7.6217 - accuracy: 0.5029
11968/25000 [=============>................] - ETA: 42s - loss: 7.6167 - accuracy: 0.5033
12000/25000 [=============>................] - ETA: 42s - loss: 7.6117 - accuracy: 0.5036
12032/25000 [=============>................] - ETA: 42s - loss: 7.6080 - accuracy: 0.5038
12064/25000 [=============>................] - ETA: 42s - loss: 7.6107 - accuracy: 0.5036
12096/25000 [=============>................] - ETA: 42s - loss: 7.6108 - accuracy: 0.5036
12128/25000 [=============>................] - ETA: 42s - loss: 7.6123 - accuracy: 0.5035
12160/25000 [=============>................] - ETA: 42s - loss: 7.6137 - accuracy: 0.5035
12192/25000 [=============>................] - ETA: 42s - loss: 7.6138 - accuracy: 0.5034
12224/25000 [=============>................] - ETA: 42s - loss: 7.6139 - accuracy: 0.5034
12256/25000 [=============>................] - ETA: 41s - loss: 7.6178 - accuracy: 0.5032
12288/25000 [=============>................] - ETA: 41s - loss: 7.6192 - accuracy: 0.5031
12320/25000 [=============>................] - ETA: 41s - loss: 7.6181 - accuracy: 0.5032
12352/25000 [=============>................] - ETA: 41s - loss: 7.6219 - accuracy: 0.5029
12384/25000 [=============>................] - ETA: 41s - loss: 7.6258 - accuracy: 0.5027
12416/25000 [=============>................] - ETA: 41s - loss: 7.6283 - accuracy: 0.5025
12448/25000 [=============>................] - ETA: 41s - loss: 7.6309 - accuracy: 0.5023
12480/25000 [=============>................] - ETA: 41s - loss: 7.6322 - accuracy: 0.5022
12512/25000 [==============>...............] - ETA: 41s - loss: 7.6299 - accuracy: 0.5024
12544/25000 [==============>...............] - ETA: 41s - loss: 7.6361 - accuracy: 0.5020
12576/25000 [==============>...............] - ETA: 40s - loss: 7.6398 - accuracy: 0.5017
12608/25000 [==============>...............] - ETA: 40s - loss: 7.6399 - accuracy: 0.5017
12640/25000 [==============>...............] - ETA: 40s - loss: 7.6460 - accuracy: 0.5013
12672/25000 [==============>...............] - ETA: 40s - loss: 7.6424 - accuracy: 0.5016
12704/25000 [==============>...............] - ETA: 40s - loss: 7.6328 - accuracy: 0.5022
12736/25000 [==============>...............] - ETA: 40s - loss: 7.6317 - accuracy: 0.5023
12768/25000 [==============>...............] - ETA: 40s - loss: 7.6318 - accuracy: 0.5023
12800/25000 [==============>...............] - ETA: 40s - loss: 7.6343 - accuracy: 0.5021
12832/25000 [==============>...............] - ETA: 40s - loss: 7.6356 - accuracy: 0.5020
12864/25000 [==============>...............] - ETA: 39s - loss: 7.6344 - accuracy: 0.5021
12896/25000 [==============>...............] - ETA: 39s - loss: 7.6369 - accuracy: 0.5019
12928/25000 [==============>...............] - ETA: 39s - loss: 7.6370 - accuracy: 0.5019
12960/25000 [==============>...............] - ETA: 39s - loss: 7.6382 - accuracy: 0.5019
12992/25000 [==============>...............] - ETA: 39s - loss: 7.6336 - accuracy: 0.5022
13024/25000 [==============>...............] - ETA: 39s - loss: 7.6348 - accuracy: 0.5021
13056/25000 [==============>...............] - ETA: 39s - loss: 7.6373 - accuracy: 0.5019
13088/25000 [==============>...............] - ETA: 39s - loss: 7.6455 - accuracy: 0.5014
13120/25000 [==============>...............] - ETA: 39s - loss: 7.6444 - accuracy: 0.5014
13152/25000 [==============>...............] - ETA: 38s - loss: 7.6456 - accuracy: 0.5014
13184/25000 [==============>...............] - ETA: 38s - loss: 7.6457 - accuracy: 0.5014
13216/25000 [==============>...............] - ETA: 38s - loss: 7.6423 - accuracy: 0.5016
13248/25000 [==============>...............] - ETA: 38s - loss: 7.6412 - accuracy: 0.5017
13280/25000 [==============>...............] - ETA: 38s - loss: 7.6458 - accuracy: 0.5014
13312/25000 [==============>...............] - ETA: 38s - loss: 7.6413 - accuracy: 0.5017
13344/25000 [===============>..............] - ETA: 38s - loss: 7.6436 - accuracy: 0.5015
13376/25000 [===============>..............] - ETA: 38s - loss: 7.6414 - accuracy: 0.5016
13408/25000 [===============>..............] - ETA: 38s - loss: 7.6403 - accuracy: 0.5017
13440/25000 [===============>..............] - ETA: 38s - loss: 7.6347 - accuracy: 0.5021
13472/25000 [===============>..............] - ETA: 37s - loss: 7.6325 - accuracy: 0.5022
13504/25000 [===============>..............] - ETA: 37s - loss: 7.6303 - accuracy: 0.5024
13536/25000 [===============>..............] - ETA: 37s - loss: 7.6326 - accuracy: 0.5022
13568/25000 [===============>..............] - ETA: 37s - loss: 7.6327 - accuracy: 0.5022
13600/25000 [===============>..............] - ETA: 37s - loss: 7.6351 - accuracy: 0.5021
13632/25000 [===============>..............] - ETA: 37s - loss: 7.6385 - accuracy: 0.5018
13664/25000 [===============>..............] - ETA: 37s - loss: 7.6408 - accuracy: 0.5017
13696/25000 [===============>..............] - ETA: 37s - loss: 7.6420 - accuracy: 0.5016
13728/25000 [===============>..............] - ETA: 37s - loss: 7.6420 - accuracy: 0.5016
13760/25000 [===============>..............] - ETA: 36s - loss: 7.6466 - accuracy: 0.5013
13792/25000 [===============>..............] - ETA: 36s - loss: 7.6399 - accuracy: 0.5017
13824/25000 [===============>..............] - ETA: 36s - loss: 7.6367 - accuracy: 0.5020
13856/25000 [===============>..............] - ETA: 36s - loss: 7.6367 - accuracy: 0.5019
13888/25000 [===============>..............] - ETA: 36s - loss: 7.6335 - accuracy: 0.5022
13920/25000 [===============>..............] - ETA: 36s - loss: 7.6281 - accuracy: 0.5025
13952/25000 [===============>..............] - ETA: 36s - loss: 7.6304 - accuracy: 0.5024
13984/25000 [===============>..............] - ETA: 36s - loss: 7.6260 - accuracy: 0.5026
14016/25000 [===============>..............] - ETA: 36s - loss: 7.6283 - accuracy: 0.5025
14048/25000 [===============>..............] - ETA: 35s - loss: 7.6251 - accuracy: 0.5027
14080/25000 [===============>..............] - ETA: 35s - loss: 7.6231 - accuracy: 0.5028
14112/25000 [===============>..............] - ETA: 35s - loss: 7.6275 - accuracy: 0.5026
14144/25000 [===============>..............] - ETA: 35s - loss: 7.6287 - accuracy: 0.5025
14176/25000 [================>.............] - ETA: 35s - loss: 7.6298 - accuracy: 0.5024
14208/25000 [================>.............] - ETA: 35s - loss: 7.6278 - accuracy: 0.5025
14240/25000 [================>.............] - ETA: 35s - loss: 7.6279 - accuracy: 0.5025
14272/25000 [================>.............] - ETA: 35s - loss: 7.6269 - accuracy: 0.5026
14304/25000 [================>.............] - ETA: 35s - loss: 7.6259 - accuracy: 0.5027
14336/25000 [================>.............] - ETA: 34s - loss: 7.6270 - accuracy: 0.5026
14368/25000 [================>.............] - ETA: 34s - loss: 7.6314 - accuracy: 0.5023
14400/25000 [================>.............] - ETA: 34s - loss: 7.6379 - accuracy: 0.5019
14432/25000 [================>.............] - ETA: 34s - loss: 7.6379 - accuracy: 0.5019
14464/25000 [================>.............] - ETA: 34s - loss: 7.6391 - accuracy: 0.5018
14496/25000 [================>.............] - ETA: 34s - loss: 7.6402 - accuracy: 0.5017
14528/25000 [================>.............] - ETA: 34s - loss: 7.6371 - accuracy: 0.5019
14560/25000 [================>.............] - ETA: 34s - loss: 7.6382 - accuracy: 0.5019
14592/25000 [================>.............] - ETA: 34s - loss: 7.6372 - accuracy: 0.5019
14624/25000 [================>.............] - ETA: 34s - loss: 7.6425 - accuracy: 0.5016
14656/25000 [================>.............] - ETA: 33s - loss: 7.6426 - accuracy: 0.5016
14688/25000 [================>.............] - ETA: 33s - loss: 7.6468 - accuracy: 0.5013
14720/25000 [================>.............] - ETA: 33s - loss: 7.6427 - accuracy: 0.5016
14752/25000 [================>.............] - ETA: 33s - loss: 7.6458 - accuracy: 0.5014
14784/25000 [================>.............] - ETA: 33s - loss: 7.6448 - accuracy: 0.5014
14816/25000 [================>.............] - ETA: 33s - loss: 7.6501 - accuracy: 0.5011
14848/25000 [================>.............] - ETA: 33s - loss: 7.6542 - accuracy: 0.5008
14880/25000 [================>.............] - ETA: 33s - loss: 7.6543 - accuracy: 0.5008
14912/25000 [================>.............] - ETA: 33s - loss: 7.6543 - accuracy: 0.5008
14944/25000 [================>.............] - ETA: 32s - loss: 7.6553 - accuracy: 0.5007
14976/25000 [================>.............] - ETA: 32s - loss: 7.6554 - accuracy: 0.5007
15008/25000 [=================>............] - ETA: 32s - loss: 7.6544 - accuracy: 0.5008
15040/25000 [=================>............] - ETA: 32s - loss: 7.6564 - accuracy: 0.5007
15072/25000 [=================>............] - ETA: 32s - loss: 7.6524 - accuracy: 0.5009
15104/25000 [=================>............] - ETA: 32s - loss: 7.6534 - accuracy: 0.5009
15136/25000 [=================>............] - ETA: 32s - loss: 7.6524 - accuracy: 0.5009
15168/25000 [=================>............] - ETA: 32s - loss: 7.6525 - accuracy: 0.5009
15200/25000 [=================>............] - ETA: 32s - loss: 7.6535 - accuracy: 0.5009
15232/25000 [=================>............] - ETA: 32s - loss: 7.6576 - accuracy: 0.5006
15264/25000 [=================>............] - ETA: 31s - loss: 7.6606 - accuracy: 0.5004
15296/25000 [=================>............] - ETA: 31s - loss: 7.6606 - accuracy: 0.5004
15328/25000 [=================>............] - ETA: 31s - loss: 7.6626 - accuracy: 0.5003
15360/25000 [=================>............] - ETA: 31s - loss: 7.6606 - accuracy: 0.5004
15392/25000 [=================>............] - ETA: 31s - loss: 7.6626 - accuracy: 0.5003
15424/25000 [=================>............] - ETA: 31s - loss: 7.6646 - accuracy: 0.5001
15456/25000 [=================>............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
15488/25000 [=================>............] - ETA: 31s - loss: 7.6627 - accuracy: 0.5003
15520/25000 [=================>............] - ETA: 31s - loss: 7.6587 - accuracy: 0.5005
15552/25000 [=================>............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
15584/25000 [=================>............] - ETA: 30s - loss: 7.6647 - accuracy: 0.5001
15616/25000 [=================>............] - ETA: 30s - loss: 7.6627 - accuracy: 0.5003
15648/25000 [=================>............] - ETA: 30s - loss: 7.6607 - accuracy: 0.5004
15680/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15712/25000 [=================>............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
15744/25000 [=================>............] - ETA: 30s - loss: 7.6686 - accuracy: 0.4999
15776/25000 [=================>............] - ETA: 30s - loss: 7.6725 - accuracy: 0.4996
15808/25000 [=================>............] - ETA: 30s - loss: 7.6724 - accuracy: 0.4996
15840/25000 [==================>...........] - ETA: 30s - loss: 7.6724 - accuracy: 0.4996
15872/25000 [==================>...........] - ETA: 29s - loss: 7.6743 - accuracy: 0.4995
15904/25000 [==================>...........] - ETA: 29s - loss: 7.6695 - accuracy: 0.4998
15936/25000 [==================>...........] - ETA: 29s - loss: 7.6685 - accuracy: 0.4999
15968/25000 [==================>...........] - ETA: 29s - loss: 7.6676 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 29s - loss: 7.6657 - accuracy: 0.5001
16032/25000 [==================>...........] - ETA: 29s - loss: 7.6657 - accuracy: 0.5001
16064/25000 [==================>...........] - ETA: 29s - loss: 7.6657 - accuracy: 0.5001
16096/25000 [==================>...........] - ETA: 29s - loss: 7.6628 - accuracy: 0.5002
16128/25000 [==================>...........] - ETA: 29s - loss: 7.6638 - accuracy: 0.5002
16160/25000 [==================>...........] - ETA: 28s - loss: 7.6581 - accuracy: 0.5006
16192/25000 [==================>...........] - ETA: 28s - loss: 7.6590 - accuracy: 0.5005
16224/25000 [==================>...........] - ETA: 28s - loss: 7.6543 - accuracy: 0.5008
16256/25000 [==================>...........] - ETA: 28s - loss: 7.6525 - accuracy: 0.5009
16288/25000 [==================>...........] - ETA: 28s - loss: 7.6553 - accuracy: 0.5007
16320/25000 [==================>...........] - ETA: 28s - loss: 7.6600 - accuracy: 0.5004
16352/25000 [==================>...........] - ETA: 28s - loss: 7.6544 - accuracy: 0.5008
16384/25000 [==================>...........] - ETA: 28s - loss: 7.6573 - accuracy: 0.5006
16416/25000 [==================>...........] - ETA: 28s - loss: 7.6582 - accuracy: 0.5005
16448/25000 [==================>...........] - ETA: 28s - loss: 7.6592 - accuracy: 0.5005
16480/25000 [==================>...........] - ETA: 27s - loss: 7.6601 - accuracy: 0.5004
16512/25000 [==================>...........] - ETA: 27s - loss: 7.6610 - accuracy: 0.5004
16544/25000 [==================>...........] - ETA: 27s - loss: 7.6601 - accuracy: 0.5004
16576/25000 [==================>...........] - ETA: 27s - loss: 7.6564 - accuracy: 0.5007
16608/25000 [==================>...........] - ETA: 27s - loss: 7.6574 - accuracy: 0.5006
16640/25000 [==================>...........] - ETA: 27s - loss: 7.6565 - accuracy: 0.5007
16672/25000 [===================>..........] - ETA: 27s - loss: 7.6583 - accuracy: 0.5005
16704/25000 [===================>..........] - ETA: 27s - loss: 7.6565 - accuracy: 0.5007
16736/25000 [===================>..........] - ETA: 27s - loss: 7.6565 - accuracy: 0.5007
16768/25000 [===================>..........] - ETA: 26s - loss: 7.6556 - accuracy: 0.5007
16800/25000 [===================>..........] - ETA: 26s - loss: 7.6593 - accuracy: 0.5005
16832/25000 [===================>..........] - ETA: 26s - loss: 7.6639 - accuracy: 0.5002
16864/25000 [===================>..........] - ETA: 26s - loss: 7.6657 - accuracy: 0.5001
16896/25000 [===================>..........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
16928/25000 [===================>..........] - ETA: 26s - loss: 7.6693 - accuracy: 0.4998
16960/25000 [===================>..........] - ETA: 26s - loss: 7.6630 - accuracy: 0.5002
16992/25000 [===================>..........] - ETA: 26s - loss: 7.6621 - accuracy: 0.5003
17024/25000 [===================>..........] - ETA: 26s - loss: 7.6612 - accuracy: 0.5004
17056/25000 [===================>..........] - ETA: 26s - loss: 7.6612 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 25s - loss: 7.6612 - accuracy: 0.5004
17120/25000 [===================>..........] - ETA: 25s - loss: 7.6568 - accuracy: 0.5006
17152/25000 [===================>..........] - ETA: 25s - loss: 7.6577 - accuracy: 0.5006
17184/25000 [===================>..........] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
17216/25000 [===================>..........] - ETA: 25s - loss: 7.6604 - accuracy: 0.5004
17248/25000 [===================>..........] - ETA: 25s - loss: 7.6604 - accuracy: 0.5004
17280/25000 [===================>..........] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
17312/25000 [===================>..........] - ETA: 25s - loss: 7.6640 - accuracy: 0.5002
17344/25000 [===================>..........] - ETA: 25s - loss: 7.6657 - accuracy: 0.5001
17376/25000 [===================>..........] - ETA: 25s - loss: 7.6675 - accuracy: 0.4999
17408/25000 [===================>..........] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6693 - accuracy: 0.4998
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6719 - accuracy: 0.4997
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6727 - accuracy: 0.4996
17568/25000 [====================>.........] - ETA: 24s - loss: 7.6710 - accuracy: 0.4997
17600/25000 [====================>.........] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
17632/25000 [====================>.........] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
17664/25000 [====================>.........] - ETA: 24s - loss: 7.6718 - accuracy: 0.4997
17696/25000 [====================>.........] - ETA: 23s - loss: 7.6710 - accuracy: 0.4997
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6701 - accuracy: 0.4998
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6683 - accuracy: 0.4999
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6752 - accuracy: 0.4994
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6769 - accuracy: 0.4993
17888/25000 [====================>.........] - ETA: 23s - loss: 7.6743 - accuracy: 0.4995
17920/25000 [====================>.........] - ETA: 23s - loss: 7.6743 - accuracy: 0.4995
17952/25000 [====================>.........] - ETA: 23s - loss: 7.6752 - accuracy: 0.4994
17984/25000 [====================>.........] - ETA: 23s - loss: 7.6726 - accuracy: 0.4996
18016/25000 [====================>.........] - ETA: 22s - loss: 7.6692 - accuracy: 0.4998
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6717 - accuracy: 0.4997
18176/25000 [====================>.........] - ETA: 22s - loss: 7.6717 - accuracy: 0.4997
18208/25000 [====================>.........] - ETA: 22s - loss: 7.6767 - accuracy: 0.4993
18240/25000 [====================>.........] - ETA: 22s - loss: 7.6767 - accuracy: 0.4993
18272/25000 [====================>.........] - ETA: 22s - loss: 7.6767 - accuracy: 0.4993
18304/25000 [====================>.........] - ETA: 21s - loss: 7.6742 - accuracy: 0.4995
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6741 - accuracy: 0.4995
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6700 - accuracy: 0.4998
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6716 - accuracy: 0.4997
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6733 - accuracy: 0.4996
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6733 - accuracy: 0.4996
18496/25000 [=====================>........] - ETA: 21s - loss: 7.6757 - accuracy: 0.4994
18528/25000 [=====================>........] - ETA: 21s - loss: 7.6782 - accuracy: 0.4992
18560/25000 [=====================>........] - ETA: 21s - loss: 7.6774 - accuracy: 0.4993
18592/25000 [=====================>........] - ETA: 21s - loss: 7.6773 - accuracy: 0.4993
18624/25000 [=====================>........] - ETA: 20s - loss: 7.6790 - accuracy: 0.4992
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6789 - accuracy: 0.4992
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6797 - accuracy: 0.4991
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6773 - accuracy: 0.4993
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6772 - accuracy: 0.4993
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6797 - accuracy: 0.4991
18816/25000 [=====================>........] - ETA: 20s - loss: 7.6829 - accuracy: 0.4989
18848/25000 [=====================>........] - ETA: 20s - loss: 7.6788 - accuracy: 0.4992
18880/25000 [=====================>........] - ETA: 20s - loss: 7.6772 - accuracy: 0.4993
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6747 - accuracy: 0.4995
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6739 - accuracy: 0.4995
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6739 - accuracy: 0.4995
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6698 - accuracy: 0.4998
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6723 - accuracy: 0.4996
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6706 - accuracy: 0.4997
19104/25000 [=====================>........] - ETA: 19s - loss: 7.6698 - accuracy: 0.4998
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6682 - accuracy: 0.4999
19168/25000 [======================>.......] - ETA: 19s - loss: 7.6682 - accuracy: 0.4999
19200/25000 [======================>.......] - ETA: 19s - loss: 7.6722 - accuracy: 0.4996
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6722 - accuracy: 0.4996
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6698 - accuracy: 0.4998
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6706 - accuracy: 0.4997
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6714 - accuracy: 0.4997
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6729 - accuracy: 0.4996
19456/25000 [======================>.......] - ETA: 18s - loss: 7.6753 - accuracy: 0.4994
19488/25000 [======================>.......] - ETA: 18s - loss: 7.6776 - accuracy: 0.4993
19520/25000 [======================>.......] - ETA: 18s - loss: 7.6745 - accuracy: 0.4995
19552/25000 [======================>.......] - ETA: 17s - loss: 7.6752 - accuracy: 0.4994
19584/25000 [======================>.......] - ETA: 17s - loss: 7.6768 - accuracy: 0.4993
19616/25000 [======================>.......] - ETA: 17s - loss: 7.6760 - accuracy: 0.4994
19648/25000 [======================>.......] - ETA: 17s - loss: 7.6775 - accuracy: 0.4993
19680/25000 [======================>.......] - ETA: 17s - loss: 7.6767 - accuracy: 0.4993
19712/25000 [======================>.......] - ETA: 17s - loss: 7.6783 - accuracy: 0.4992
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6814 - accuracy: 0.4990
19776/25000 [======================>.......] - ETA: 17s - loss: 7.6814 - accuracy: 0.4990
19808/25000 [======================>.......] - ETA: 17s - loss: 7.6821 - accuracy: 0.4990
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6798 - accuracy: 0.4991
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6782 - accuracy: 0.4992
19904/25000 [======================>.......] - ETA: 16s - loss: 7.6766 - accuracy: 0.4993
19936/25000 [======================>.......] - ETA: 16s - loss: 7.6751 - accuracy: 0.4994
19968/25000 [======================>.......] - ETA: 16s - loss: 7.6751 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 16s - loss: 7.6728 - accuracy: 0.4996
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6735 - accuracy: 0.4996
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6743 - accuracy: 0.4995
20096/25000 [=======================>......] - ETA: 16s - loss: 7.6742 - accuracy: 0.4995
20128/25000 [=======================>......] - ETA: 16s - loss: 7.6720 - accuracy: 0.4997
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6735 - accuracy: 0.4996
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6742 - accuracy: 0.4995
20224/25000 [=======================>......] - ETA: 15s - loss: 7.6734 - accuracy: 0.4996
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6727 - accuracy: 0.4996
20288/25000 [=======================>......] - ETA: 15s - loss: 7.6727 - accuracy: 0.4996
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6749 - accuracy: 0.4995
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6772 - accuracy: 0.4993
20384/25000 [=======================>......] - ETA: 15s - loss: 7.6772 - accuracy: 0.4993
20416/25000 [=======================>......] - ETA: 15s - loss: 7.6741 - accuracy: 0.4995
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6764 - accuracy: 0.4994
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6764 - accuracy: 0.4994
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6793 - accuracy: 0.4992
20544/25000 [=======================>......] - ETA: 14s - loss: 7.6786 - accuracy: 0.4992
20576/25000 [=======================>......] - ETA: 14s - loss: 7.6808 - accuracy: 0.4991
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6815 - accuracy: 0.4990
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6792 - accuracy: 0.4992
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6822 - accuracy: 0.4990
20704/25000 [=======================>......] - ETA: 14s - loss: 7.6822 - accuracy: 0.4990
20736/25000 [=======================>......] - ETA: 14s - loss: 7.6821 - accuracy: 0.4990
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6821 - accuracy: 0.4990
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6806 - accuracy: 0.4991
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6791 - accuracy: 0.4992
20864/25000 [========================>.....] - ETA: 13s - loss: 7.6784 - accuracy: 0.4992
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6813 - accuracy: 0.4990
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6827 - accuracy: 0.4989
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6871 - accuracy: 0.4987
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6871 - accuracy: 0.4987
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6892 - accuracy: 0.4985
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6885 - accuracy: 0.4986
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6884 - accuracy: 0.4986
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6862 - accuracy: 0.4987
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6840 - accuracy: 0.4989
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6869 - accuracy: 0.4987
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6883 - accuracy: 0.4986
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6897 - accuracy: 0.4985
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6854 - accuracy: 0.4988
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6904 - accuracy: 0.4985
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6954 - accuracy: 0.4981
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6960 - accuracy: 0.4981
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6996 - accuracy: 0.4979
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6981 - accuracy: 0.4979
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6973 - accuracy: 0.4980
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6923 - accuracy: 0.4983
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6887 - accuracy: 0.4986
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6915 - accuracy: 0.4984
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6879 - accuracy: 0.4986
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6850 - accuracy: 0.4988
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6857 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6878 - accuracy: 0.4986
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6871 - accuracy: 0.4987
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6856 - accuracy: 0.4988
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6884 - accuracy: 0.4986
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6898 - accuracy: 0.4985
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6891 - accuracy: 0.4985
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6876 - accuracy: 0.4986
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6918 - accuracy: 0.4984
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6918 - accuracy: 0.4984
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6938 - accuracy: 0.4982 
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6931 - accuracy: 0.4983
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6924 - accuracy: 0.4983
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6923 - accuracy: 0.4983
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6895 - accuracy: 0.4985
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6909 - accuracy: 0.4984
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6901 - accuracy: 0.4985
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6894 - accuracy: 0.4985
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6894 - accuracy: 0.4985
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6921 - accuracy: 0.4983
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6927 - accuracy: 0.4983
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6920 - accuracy: 0.4983
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6913 - accuracy: 0.4984
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6899 - accuracy: 0.4985
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6899 - accuracy: 0.4985
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6891 - accuracy: 0.4985
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6884 - accuracy: 0.4986
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6877 - accuracy: 0.4986
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6870 - accuracy: 0.4987
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6849 - accuracy: 0.4988
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6842 - accuracy: 0.4989
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6828 - accuracy: 0.4989
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6842 - accuracy: 0.4989
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6848 - accuracy: 0.4988
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6841 - accuracy: 0.4989
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6854 - accuracy: 0.4988
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6867 - accuracy: 0.4987
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6873 - accuracy: 0.4986
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6866 - accuracy: 0.4987
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6873 - accuracy: 0.4987
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6853 - accuracy: 0.4988
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6859 - accuracy: 0.4987
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6879 - accuracy: 0.4986
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6872 - accuracy: 0.4987
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6885 - accuracy: 0.4986
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6858 - accuracy: 0.4988
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6838 - accuracy: 0.4989
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6824 - accuracy: 0.4990
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6811 - accuracy: 0.4991
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6831 - accuracy: 0.4989
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6843 - accuracy: 0.4988
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6830 - accuracy: 0.4989
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6830 - accuracy: 0.4989
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6843 - accuracy: 0.4988
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6849 - accuracy: 0.4988
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6849 - accuracy: 0.4988
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6888 - accuracy: 0.4986
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6913 - accuracy: 0.4984
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6932 - accuracy: 0.4983
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6945 - accuracy: 0.4982
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6945 - accuracy: 0.4982
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6944 - accuracy: 0.4982
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6931 - accuracy: 0.4983
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6944 - accuracy: 0.4982
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6956 - accuracy: 0.4981
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6936 - accuracy: 0.4982
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6878 - accuracy: 0.4986
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6859 - accuracy: 0.4987
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6801 - accuracy: 0.4991
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6794 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6813 - accuracy: 0.4990
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6807 - accuracy: 0.4991
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6800 - accuracy: 0.4991
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6813 - accuracy: 0.4990
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6800 - accuracy: 0.4991
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6761 - accuracy: 0.4994
24192/25000 [============================>.] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24224/25000 [============================>.] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24256/25000 [============================>.] - ETA: 2s - loss: 7.6774 - accuracy: 0.4993
24288/25000 [============================>.] - ETA: 2s - loss: 7.6767 - accuracy: 0.4993
24320/25000 [============================>.] - ETA: 2s - loss: 7.6767 - accuracy: 0.4993
24352/25000 [============================>.] - ETA: 2s - loss: 7.6780 - accuracy: 0.4993
24384/25000 [============================>.] - ETA: 2s - loss: 7.6779 - accuracy: 0.4993
24416/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24480/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24512/25000 [============================>.] - ETA: 1s - loss: 7.6791 - accuracy: 0.4992
24544/25000 [============================>.] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
24576/25000 [============================>.] - ETA: 1s - loss: 7.6778 - accuracy: 0.4993
24608/25000 [============================>.] - ETA: 1s - loss: 7.6778 - accuracy: 0.4993
24640/25000 [============================>.] - ETA: 1s - loss: 7.6772 - accuracy: 0.4993
24672/25000 [============================>.] - ETA: 1s - loss: 7.6772 - accuracy: 0.4993
24704/25000 [============================>.] - ETA: 0s - loss: 7.6772 - accuracy: 0.4993
24736/25000 [============================>.] - ETA: 0s - loss: 7.6759 - accuracy: 0.4994
24768/25000 [============================>.] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24800/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24832/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24864/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24896/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 99s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
