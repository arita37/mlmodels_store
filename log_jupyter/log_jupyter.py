
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
 40%|████      | 2/5 [00:51<01:16, 25.58s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.18687690343593535, 'embedding_size_factor': 1.4347369369301717, 'layers.choice': 3, 'learning_rate': 0.001581500013612377, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 7.299775256338126e-10} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc7\xeb\x95\x16QJ\xb5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xf4\xae\xb7\xe7\xa7 X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?Y\xe9J\xb5\x92\xdc\x8bX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\t\x14\xf3A\x97i\xdfu.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc7\xeb\x95\x16QJ\xb5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xf4\xae\xb7\xe7\xa7 X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?Y\xe9J\xb5\x92\xdc\x8bX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\t\x14\xf3A\x97i\xdfu.' and reward: 0.3894
 60%|██████    | 3/5 [01:46<01:08, 34.40s/it] 60%|██████    | 3/5 [01:46<01:10, 35.38s/it]
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.06215024564645312, 'embedding_size_factor': 0.9365509852220506, 'layers.choice': 2, 'learning_rate': 0.0002773543730826957, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.00040788006323478487} and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xaf\xd2(0\xfa\xd1\xe2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xf89\xc5\x92\x16\xd4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?2-;\xf68\x0c\xe5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?:\xbb\x17\x88C\xc0\x81u.' and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xaf\xd2(0\xfa\xd1\xe2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xf89\xc5\x92\x16\xd4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?2-;\xf68\x0c\xe5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?:\xbb\x17\x88C\xc0\x81u.' and reward: 0.3844
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 158.1772482395172
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 1, 'dropout_prob': 0.18687690343593535, 'embedding_size_factor': 1.4347369369301717, 'layers.choice': 3, 'learning_rate': 0.001581500013612377, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 7.299775256338126e-10}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -40.72s of remaining time.
Ensemble size: 21
Ensemble weights: 
[0.61904762 0.         0.38095238]
	0.3952	 = Validation accuracy score
	1.04s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 161.8s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f435968f978> 

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
 [-0.02961443  0.03927916  0.06843434  0.11651243 -0.0186705  -0.05423841]
 [ 0.00121621  0.15237202  0.08455319  0.03544518  0.01486084  0.08430751]
 [ 0.06816624  0.12905861  0.05586601  0.06343503 -0.09872907 -0.21084364]
 [ 0.10576767  0.16981459  0.11736294 -0.20470707  0.14095576  0.36997426]
 [ 0.06350116  0.01124084  0.07289789 -0.47264779  0.11249485  0.0202719 ]
 [ 0.16559057 -0.23143665  0.10283593  0.28736278  0.97745305  0.19303444]
 [ 0.23886891 -0.18105844 -0.14508519 -0.70668513  0.8611958  -0.02046559]
 [-0.08808953 -0.0909259  -0.10531657 -0.04807606  0.26988047  0.14073224]
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
{'loss': 0.5083150062710047, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 10:16:20.560931: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.45720234513282776, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 10:16:21.675646: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 2:10
   24576/17464789 [..............................] - ETA: 1:27
   40960/17464789 [..............................] - ETA: 1:18
   57344/17464789 [..............................] - ETA: 1:14
   73728/17464789 [..............................] - ETA: 1:12
   90112/17464789 [..............................] - ETA: 1:11
  122880/17464789 [..............................] - ETA: 1:00
  139264/17464789 [..............................] - ETA: 1:01
  180224/17464789 [..............................] - ETA: 53s 
  196608/17464789 [..............................] - ETA: 54s
  212992/17464789 [..............................] - ETA: 54s
  245760/17464789 [..............................] - ETA: 51s
  262144/17464789 [..............................] - ETA: 52s
  303104/17464789 [..............................] - ETA: 48s
  319488/17464789 [..............................] - ETA: 49s
  352256/17464789 [..............................] - ETA: 47s
  368640/17464789 [..............................] - ETA: 48s
  401408/17464789 [..............................] - ETA: 47s
  442368/17464789 [..............................] - ETA: 45s
  458752/17464789 [..............................] - ETA: 45s
  491520/17464789 [..............................] - ETA: 44s
  524288/17464789 [..............................] - ETA: 43s
  557056/17464789 [..............................] - ETA: 43s
  598016/17464789 [>.............................] - ETA: 41s
  614400/17464789 [>.............................] - ETA: 42s
  647168/17464789 [>.............................] - ETA: 41s
  679936/17464789 [>.............................] - ETA: 41s
  720896/17464789 [>.............................] - ETA: 40s
  753664/17464789 [>.............................] - ETA: 39s
  786432/17464789 [>.............................] - ETA: 39s
  819200/17464789 [>.............................] - ETA: 38s
  876544/17464789 [>.............................] - ETA: 37s
  909312/17464789 [>.............................] - ETA: 37s
  942080/17464789 [>.............................] - ETA: 36s
  974848/17464789 [>.............................] - ETA: 36s
 1032192/17464789 [>.............................] - ETA: 35s
 1064960/17464789 [>.............................] - ETA: 35s
 1097728/17464789 [>.............................] - ETA: 34s
 1155072/17464789 [>.............................] - ETA: 33s
 1187840/17464789 [=>............................] - ETA: 33s
 1220608/17464789 [=>............................] - ETA: 33s
 1277952/17464789 [=>............................] - ETA: 32s
 1310720/17464789 [=>............................] - ETA: 32s
 1359872/17464789 [=>............................] - ETA: 32s
 1409024/17464789 [=>............................] - ETA: 31s
 1449984/17464789 [=>............................] - ETA: 31s
 1499136/17464789 [=>............................] - ETA: 30s
 1531904/17464789 [=>............................] - ETA: 30s
 1589248/17464789 [=>............................] - ETA: 30s
 1638400/17464789 [=>............................] - ETA: 29s
 1695744/17464789 [=>............................] - ETA: 29s
 1744896/17464789 [=>............................] - ETA: 28s
 1794048/17464789 [==>...........................] - ETA: 28s
 1851392/17464789 [==>...........................] - ETA: 28s
 1900544/17464789 [==>...........................] - ETA: 27s
 1949696/17464789 [==>...........................] - ETA: 27s
 2007040/17464789 [==>...........................] - ETA: 27s
 2056192/17464789 [==>...........................] - ETA: 26s
 2113536/17464789 [==>...........................] - ETA: 26s
 2162688/17464789 [==>...........................] - ETA: 26s
 2211840/17464789 [==>...........................] - ETA: 25s
 2285568/17464789 [==>...........................] - ETA: 25s
 2334720/17464789 [===>..........................] - ETA: 25s
 2392064/17464789 [===>..........................] - ETA: 24s
 2457600/17464789 [===>..........................] - ETA: 24s
 2506752/17464789 [===>..........................] - ETA: 24s
 2580480/17464789 [===>..........................] - ETA: 23s
 2629632/17464789 [===>..........................] - ETA: 23s
 2703360/17464789 [===>..........................] - ETA: 23s
 2752512/17464789 [===>..........................] - ETA: 23s
 2826240/17464789 [===>..........................] - ETA: 22s
 2875392/17464789 [===>..........................] - ETA: 22s
 2949120/17464789 [====>.........................] - ETA: 22s
 3014656/17464789 [====>.........................] - ETA: 21s
 3063808/17464789 [====>.........................] - ETA: 21s
 3137536/17464789 [====>.........................] - ETA: 21s
 3203072/17464789 [====>.........................] - ETA: 21s
 3276800/17464789 [====>.........................] - ETA: 20s
 3342336/17464789 [====>.........................] - ETA: 20s
 3416064/17464789 [====>.........................] - ETA: 20s
 3481600/17464789 [====>.........................] - ETA: 20s
 3555328/17464789 [=====>........................] - ETA: 19s
 3620864/17464789 [=====>........................] - ETA: 19s
 3694592/17464789 [=====>........................] - ETA: 19s
 3760128/17464789 [=====>........................] - ETA: 19s
 3833856/17464789 [=====>........................] - ETA: 18s
 3899392/17464789 [=====>........................] - ETA: 18s
 3989504/17464789 [=====>........................] - ETA: 18s
 4063232/17464789 [=====>........................] - ETA: 18s
 4145152/17464789 [======>.......................] - ETA: 17s
 4218880/17464789 [======>.......................] - ETA: 17s
 4284416/17464789 [======>.......................] - ETA: 17s
 4341760/17464789 [======>.......................] - ETA: 17s
 4390912/17464789 [======>.......................] - ETA: 17s
 4481024/17464789 [======>.......................] - ETA: 16s
 4562944/17464789 [======>.......................] - ETA: 16s
 4636672/17464789 [======>.......................] - ETA: 16s
 4718592/17464789 [=======>......................] - ETA: 16s
 4792320/17464789 [=======>......................] - ETA: 16s
 4874240/17464789 [=======>......................] - ETA: 15s
 4964352/17464789 [=======>......................] - ETA: 15s
 5038080/17464789 [=======>......................] - ETA: 15s
 5120000/17464789 [=======>......................] - ETA: 15s
 5210112/17464789 [=======>......................] - ETA: 15s
 5292032/17464789 [========>.....................] - ETA: 14s
 5382144/17464789 [========>.....................] - ETA: 14s
 5472256/17464789 [========>.....................] - ETA: 14s
 5554176/17464789 [========>.....................] - ETA: 14s
 5644288/17464789 [========>.....................] - ETA: 14s
 5734400/17464789 [========>.....................] - ETA: 13s
 5816320/17464789 [========>.....................] - ETA: 13s
 5922816/17464789 [=========>....................] - ETA: 13s
 6012928/17464789 [=========>....................] - ETA: 13s
 6094848/17464789 [=========>....................] - ETA: 13s
 6184960/17464789 [=========>....................] - ETA: 12s
 6291456/17464789 [=========>....................] - ETA: 12s
 6373376/17464789 [=========>....................] - ETA: 12s
 6479872/17464789 [==========>...................] - ETA: 12s
 6569984/17464789 [==========>...................] - ETA: 12s
 6668288/17464789 [==========>...................] - ETA: 11s
 6758400/17464789 [==========>...................] - ETA: 11s
 6864896/17464789 [==========>...................] - ETA: 11s
 6946816/17464789 [==========>...................] - ETA: 11s
 7053312/17464789 [===========>..................] - ETA: 11s
 7159808/17464789 [===========>..................] - ETA: 11s
 7241728/17464789 [===========>..................] - ETA: 10s
 7348224/17464789 [===========>..................] - ETA: 10s
 7454720/17464789 [===========>..................] - ETA: 10s
 7561216/17464789 [===========>..................] - ETA: 10s
 7684096/17464789 [============>.................] - ETA: 10s
 7782400/17464789 [============>.................] - ETA: 10s
 7905280/17464789 [============>.................] - ETA: 9s 
 8028160/17464789 [============>.................] - ETA: 9s
 8151040/17464789 [=============>................] - ETA: 9s
 8273920/17464789 [=============>................] - ETA: 9s
 8413184/17464789 [=============>................] - ETA: 8s
 8536064/17464789 [=============>................] - ETA: 8s
 8675328/17464789 [=============>................] - ETA: 8s
 8814592/17464789 [==============>...............] - ETA: 8s
 8970240/17464789 [==============>...............] - ETA: 8s
 9109504/17464789 [==============>...............] - ETA: 7s
 9265152/17464789 [==============>...............] - ETA: 7s
 9420800/17464789 [===============>..............] - ETA: 7s
 9576448/17464789 [===============>..............] - ETA: 7s
 9732096/17464789 [===============>..............] - ETA: 7s
 9912320/17464789 [================>.............] - ETA: 6s
10084352/17464789 [================>.............] - ETA: 6s
10240000/17464789 [================>.............] - ETA: 6s
10428416/17464789 [================>.............] - ETA: 6s
10608640/17464789 [=================>............] - ETA: 5s
10797056/17464789 [=================>............] - ETA: 5s
11001856/17464789 [=================>............] - ETA: 5s
11198464/17464789 [==================>...........] - ETA: 5s
11403264/17464789 [==================>...........] - ETA: 5s
11616256/17464789 [==================>...........] - ETA: 4s
11821056/17464789 [===================>..........] - ETA: 4s
12050432/17464789 [===================>..........] - ETA: 4s
12255232/17464789 [====================>.........] - ETA: 4s
12500992/17464789 [====================>.........] - ETA: 3s
12730368/17464789 [====================>.........] - ETA: 3s
12976128/17464789 [=====================>........] - ETA: 3s
13213696/17464789 [=====================>........] - ETA: 3s
13459456/17464789 [======================>.......] - ETA: 2s
13721600/17464789 [======================>.......] - ETA: 2s
13967360/17464789 [======================>.......] - ETA: 2s
14245888/17464789 [=======================>......] - ETA: 2s
14508032/17464789 [=======================>......] - ETA: 2s
14786560/17464789 [========================>.....] - ETA: 1s
15081472/17464789 [========================>.....] - ETA: 1s
15376384/17464789 [=========================>....] - ETA: 1s
15671296/17464789 [=========================>....] - ETA: 1s
15982592/17464789 [==========================>...] - ETA: 0s
16293888/17464789 [==========================>...] - ETA: 0s
16613376/17464789 [===========================>..] - ETA: 0s
16941056/17464789 [============================>.] - ETA: 0s
17268736/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 11s 1us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-18 10:16:43.927089: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-18 10:16:43.931545: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-18 10:16:43.931758: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5561b60d93a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 10:16:43.931774: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:26 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:44 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:12 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 1:55 - loss: 6.8281 - accuracy: 0.5547
  160/25000 [..............................] - ETA: 1:43 - loss: 6.8041 - accuracy: 0.5562
  192/25000 [..............................] - ETA: 1:36 - loss: 6.5486 - accuracy: 0.5729
  224/25000 [..............................] - ETA: 1:31 - loss: 6.2291 - accuracy: 0.5938
  256/25000 [..............................] - ETA: 1:27 - loss: 6.7083 - accuracy: 0.5625
  288/25000 [..............................] - ETA: 1:24 - loss: 6.8680 - accuracy: 0.5521
  320/25000 [..............................] - ETA: 1:21 - loss: 6.9479 - accuracy: 0.5469
  352/25000 [..............................] - ETA: 1:19 - loss: 6.9697 - accuracy: 0.5455
  384/25000 [..............................] - ETA: 1:17 - loss: 6.9479 - accuracy: 0.5469
  416/25000 [..............................] - ETA: 1:16 - loss: 6.8926 - accuracy: 0.5505
  448/25000 [..............................] - ETA: 1:14 - loss: 6.9479 - accuracy: 0.5469
  480/25000 [..............................] - ETA: 1:13 - loss: 6.9319 - accuracy: 0.5479
  512/25000 [..............................] - ETA: 1:12 - loss: 6.9179 - accuracy: 0.5488
  544/25000 [..............................] - ETA: 1:11 - loss: 6.8492 - accuracy: 0.5533
  576/25000 [..............................] - ETA: 1:11 - loss: 6.9745 - accuracy: 0.5451
  608/25000 [..............................] - ETA: 1:10 - loss: 7.0866 - accuracy: 0.5378
  640/25000 [..............................] - ETA: 1:09 - loss: 7.0916 - accuracy: 0.5375
  672/25000 [..............................] - ETA: 1:09 - loss: 7.2103 - accuracy: 0.5298
  704/25000 [..............................] - ETA: 1:09 - loss: 7.2528 - accuracy: 0.5270
  736/25000 [..............................] - ETA: 1:08 - loss: 7.3958 - accuracy: 0.5177
  768/25000 [..............................] - ETA: 1:08 - loss: 7.3472 - accuracy: 0.5208
  800/25000 [..............................] - ETA: 1:08 - loss: 7.3600 - accuracy: 0.5200
  832/25000 [..............................] - ETA: 1:07 - loss: 7.3349 - accuracy: 0.5216
  864/25000 [>.............................] - ETA: 1:07 - loss: 7.3294 - accuracy: 0.5220
  896/25000 [>.............................] - ETA: 1:07 - loss: 7.3072 - accuracy: 0.5234
  928/25000 [>.............................] - ETA: 1:06 - loss: 7.3527 - accuracy: 0.5205
  960/25000 [>.............................] - ETA: 1:06 - loss: 7.2833 - accuracy: 0.5250
  992/25000 [>.............................] - ETA: 1:05 - loss: 7.3575 - accuracy: 0.5202
 1024/25000 [>.............................] - ETA: 1:05 - loss: 7.3522 - accuracy: 0.5205
 1056/25000 [>.............................] - ETA: 1:05 - loss: 7.3762 - accuracy: 0.5189
 1088/25000 [>.............................] - ETA: 1:04 - loss: 7.3566 - accuracy: 0.5202
 1120/25000 [>.............................] - ETA: 1:04 - loss: 7.2970 - accuracy: 0.5241
 1152/25000 [>.............................] - ETA: 1:04 - loss: 7.3605 - accuracy: 0.5200
 1184/25000 [>.............................] - ETA: 1:04 - loss: 7.3040 - accuracy: 0.5236
 1216/25000 [>.............................] - ETA: 1:04 - loss: 7.3640 - accuracy: 0.5197
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.3717 - accuracy: 0.5192
 1280/25000 [>.............................] - ETA: 1:03 - loss: 7.3552 - accuracy: 0.5203
 1312/25000 [>.............................] - ETA: 1:03 - loss: 7.4095 - accuracy: 0.5168
 1344/25000 [>.............................] - ETA: 1:03 - loss: 7.4384 - accuracy: 0.5149
 1376/25000 [>.............................] - ETA: 1:03 - loss: 7.4326 - accuracy: 0.5153
 1408/25000 [>.............................] - ETA: 1:03 - loss: 7.4597 - accuracy: 0.5135
 1440/25000 [>.............................] - ETA: 1:03 - loss: 7.4643 - accuracy: 0.5132
 1472/25000 [>.............................] - ETA: 1:02 - loss: 7.4687 - accuracy: 0.5129
 1504/25000 [>.............................] - ETA: 1:02 - loss: 7.4729 - accuracy: 0.5126
 1536/25000 [>.............................] - ETA: 1:02 - loss: 7.4570 - accuracy: 0.5137
 1568/25000 [>.............................] - ETA: 1:02 - loss: 7.5102 - accuracy: 0.5102
 1600/25000 [>.............................] - ETA: 1:02 - loss: 7.5229 - accuracy: 0.5094
 1632/25000 [>.............................] - ETA: 1:02 - loss: 7.5257 - accuracy: 0.5092
 1664/25000 [>.............................] - ETA: 1:01 - loss: 7.5192 - accuracy: 0.5096
 1696/25000 [=>............................] - ETA: 1:01 - loss: 7.5491 - accuracy: 0.5077
 1728/25000 [=>............................] - ETA: 1:01 - loss: 7.5779 - accuracy: 0.5058
 1760/25000 [=>............................] - ETA: 1:01 - loss: 7.6143 - accuracy: 0.5034
 1792/25000 [=>............................] - ETA: 1:01 - loss: 7.5982 - accuracy: 0.5045
 1824/25000 [=>............................] - ETA: 1:01 - loss: 7.5910 - accuracy: 0.5049
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.5757 - accuracy: 0.5059
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.5692 - accuracy: 0.5064
 1920/25000 [=>............................] - ETA: 1:00 - loss: 7.5628 - accuracy: 0.5068
 1952/25000 [=>............................] - ETA: 1:00 - loss: 7.5881 - accuracy: 0.5051
 1984/25000 [=>............................] - ETA: 1:00 - loss: 7.6202 - accuracy: 0.5030
 2016/25000 [=>............................] - ETA: 1:00 - loss: 7.6438 - accuracy: 0.5015
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.6591 - accuracy: 0.5005
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.7035 - accuracy: 0.4976
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.7174 - accuracy: 0.4967
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.7167 - accuracy: 0.4967
 2176/25000 [=>............................] - ETA: 1:00 - loss: 7.7159 - accuracy: 0.4968
 2208/25000 [=>............................] - ETA: 59s - loss: 7.6875 - accuracy: 0.4986 
 2240/25000 [=>............................] - ETA: 59s - loss: 7.6735 - accuracy: 0.4996
 2272/25000 [=>............................] - ETA: 59s - loss: 7.6599 - accuracy: 0.5004
 2304/25000 [=>............................] - ETA: 59s - loss: 7.6733 - accuracy: 0.4996
 2336/25000 [=>............................] - ETA: 59s - loss: 7.6797 - accuracy: 0.4991
 2368/25000 [=>............................] - ETA: 59s - loss: 7.6731 - accuracy: 0.4996
 2400/25000 [=>............................] - ETA: 59s - loss: 7.6794 - accuracy: 0.4992
 2432/25000 [=>............................] - ETA: 59s - loss: 7.7108 - accuracy: 0.4971
 2464/25000 [=>............................] - ETA: 59s - loss: 7.7102 - accuracy: 0.4972
 2496/25000 [=>............................] - ETA: 59s - loss: 7.7281 - accuracy: 0.4960
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.7455 - accuracy: 0.4949
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.7804 - accuracy: 0.4926
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.7494 - accuracy: 0.4946
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.7543 - accuracy: 0.4943
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.7705 - accuracy: 0.4932
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.7978 - accuracy: 0.4914
 2720/25000 [==>...........................] - ETA: 58s - loss: 7.8470 - accuracy: 0.4882
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.8226 - accuracy: 0.4898
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.8208 - accuracy: 0.4899
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.8082 - accuracy: 0.4908
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.8335 - accuracy: 0.4891
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.8476 - accuracy: 0.4882
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.8299 - accuracy: 0.4894
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.8385 - accuracy: 0.4888
 2976/25000 [==>...........................] - ETA: 57s - loss: 7.8315 - accuracy: 0.4892
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.8195 - accuracy: 0.4900
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.8230 - accuracy: 0.4898
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.8263 - accuracy: 0.4896
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.8296 - accuracy: 0.4894
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.8084 - accuracy: 0.4908
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.8070 - accuracy: 0.4908
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.8152 - accuracy: 0.4903
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.8422 - accuracy: 0.4886
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.8451 - accuracy: 0.4884
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.8341 - accuracy: 0.4891
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.8555 - accuracy: 0.4877
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.8583 - accuracy: 0.4875
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.8565 - accuracy: 0.4876
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.8547 - accuracy: 0.4877
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.8751 - accuracy: 0.4864
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.8820 - accuracy: 0.4860
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.8626 - accuracy: 0.4872
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.8393 - accuracy: 0.4887
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.8377 - accuracy: 0.4888
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.8532 - accuracy: 0.4878
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.8347 - accuracy: 0.4890
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.8375 - accuracy: 0.4889
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.8360 - accuracy: 0.4890
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.8181 - accuracy: 0.4901
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.8209 - accuracy: 0.4899
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.8317 - accuracy: 0.4892
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.8303 - accuracy: 0.4893
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.8250 - accuracy: 0.4897
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.8276 - accuracy: 0.4895
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.8341 - accuracy: 0.4891
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.8405 - accuracy: 0.4887
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.8468 - accuracy: 0.4882
 4032/25000 [===>..........................] - ETA: 54s - loss: 7.8606 - accuracy: 0.4874
 4064/25000 [===>..........................] - ETA: 54s - loss: 7.8553 - accuracy: 0.4877
 4096/25000 [===>..........................] - ETA: 54s - loss: 7.8613 - accuracy: 0.4873
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.8561 - accuracy: 0.4876
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.8509 - accuracy: 0.4880
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.8495 - accuracy: 0.4881
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.8554 - accuracy: 0.4877
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.8432 - accuracy: 0.4885
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.8168 - accuracy: 0.4902
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.8157 - accuracy: 0.4903
 4352/25000 [====>.........................] - ETA: 53s - loss: 7.8252 - accuracy: 0.4897
 4384/25000 [====>.........................] - ETA: 53s - loss: 7.8135 - accuracy: 0.4904
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.8125 - accuracy: 0.4905
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.8080 - accuracy: 0.4908
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.8069 - accuracy: 0.4908
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.8127 - accuracy: 0.4905
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.8117 - accuracy: 0.4905
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.8107 - accuracy: 0.4906
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.8064 - accuracy: 0.4909
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.8087 - accuracy: 0.4907
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.8012 - accuracy: 0.4912
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.7970 - accuracy: 0.4915
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.7929 - accuracy: 0.4918
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.7888 - accuracy: 0.4920
 4800/25000 [====>.........................] - ETA: 52s - loss: 7.7880 - accuracy: 0.4921
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.7936 - accuracy: 0.4917
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.8053 - accuracy: 0.4910
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.8044 - accuracy: 0.4910
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.8004 - accuracy: 0.4913
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.8026 - accuracy: 0.4911
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.8079 - accuracy: 0.4908
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.8131 - accuracy: 0.4904
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.8092 - accuracy: 0.4907
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.8022 - accuracy: 0.4912
 5120/25000 [=====>........................] - ETA: 51s - loss: 7.7864 - accuracy: 0.4922
 5152/25000 [=====>........................] - ETA: 51s - loss: 7.7797 - accuracy: 0.4926
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.7879 - accuracy: 0.4921
 5216/25000 [=====>........................] - ETA: 51s - loss: 7.7901 - accuracy: 0.4919
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7952 - accuracy: 0.4916
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.7944 - accuracy: 0.4917
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.7936 - accuracy: 0.4917
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.7986 - accuracy: 0.4914
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.7978 - accuracy: 0.4914
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.7914 - accuracy: 0.4919
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.7991 - accuracy: 0.4914
 5472/25000 [=====>........................] - ETA: 50s - loss: 7.7983 - accuracy: 0.4914
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.7864 - accuracy: 0.4922
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.7968 - accuracy: 0.4915
 5568/25000 [=====>........................] - ETA: 50s - loss: 7.7905 - accuracy: 0.4919
 5600/25000 [=====>........................] - ETA: 50s - loss: 7.7707 - accuracy: 0.4932
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.7728 - accuracy: 0.4931
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.7830 - accuracy: 0.4924
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.7878 - accuracy: 0.4921
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.7951 - accuracy: 0.4916
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.7891 - accuracy: 0.4920
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.7963 - accuracy: 0.4915
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.7930 - accuracy: 0.4918
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.7766 - accuracy: 0.4928
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.7864 - accuracy: 0.4922
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.7987 - accuracy: 0.4914
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.7903 - accuracy: 0.4919
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.8075 - accuracy: 0.4908
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.8119 - accuracy: 0.4905
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.8111 - accuracy: 0.4906
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.8230 - accuracy: 0.4898
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.8197 - accuracy: 0.4900
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.8213 - accuracy: 0.4899
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.8305 - accuracy: 0.4893
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.8370 - accuracy: 0.4889
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.8386 - accuracy: 0.4888
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.8475 - accuracy: 0.4882
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.8417 - accuracy: 0.4886
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.8360 - accuracy: 0.4890
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.8424 - accuracy: 0.4885
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.8367 - accuracy: 0.4889
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.8359 - accuracy: 0.4890
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.8327 - accuracy: 0.4892
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.8271 - accuracy: 0.4895
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.8193 - accuracy: 0.4900
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.8232 - accuracy: 0.4898
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.8178 - accuracy: 0.4901
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.8125 - accuracy: 0.4905
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.8164 - accuracy: 0.4902
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.8019 - accuracy: 0.4912
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.8127 - accuracy: 0.4905
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.8097 - accuracy: 0.4907
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.8158 - accuracy: 0.4903
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.8173 - accuracy: 0.4902
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.8099 - accuracy: 0.4907
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.8115 - accuracy: 0.4906
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.8197 - accuracy: 0.4900
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.8146 - accuracy: 0.4904
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.8139 - accuracy: 0.4904
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.8220 - accuracy: 0.4899
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.8234 - accuracy: 0.4898
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.8314 - accuracy: 0.4893
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.8220 - accuracy: 0.4899
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.8170 - accuracy: 0.4902
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.8142 - accuracy: 0.4904
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.8178 - accuracy: 0.4901
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.8066 - accuracy: 0.4909
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.7954 - accuracy: 0.4916
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.8053 - accuracy: 0.4910
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.7859 - accuracy: 0.4922
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.7895 - accuracy: 0.4920
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.7849 - accuracy: 0.4923
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.7781 - accuracy: 0.4927
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.7880 - accuracy: 0.4921
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.7833 - accuracy: 0.4924
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.7849 - accuracy: 0.4923
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.7966 - accuracy: 0.4915
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.7879 - accuracy: 0.4921
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.7955 - accuracy: 0.4916
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.8030 - accuracy: 0.4911
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.8024 - accuracy: 0.4911
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.7998 - accuracy: 0.4913
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.7993 - accuracy: 0.4913
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.7948 - accuracy: 0.4916
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.8060 - accuracy: 0.4909
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.8113 - accuracy: 0.4906
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.8069 - accuracy: 0.4909
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.8102 - accuracy: 0.4906
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.8154 - accuracy: 0.4903
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.8071 - accuracy: 0.4908
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.8065 - accuracy: 0.4909
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.8041 - accuracy: 0.4910
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.8035 - accuracy: 0.4911
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.8011 - accuracy: 0.4912
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.7968 - accuracy: 0.4915
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.7906 - accuracy: 0.4919
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.7902 - accuracy: 0.4919
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.7859 - accuracy: 0.4922
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.7836 - accuracy: 0.4924
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.7850 - accuracy: 0.4923
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.7938 - accuracy: 0.4917
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.7915 - accuracy: 0.4919
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.7910 - accuracy: 0.4919
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.7887 - accuracy: 0.4920
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.7828 - accuracy: 0.4924
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.7860 - accuracy: 0.4922
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.7819 - accuracy: 0.4925
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.7761 - accuracy: 0.4929
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.7739 - accuracy: 0.4930
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.7699 - accuracy: 0.4933
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.7607 - accuracy: 0.4939
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.7586 - accuracy: 0.4940
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7600 - accuracy: 0.4939
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.7614 - accuracy: 0.4938
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.7506 - accuracy: 0.4945
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.7537 - accuracy: 0.4943
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.7534 - accuracy: 0.4943
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.7514 - accuracy: 0.4945
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.7494 - accuracy: 0.4946
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.7439 - accuracy: 0.4950
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.7419 - accuracy: 0.4951
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.7468 - accuracy: 0.4948
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.7397 - accuracy: 0.4952
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.7360 - accuracy: 0.4955
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.7476 - accuracy: 0.4947
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.7473 - accuracy: 0.4947
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.7521 - accuracy: 0.4944
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.7468 - accuracy: 0.4948
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.7448 - accuracy: 0.4949
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.7462 - accuracy: 0.4948
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.7426 - accuracy: 0.4950
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.7407 - accuracy: 0.4952
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.7355 - accuracy: 0.4955
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.7337 - accuracy: 0.4956
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7286 - accuracy: 0.4960
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7365 - accuracy: 0.4954
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7411 - accuracy: 0.4951
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7295 - accuracy: 0.4959
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7277 - accuracy: 0.4960
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7291 - accuracy: 0.4959
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.7273 - accuracy: 0.4960
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.7223 - accuracy: 0.4964
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.7237 - accuracy: 0.4963
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.7220 - accuracy: 0.4964
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.7297 - accuracy: 0.4959
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.7279 - accuracy: 0.4960
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.7261 - accuracy: 0.4961
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.7322 - accuracy: 0.4957
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7304 - accuracy: 0.4958
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7286 - accuracy: 0.4960
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.7238 - accuracy: 0.4963
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.7252 - accuracy: 0.4962
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.7311 - accuracy: 0.4958
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7324 - accuracy: 0.4957
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7338 - accuracy: 0.4956
10080/25000 [===========>..................] - ETA: 38s - loss: 7.7290 - accuracy: 0.4959
10112/25000 [===========>..................] - ETA: 38s - loss: 7.7349 - accuracy: 0.4955
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7392 - accuracy: 0.4953
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7329 - accuracy: 0.4957
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7357 - accuracy: 0.4955
10240/25000 [===========>..................] - ETA: 37s - loss: 7.7355 - accuracy: 0.4955
10272/25000 [===========>..................] - ETA: 37s - loss: 7.7308 - accuracy: 0.4958
10304/25000 [===========>..................] - ETA: 37s - loss: 7.7291 - accuracy: 0.4959
10336/25000 [===========>..................] - ETA: 37s - loss: 7.7245 - accuracy: 0.4962
10368/25000 [===========>..................] - ETA: 37s - loss: 7.7317 - accuracy: 0.4958
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7344 - accuracy: 0.4956
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7416 - accuracy: 0.4951
10464/25000 [===========>..................] - ETA: 37s - loss: 7.7355 - accuracy: 0.4955
10496/25000 [===========>..................] - ETA: 36s - loss: 7.7294 - accuracy: 0.4959
10528/25000 [===========>..................] - ETA: 36s - loss: 7.7249 - accuracy: 0.4962
10560/25000 [===========>..................] - ETA: 36s - loss: 7.7232 - accuracy: 0.4963
10592/25000 [===========>..................] - ETA: 36s - loss: 7.7260 - accuracy: 0.4961
10624/25000 [===========>..................] - ETA: 36s - loss: 7.7244 - accuracy: 0.4962
10656/25000 [===========>..................] - ETA: 36s - loss: 7.7213 - accuracy: 0.4964
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7168 - accuracy: 0.4967
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7138 - accuracy: 0.4969
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7194 - accuracy: 0.4966
10784/25000 [===========>..................] - ETA: 36s - loss: 7.7192 - accuracy: 0.4966
10816/25000 [===========>..................] - ETA: 36s - loss: 7.7148 - accuracy: 0.4969
10848/25000 [============>.................] - ETA: 36s - loss: 7.7147 - accuracy: 0.4969
10880/25000 [============>.................] - ETA: 35s - loss: 7.7089 - accuracy: 0.4972
10912/25000 [============>.................] - ETA: 35s - loss: 7.7130 - accuracy: 0.4970
10944/25000 [============>.................] - ETA: 35s - loss: 7.7143 - accuracy: 0.4969
10976/25000 [============>.................] - ETA: 35s - loss: 7.7127 - accuracy: 0.4970
11008/25000 [============>.................] - ETA: 35s - loss: 7.7126 - accuracy: 0.4970
11040/25000 [============>.................] - ETA: 35s - loss: 7.7125 - accuracy: 0.4970
11072/25000 [============>.................] - ETA: 35s - loss: 7.7082 - accuracy: 0.4973
11104/25000 [============>.................] - ETA: 35s - loss: 7.7108 - accuracy: 0.4971
11136/25000 [============>.................] - ETA: 35s - loss: 7.7148 - accuracy: 0.4969
11168/25000 [============>.................] - ETA: 35s - loss: 7.7174 - accuracy: 0.4967
11200/25000 [============>.................] - ETA: 35s - loss: 7.7159 - accuracy: 0.4968
11232/25000 [============>.................] - ETA: 35s - loss: 7.7199 - accuracy: 0.4965
11264/25000 [============>.................] - ETA: 34s - loss: 7.7183 - accuracy: 0.4966
11296/25000 [============>.................] - ETA: 34s - loss: 7.7182 - accuracy: 0.4966
11328/25000 [============>.................] - ETA: 34s - loss: 7.7208 - accuracy: 0.4965
11360/25000 [============>.................] - ETA: 34s - loss: 7.7206 - accuracy: 0.4965
11392/25000 [============>.................] - ETA: 34s - loss: 7.7205 - accuracy: 0.4965
11424/25000 [============>.................] - ETA: 34s - loss: 7.7190 - accuracy: 0.4966
11456/25000 [============>.................] - ETA: 34s - loss: 7.7175 - accuracy: 0.4967
11488/25000 [============>.................] - ETA: 34s - loss: 7.7227 - accuracy: 0.4963
11520/25000 [============>.................] - ETA: 34s - loss: 7.7185 - accuracy: 0.4966
11552/25000 [============>.................] - ETA: 34s - loss: 7.7131 - accuracy: 0.4970
11584/25000 [============>.................] - ETA: 34s - loss: 7.7196 - accuracy: 0.4965
11616/25000 [============>.................] - ETA: 34s - loss: 7.7207 - accuracy: 0.4965
11648/25000 [============>.................] - ETA: 33s - loss: 7.7193 - accuracy: 0.4966
11680/25000 [=============>................] - ETA: 33s - loss: 7.7165 - accuracy: 0.4967
11712/25000 [=============>................] - ETA: 33s - loss: 7.7138 - accuracy: 0.4969
11744/25000 [=============>................] - ETA: 33s - loss: 7.7188 - accuracy: 0.4966
11776/25000 [=============>................] - ETA: 33s - loss: 7.7226 - accuracy: 0.4963
11808/25000 [=============>................] - ETA: 33s - loss: 7.7238 - accuracy: 0.4963
11840/25000 [=============>................] - ETA: 33s - loss: 7.7275 - accuracy: 0.4960
11872/25000 [=============>................] - ETA: 33s - loss: 7.7299 - accuracy: 0.4959
11904/25000 [=============>................] - ETA: 33s - loss: 7.7272 - accuracy: 0.4961
11936/25000 [=============>................] - ETA: 33s - loss: 7.7193 - accuracy: 0.4966
11968/25000 [=============>................] - ETA: 33s - loss: 7.7268 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 33s - loss: 7.7254 - accuracy: 0.4962
12032/25000 [=============>................] - ETA: 32s - loss: 7.7303 - accuracy: 0.4958
12064/25000 [=============>................] - ETA: 32s - loss: 7.7200 - accuracy: 0.4965
12096/25000 [=============>................] - ETA: 32s - loss: 7.7249 - accuracy: 0.4962
12128/25000 [=============>................] - ETA: 32s - loss: 7.7298 - accuracy: 0.4959
12160/25000 [=============>................] - ETA: 32s - loss: 7.7246 - accuracy: 0.4962
12192/25000 [=============>................] - ETA: 32s - loss: 7.7257 - accuracy: 0.4961
12224/25000 [=============>................] - ETA: 32s - loss: 7.7268 - accuracy: 0.4961
12256/25000 [=============>................] - ETA: 32s - loss: 7.7267 - accuracy: 0.4961
12288/25000 [=============>................] - ETA: 32s - loss: 7.7290 - accuracy: 0.4959
12320/25000 [=============>................] - ETA: 32s - loss: 7.7251 - accuracy: 0.4962
12352/25000 [=============>................] - ETA: 32s - loss: 7.7287 - accuracy: 0.4960
12384/25000 [=============>................] - ETA: 32s - loss: 7.7372 - accuracy: 0.4954
12416/25000 [=============>................] - ETA: 32s - loss: 7.7370 - accuracy: 0.4954
12448/25000 [=============>................] - ETA: 31s - loss: 7.7381 - accuracy: 0.4953
12480/25000 [=============>................] - ETA: 31s - loss: 7.7342 - accuracy: 0.4956
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7328 - accuracy: 0.4957
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7351 - accuracy: 0.4955
12576/25000 [==============>...............] - ETA: 31s - loss: 7.7300 - accuracy: 0.4959
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7347 - accuracy: 0.4956
12640/25000 [==============>...............] - ETA: 31s - loss: 7.7346 - accuracy: 0.4956
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7368 - accuracy: 0.4954
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7354 - accuracy: 0.4955
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7364 - accuracy: 0.4954
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7351 - accuracy: 0.4955
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7337 - accuracy: 0.4956
12832/25000 [==============>...............] - ETA: 30s - loss: 7.7335 - accuracy: 0.4956
12864/25000 [==============>...............] - ETA: 30s - loss: 7.7334 - accuracy: 0.4956
12896/25000 [==============>...............] - ETA: 30s - loss: 7.7368 - accuracy: 0.4954
12928/25000 [==============>...............] - ETA: 30s - loss: 7.7378 - accuracy: 0.4954
12960/25000 [==============>...............] - ETA: 30s - loss: 7.7364 - accuracy: 0.4954
12992/25000 [==============>...............] - ETA: 30s - loss: 7.7374 - accuracy: 0.4954
13024/25000 [==============>...............] - ETA: 30s - loss: 7.7396 - accuracy: 0.4952
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7359 - accuracy: 0.4955
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7322 - accuracy: 0.4957
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7356 - accuracy: 0.4955
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7366 - accuracy: 0.4954
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7283 - accuracy: 0.4960
13216/25000 [==============>...............] - ETA: 29s - loss: 7.7246 - accuracy: 0.4962
13248/25000 [==============>...............] - ETA: 29s - loss: 7.7233 - accuracy: 0.4963
13280/25000 [==============>...............] - ETA: 29s - loss: 7.7209 - accuracy: 0.4965
13312/25000 [==============>...............] - ETA: 29s - loss: 7.7173 - accuracy: 0.4967
13344/25000 [===============>..............] - ETA: 29s - loss: 7.7091 - accuracy: 0.4972
13376/25000 [===============>..............] - ETA: 29s - loss: 7.7067 - accuracy: 0.4974
13408/25000 [===============>..............] - ETA: 29s - loss: 7.7009 - accuracy: 0.4978
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6974 - accuracy: 0.4980
13472/25000 [===============>..............] - ETA: 29s - loss: 7.7030 - accuracy: 0.4976
13504/25000 [===============>..............] - ETA: 29s - loss: 7.7041 - accuracy: 0.4976
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6995 - accuracy: 0.4979
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6949 - accuracy: 0.4982
13600/25000 [===============>..............] - ETA: 28s - loss: 7.7027 - accuracy: 0.4976
13632/25000 [===============>..............] - ETA: 28s - loss: 7.7071 - accuracy: 0.4974
13664/25000 [===============>..............] - ETA: 28s - loss: 7.7081 - accuracy: 0.4973
13696/25000 [===============>..............] - ETA: 28s - loss: 7.7069 - accuracy: 0.4974
13728/25000 [===============>..............] - ETA: 28s - loss: 7.7068 - accuracy: 0.4974
13760/25000 [===============>..............] - ETA: 28s - loss: 7.7056 - accuracy: 0.4975
13792/25000 [===============>..............] - ETA: 28s - loss: 7.7033 - accuracy: 0.4976
13824/25000 [===============>..............] - ETA: 28s - loss: 7.7021 - accuracy: 0.4977
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6987 - accuracy: 0.4979
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6953 - accuracy: 0.4981
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7041 - accuracy: 0.4976
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7040 - accuracy: 0.4976
13984/25000 [===============>..............] - ETA: 27s - loss: 7.7017 - accuracy: 0.4977
14016/25000 [===============>..............] - ETA: 27s - loss: 7.7038 - accuracy: 0.4976
14048/25000 [===============>..............] - ETA: 27s - loss: 7.7048 - accuracy: 0.4975
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6993 - accuracy: 0.4979
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6992 - accuracy: 0.4979
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6981 - accuracy: 0.4979
14176/25000 [================>.............] - ETA: 27s - loss: 7.6937 - accuracy: 0.4982
14208/25000 [================>.............] - ETA: 27s - loss: 7.6860 - accuracy: 0.4987
14240/25000 [================>.............] - ETA: 27s - loss: 7.6860 - accuracy: 0.4987
14272/25000 [================>.............] - ETA: 27s - loss: 7.6817 - accuracy: 0.4990
14304/25000 [================>.............] - ETA: 27s - loss: 7.6827 - accuracy: 0.4990
14336/25000 [================>.............] - ETA: 27s - loss: 7.6784 - accuracy: 0.4992
14368/25000 [================>.............] - ETA: 26s - loss: 7.6784 - accuracy: 0.4992
14400/25000 [================>.............] - ETA: 26s - loss: 7.6773 - accuracy: 0.4993
14432/25000 [================>.............] - ETA: 26s - loss: 7.6783 - accuracy: 0.4992
14464/25000 [================>.............] - ETA: 26s - loss: 7.6740 - accuracy: 0.4995
14496/25000 [================>.............] - ETA: 26s - loss: 7.6740 - accuracy: 0.4995
14528/25000 [================>.............] - ETA: 26s - loss: 7.6740 - accuracy: 0.4995
14560/25000 [================>.............] - ETA: 26s - loss: 7.6719 - accuracy: 0.4997
14592/25000 [================>.............] - ETA: 26s - loss: 7.6750 - accuracy: 0.4995
14624/25000 [================>.............] - ETA: 26s - loss: 7.6761 - accuracy: 0.4994
14656/25000 [================>.............] - ETA: 26s - loss: 7.6771 - accuracy: 0.4993
14688/25000 [================>.............] - ETA: 26s - loss: 7.6750 - accuracy: 0.4995
14720/25000 [================>.............] - ETA: 26s - loss: 7.6781 - accuracy: 0.4993
14752/25000 [================>.............] - ETA: 25s - loss: 7.6791 - accuracy: 0.4992
14784/25000 [================>.............] - ETA: 25s - loss: 7.6780 - accuracy: 0.4993
14816/25000 [================>.............] - ETA: 25s - loss: 7.6749 - accuracy: 0.4995
14848/25000 [================>.............] - ETA: 25s - loss: 7.6769 - accuracy: 0.4993
14880/25000 [================>.............] - ETA: 25s - loss: 7.6790 - accuracy: 0.4992
14912/25000 [================>.............] - ETA: 25s - loss: 7.6831 - accuracy: 0.4989
14944/25000 [================>.............] - ETA: 25s - loss: 7.6810 - accuracy: 0.4991
14976/25000 [================>.............] - ETA: 25s - loss: 7.6769 - accuracy: 0.4993
15008/25000 [=================>............] - ETA: 25s - loss: 7.6738 - accuracy: 0.4995
15040/25000 [=================>............] - ETA: 25s - loss: 7.6738 - accuracy: 0.4995
15072/25000 [=================>............] - ETA: 25s - loss: 7.6798 - accuracy: 0.4991
15104/25000 [=================>............] - ETA: 25s - loss: 7.6818 - accuracy: 0.4990
15136/25000 [=================>............] - ETA: 24s - loss: 7.6808 - accuracy: 0.4991
15168/25000 [=================>............] - ETA: 24s - loss: 7.6828 - accuracy: 0.4989
15200/25000 [=================>............] - ETA: 24s - loss: 7.6858 - accuracy: 0.4988
15232/25000 [=================>............] - ETA: 24s - loss: 7.6847 - accuracy: 0.4988
15264/25000 [=================>............] - ETA: 24s - loss: 7.6857 - accuracy: 0.4988
15296/25000 [=================>............] - ETA: 24s - loss: 7.6837 - accuracy: 0.4989
15328/25000 [=================>............] - ETA: 24s - loss: 7.6866 - accuracy: 0.4987
15360/25000 [=================>............] - ETA: 24s - loss: 7.6916 - accuracy: 0.4984
15392/25000 [=================>............] - ETA: 24s - loss: 7.6925 - accuracy: 0.4983
15424/25000 [=================>............] - ETA: 24s - loss: 7.6915 - accuracy: 0.4984
15456/25000 [=================>............] - ETA: 24s - loss: 7.6954 - accuracy: 0.4981
15488/25000 [=================>............] - ETA: 24s - loss: 7.6973 - accuracy: 0.4980
15520/25000 [=================>............] - ETA: 23s - loss: 7.6992 - accuracy: 0.4979
15552/25000 [=================>............] - ETA: 23s - loss: 7.6982 - accuracy: 0.4979
15584/25000 [=================>............] - ETA: 23s - loss: 7.6952 - accuracy: 0.4981
15616/25000 [=================>............] - ETA: 23s - loss: 7.6980 - accuracy: 0.4980
15648/25000 [=================>............] - ETA: 23s - loss: 7.7019 - accuracy: 0.4977
15680/25000 [=================>............] - ETA: 23s - loss: 7.6979 - accuracy: 0.4980
15712/25000 [=================>............] - ETA: 23s - loss: 7.6969 - accuracy: 0.4980
15744/25000 [=================>............] - ETA: 23s - loss: 7.6939 - accuracy: 0.4982
15776/25000 [=================>............] - ETA: 23s - loss: 7.6899 - accuracy: 0.4985
15808/25000 [=================>............] - ETA: 23s - loss: 7.6889 - accuracy: 0.4985
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6937 - accuracy: 0.4982
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6927 - accuracy: 0.4983
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6859 - accuracy: 0.4987
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6897 - accuracy: 0.4985
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6868 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6858 - accuracy: 0.4988
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6896 - accuracy: 0.4985
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6876 - accuracy: 0.4986
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6876 - accuracy: 0.4986
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6847 - accuracy: 0.4988
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6846 - accuracy: 0.4988
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6780 - accuracy: 0.4993
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6770 - accuracy: 0.4993
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6789 - accuracy: 0.4992
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6760 - accuracy: 0.4994
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6723 - accuracy: 0.4996
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6713 - accuracy: 0.4997
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6676 - accuracy: 0.4999
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6694 - accuracy: 0.4998
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6731 - accuracy: 0.4996
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6787 - accuracy: 0.4992
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6768 - accuracy: 0.4993
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6731 - accuracy: 0.4996
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6749 - accuracy: 0.4995
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6703 - accuracy: 0.4998
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6703 - accuracy: 0.4998
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6685 - accuracy: 0.4999
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6712 - accuracy: 0.4997
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6721 - accuracy: 0.4996
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6721 - accuracy: 0.4996
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6721 - accuracy: 0.4996
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6703 - accuracy: 0.4998
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6693 - accuracy: 0.4998
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6684 - accuracy: 0.4999
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6684 - accuracy: 0.4999
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6684 - accuracy: 0.4999
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6675 - accuracy: 0.4999
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6639 - accuracy: 0.5002
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6639 - accuracy: 0.5002
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6711 - accuracy: 0.4997
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6728 - accuracy: 0.4996
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6728 - accuracy: 0.4996
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6728 - accuracy: 0.4996
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6754 - accuracy: 0.4994
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6745 - accuracy: 0.4995
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6737 - accuracy: 0.4995
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6745 - accuracy: 0.4995
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6754 - accuracy: 0.4994
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6754 - accuracy: 0.4994
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6727 - accuracy: 0.4996
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6762 - accuracy: 0.4994
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6727 - accuracy: 0.4996
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6753 - accuracy: 0.4994
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6788 - accuracy: 0.4992
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6753 - accuracy: 0.4994
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6744 - accuracy: 0.4995
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6752 - accuracy: 0.4994
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6726 - accuracy: 0.4996
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6752 - accuracy: 0.4994
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6726 - accuracy: 0.4996
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6735 - accuracy: 0.4996
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6743 - accuracy: 0.4995
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6734 - accuracy: 0.4996
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6734 - accuracy: 0.4996
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6777 - accuracy: 0.4993
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6759 - accuracy: 0.4994
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6751 - accuracy: 0.4994
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6751 - accuracy: 0.4994
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6751 - accuracy: 0.4994
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6759 - accuracy: 0.4994
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6767 - accuracy: 0.4993
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6750 - accuracy: 0.4995
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6750 - accuracy: 0.4995
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6741 - accuracy: 0.4995
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6724 - accuracy: 0.4996
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6741 - accuracy: 0.4995
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6765 - accuracy: 0.4994
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6757 - accuracy: 0.4994
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6773 - accuracy: 0.4993
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6740 - accuracy: 0.4995
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6707 - accuracy: 0.4997
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6715 - accuracy: 0.4997
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6732 - accuracy: 0.4996
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6691 - accuracy: 0.4998
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6691 - accuracy: 0.4998
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6666 - accuracy: 0.5000
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6682 - accuracy: 0.4999
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6698 - accuracy: 0.4998
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6739 - accuracy: 0.4995
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6754 - accuracy: 0.4994
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6762 - accuracy: 0.4994
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6778 - accuracy: 0.4993
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6778 - accuracy: 0.4993
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6794 - accuracy: 0.4992
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6809 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6825 - accuracy: 0.4990
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6825 - accuracy: 0.4990
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6801 - accuracy: 0.4991
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6801 - accuracy: 0.4991
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6792 - accuracy: 0.4992
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6816 - accuracy: 0.4990
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6815 - accuracy: 0.4990
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6815 - accuracy: 0.4990
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6838 - accuracy: 0.4989
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6814 - accuracy: 0.4990
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6783 - accuracy: 0.4992
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6806 - accuracy: 0.4991
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6822 - accuracy: 0.4990
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6821 - accuracy: 0.4990
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6782 - accuracy: 0.4992
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6798 - accuracy: 0.4991
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6774 - accuracy: 0.4993
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6759 - accuracy: 0.4994
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6766 - accuracy: 0.4993
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6774 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6804 - accuracy: 0.4991
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6819 - accuracy: 0.4990
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6804 - accuracy: 0.4991
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6788 - accuracy: 0.4992
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6765 - accuracy: 0.4994
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6742 - accuracy: 0.4995
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6712 - accuracy: 0.4997
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6704 - accuracy: 0.4998
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6704 - accuracy: 0.4998
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6779 - accuracy: 0.4993
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6787 - accuracy: 0.4992
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6779 - accuracy: 0.4993
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6764 - accuracy: 0.4994
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6734 - accuracy: 0.4996
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6741 - accuracy: 0.4995
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6711 - accuracy: 0.4997
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6689 - accuracy: 0.4999
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6681 - accuracy: 0.4999
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6615 - accuracy: 0.5003
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6615 - accuracy: 0.5003
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6622 - accuracy: 0.5003
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6615 - accuracy: 0.5003
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6644 - accuracy: 0.5001 
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6644 - accuracy: 0.5001
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6717 - accuracy: 0.4997
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6738 - accuracy: 0.4995
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6731 - accuracy: 0.4996
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6745 - accuracy: 0.4995
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6774 - accuracy: 0.4993
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6788 - accuracy: 0.4992
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6795 - accuracy: 0.4992
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6802 - accuracy: 0.4991
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6802 - accuracy: 0.4991
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6787 - accuracy: 0.4992
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6808 - accuracy: 0.4991
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6815 - accuracy: 0.4990
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6822 - accuracy: 0.4990
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6815 - accuracy: 0.4990
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6836 - accuracy: 0.4989
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6821 - accuracy: 0.4990
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6814 - accuracy: 0.4990
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6772 - accuracy: 0.4993
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6764 - accuracy: 0.4994
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6722 - accuracy: 0.4996
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6757 - accuracy: 0.4994
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6729 - accuracy: 0.4996
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6756 - accuracy: 0.4994
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6804 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6811 - accuracy: 0.4991
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6804 - accuracy: 0.4991
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6817 - accuracy: 0.4990
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6783 - accuracy: 0.4992
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6776 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6748 - accuracy: 0.4995
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6795 - accuracy: 0.4992
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6727 - accuracy: 0.4996
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6714 - accuracy: 0.4997
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6693 - accuracy: 0.4998
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6693 - accuracy: 0.4998
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6713 - accuracy: 0.4997
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6740 - accuracy: 0.4995
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6780 - accuracy: 0.4993
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6780 - accuracy: 0.4993
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6780 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6773 - accuracy: 0.4993
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6793 - accuracy: 0.4992
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6813 - accuracy: 0.4990
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6831 - accuracy: 0.4989
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6838 - accuracy: 0.4989
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6831 - accuracy: 0.4989
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6850 - accuracy: 0.4988
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6863 - accuracy: 0.4987
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6863 - accuracy: 0.4987
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6849 - accuracy: 0.4988
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6862 - accuracy: 0.4987
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6849 - accuracy: 0.4988
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6855 - accuracy: 0.4988
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6842 - accuracy: 0.4989
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6829 - accuracy: 0.4989
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6816 - accuracy: 0.4990
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6802 - accuracy: 0.4991
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6783 - accuracy: 0.4992
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6796 - accuracy: 0.4992
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6770 - accuracy: 0.4993
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6769 - accuracy: 0.4993
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6776 - accuracy: 0.4993
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 1s - loss: 7.6673 - accuracy: 0.5000
24256/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24288/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24320/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24352/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24384/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24416/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24448/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24480/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24512/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24544/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24672/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24832/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 72s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
