
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/93a93a77223ff27cba39886e979e8d7be4c12485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '93a93a77223ff27cba39886e979e8d7be4c12485', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/93a93a77223ff27cba39886e979e8d7be4c12485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/93a93a77223ff27cba39886e979e8d7be4c12485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/93a93a77223ff27cba39886e979e8d7be4c12485

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n']





 ********************************************************************************************************************************************
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do model_list  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models  --do generate_config  --model_uri model_tf.1_lstm  --save_folder "ztest/"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do fit     --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models  --do test  --model_uri model_tf.1_lstm  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do fit  --config_file dataset/json/benchmark_timeseries/gluonts_m4.json --config_mode "deepar"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do fit  --config_file dataset/json/benchmark_timeseries/gluonts_m5.json --config_mode "deepar"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_models --do test  --model_uri "example/custom_model/1_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_optim --do search  --config_file template/optim_config.json  --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 32, in <module>
    from mlmodels.models import model_create, module_load
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_optim --do search  --config_file template/optim_config_prune.json   --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 32, in <module>
    from mlmodels.models import model_create, module_load
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 32, in <module>
    from mlmodels.models import model_create, module_load
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test02/model_list.json  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_benchmark", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_benchmark')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 37, in <module>
    from mlmodels.models import module_load
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level





 ********************************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_benchmark", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_benchmark')()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
    return get_distribution(dist).load_entry_point(group, name)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
    return ep.load()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
    return self.resolve()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 37, in <module>
    from mlmodels.models import module_load
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 92
    else :
         ^
IndentationError: unindent does not match any outer indentation level
