
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd67c613373df800885b9f1e6941d6f5879aa2c04', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d67c613373df800885b9f1e6941d6f5879aa2c04

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 638, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 363, in test_cli
    with open( fileconfig, mode="r" ) as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md'
