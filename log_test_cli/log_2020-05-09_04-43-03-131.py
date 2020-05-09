  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_cli', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_cli') 
  ('# Testing Command Line System  ',) 
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 553, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 261, in test_cli
    log_info_repo(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 72, in log_info_repo
    dd = { k: locals()[k]  for k in [ "github_repo_url", "url_branch_file", "repo", "branch", "sha", "workflow"  ]}
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 72, in <dictcomp>
    dd = { k: locals()[k]  for k in [ "github_repo_url", "url_branch_file", "repo", "branch", "sha", "workflow"  ]}
KeyError: 'github_repo_url'
