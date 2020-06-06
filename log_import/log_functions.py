
  test_functions /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_functions', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_functions 
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 638, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 177, in test_functions
    dd   = json.load(open( path ))['test']
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jsoncomment/comments.py", line 58, in load
    return self.loads(jsonf.read(), *args, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jsoncomment/comments.py", line 48, in loads
    self.obj = self.wrapped.loads(jsons, *args, **kwargs)
ValueError: Unexpected character in found when decoding object value
