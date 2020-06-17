
  test_functions /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_functions', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_functions 

  

 #################### {'uri': 'mlmodels.util.log', 'args': ['x1', 'passed'], 'kw_args': {}} 

    Missing :   in  uri_name module_name:function_or_class  {'uri': 'mlmodels.util.log', 'args': ['x1', 'passed'], 'kw_args': {}} 

  

 #################### {'uri': 'mlmodels.util:log', 'args': ['x1', 'passed'], 'kw_args': {}} 

  <function log at 0x7f1c47942730> 

  x1 passed 

  None 

  

 #################### {'uri': 'mlmodels.util:load_function_uri', 'args': [], 'kw_args': {'uri_name': 'mlmodels.util:log'}} 

  <function load_function_uri at 0x7f1c4794f0d0> 

  <function log at 0x7f1c47942730> 

  

 #################### {'uri': 'mlmodels.util:os_package_root_path', 'args': [], 'kw_args': {}} 

  <function os_package_root_path at 0x7f1c479428c8> 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ 

  

 #################### {'uri': 'gluonts.distribution.neg_binomial:NegativeBinomialOutput', 'args': [], 'kw_args': {}} 

  <class 'gluonts.distribution.neg_binomial.NegativeBinomialOutput'> 

  gluonts.distribution.neg_binomial.NegativeBinomialOutput() 

  

 #################### {'uri': 'mlmodels.data:download_googledrive', 'args': [[{'fileid': '1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4', 'path_target': 'ztest/covid19/test.json'}, {'fileid': '1-8Ij1ZXL9YmQRylwRloABdqnxEC1mhP_', 'path_target': 'ztest/covid19/train.json'}]], 'kw_args': {}} 

  <function download_googledrive at 0x7f1c1027c2f0> 
Downloading...
From: https://drive.google.com/uc?id=1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4
To: /home/runner/work/mlmodels/mlmodels/ztest/covid19/test.json
0.00B [00:00, ?B/s]2.95MB [00:00, 45.6MB/s]
Downloading...
From: https://drive.google.com/uc?id=1-8Ij1ZXL9YmQRylwRloABdqnxEC1mhP_
To: /home/runner/work/mlmodels/mlmodels/ztest/covid19/train.json
0.00B [00:00, ?B/s]2.52MB [00:00, 42.7MB/s]
  ztest/covid19/train.json 

