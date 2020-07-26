
  test_functions /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_functions', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_functions 

  

 #################### {'uri': 'mlmodels.util.log', 'args': ['x1', 'passed'], 'kw_args': {}} 

    Missing :   in  uri_name module_name:function_or_class  {'uri': 'mlmodels.util.log', 'args': ['x1', 'passed'], 'kw_args': {}} 

  

 #################### {'uri': 'mlmodels.util:log', 'args': ['x1', 'passed'], 'kw_args': {}} 

  <function log at 0x7ff403d557b8> 

  x1 passed 

  None 

  

 #################### {'uri': 'mlmodels.util:load_function_uri', 'args': [], 'kw_args': {'uri_name': 'mlmodels.util:log'}} 

  <function load_function_uri at 0x7ff403d65158> 

  <function log at 0x7ff403d557b8> 

  

 #################### {'uri': 'mlmodels.util:os_package_root_path', 'args': [], 'kw_args': {}} 

  <function os_package_root_path at 0x7ff403d55950> 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ 

  

 #################### {'uri': 'gluonts.distribution.neg_binomial:NegativeBinomialOutput', 'args': [], 'kw_args': {}} 

  <class 'gluonts.distribution.neg_binomial.NegativeBinomialOutput'> 

  gluonts.distribution.neg_binomial.NegativeBinomialOutput() 

  

 #################### {'uri': 'mlmodels.data:download_gogledrive', 'args': [[{'fileid': '1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4', 'path_target': 'ztest/covid19/test.json'}, {'fileid': '1-8Ij1ZXL9YmQRylwRloABdqnxEC1mhP_', 'path_target': 'ztest/covid19/train.json'}]], 'kw_args': {}} 

  Module ['mlmodels.data', 'download_gogledrive'] notfound, invalid syntax (data.py, line 126), tuple index out of range {'uri': 'mlmodels.data:download_gogledrive', 'args': [[{'fileid': '1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4', 'path_target': 'ztest/covid19/test.json'}, {'fileid': '1-8Ij1ZXL9YmQRylwRloABdqnxEC1mhP_', 'path_target': 'ztest/covid19/train.json'}]], 'kw_args': {}} 

  

 #################### {'uri': 'mlmodels.model_tch.textcnn:analyze_datainfo_paths', 'args': [{'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}], 'kw_args': {}} 

  <function analyze_datainfo_paths at 0x7ff373b847b8> 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/train/IMDB_sample.csv', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/test/IMDB_sample.csv') 

  

 #################### {'uri': 'mlmodels.models:module_load', 'args': ['model_tch.torchhub.py'], 'kw_args': {}} 

  <function module_load at 0x7ff373b86840> 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  

 #################### {'uri': 'mlmodels.models:module_load_full', 'args': ['model_tch.textcnn.py', {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, {'data_info': {'dataset': 'mlmodels/dataset/text/imdb', 'pass_data_pars': False, 'train': True, 'maxlen': 40, 'max_features': 5}, 'preprocessors': [{'name': 'loader', 'uri': 'mlmodels.preprocess.generic:NumpyDataset', 'args': {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"}}, {'name': 'imdb_process', 'uri': 'mlmodels.preprocess.text_keras:IMDBDataset', 'args': {'num_words': 5}}]}, {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}], 'kw_args': {}} 

  <function module_load_full at 0x7ff373b868c8> 
{'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}

  'kernel_height' {'uri': 'mlmodels.models:module_load_full', 'args': ['model_tch.textcnn.py', {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, {'data_info': {'dataset': 'mlmodels/dataset/text/imdb', 'pass_data_pars': False, 'train': True, 'maxlen': 40, 'max_features': 5}, 'preprocessors': [{'name': 'loader', 'uri': 'mlmodels.preprocess.generic:NumpyDataset', 'args': {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"}}, {'name': 'imdb_process', 'uri': 'mlmodels.preprocess.text_keras:IMDBDataset', 'args': {'num_words': 5}}]}, {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}], 'kw_args': {}} 

  

 #################### {'uri': 'mlmodels.util:path_norm', 'args': ['model_keras.charcnn.py'], 'kw_args': {}} 

  <function path_norm at 0x7ff403d55ea0> 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras.charcnn.py 

  

 #################### {'uri': 'mlmodels.util:path_norm_dict', 'args': [{'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}], 'kw_args': {}} 

  <function path_norm_dict at 0x7ff403d55f28> 

  {'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} 
