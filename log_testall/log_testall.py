
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_all 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.armdn', 'model_keras.deepctr', 'model_keras.textcnn', 'model_keras.namentity_crm_bilstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_tch.matchZoo', 'model_tch.torchhub', 'model_tch.textcnn', 'model_tch.transformer_sentence', 'model_tf.1_lstm', 'model_dev.temporal_fusion_google'] 

  Used ['model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.armdn', 'model_keras.deepctr', 'model_keras.textcnn', 'model_keras.namentity_crm_bilstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_tch.matchZoo', 'model_tch.torchhub', 'model_tch.textcnn', 'model_tch.transformer_sentence', 'model_tf.1_lstm', 'model_dev.temporal_fusion_google'] 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 373, in <module>
    test(pars_choice="json", data_path= f"dataset/json/refactor/charcnn.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 330, in test
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 266, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/charcnn.json'
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 642, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 509, in test_all
    log_remote_push()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 154, in log_remote_push
    tag = "m_" + str(arg.name)
AttributeError: 'NoneType' object has no attribute 'name'
