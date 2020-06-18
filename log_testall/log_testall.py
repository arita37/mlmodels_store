
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_all 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_dev.temporal_fusion_google', 'model_gluon.gluonts_model_old', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tf.1_lstm', 'model_tch.textcnn', 'model_tch.torchhub', 'model_tch.transformer_sentence', 'model_tch.matchZoo', 'model_keras.charcnn', 'model_keras.textcnn', 'model_keras.charcnn_zhang', 'model_keras.deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.armdn'] 

  Used ['model_dev.temporal_fusion_google', 'model_gluon.gluonts_model_old', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tf.1_lstm', 'model_tch.textcnn', 'model_tch.torchhub', 'model_tch.transformer_sentence', 'model_tch.matchZoo', 'model_keras.charcnn', 'model_keras.textcnn', 'model_keras.charcnn_zhang', 'model_keras.deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.armdn'] 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_dev//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_dev//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 642, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 509, in test_all
    log_remote_push()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 154, in log_remote_push
    tag = "m_" + str(arg.name)
AttributeError: 'NoneType' object has no attribute 'name'
