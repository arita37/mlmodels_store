
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_all 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1cb4f2a40d6a2bdba8af9def8a37ee0987eff43a', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1cb4f2a40d6a2bdba8af9def8a37ee0987eff43a', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1cb4f2a40d6a2bdba8af9def8a37ee0987eff43a

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1cb4f2a40d6a2bdba8af9def8a37ee0987eff43a

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/1cb4f2a40d6a2bdba8af9def8a37ee0987eff43a

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_tch.transformer_sentence', 'model_tch.textcnn', 'model_tch.torchhub', 'model_tch.matchZoo', 'model_gluon.gluonts_model_old', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_tf.1_lstm', 'model_dev.temporal_fusion_google', 'model_keras.armdn', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.charcnn_zhang', 'model_keras.textcnn', 'model_keras.Autokeras', 'model_keras.deepctr'] 

  Used ['model_tch.transformer_sentence', 'model_tch.textcnn', 'model_tch.torchhub', 'model_tch.matchZoo', 'model_gluon.gluonts_model_old', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_tf.1_lstm', 'model_dev.temporal_fusion_google', 'model_keras.armdn', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.charcnn_zhang', 'model_keras.textcnn', 'model_keras.Autokeras', 'model_keras.deepctr'] 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Loading dataset   ############################################# 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 459, in <module>
    test(pars_choice="json", data_path="model_tch/transformer_sentence.json", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 414, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 219, in get_dataset
    from mlmodels.dataloader import DataLoader
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 318
    else :
         ^
IndentationError: unindent does not match any outer indentation level
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_test", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 655, in main
    globals()[arg.do](arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 509, in test_all
    log_remote_push()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 154, in log_remote_push
    tag = "m_" + str(arg.name)
AttributeError: 'NoneType' object has no attribute 'name'
