
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '1f9d17efb377b015a330ae693d8b5624cdd08eef', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f9d17efb377b015a330ae693d8b5624cdd08eef

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 253, in compute
    log("cls_name :", cls_name, flush=True)
TypeError: log() got an unexpected keyword argument 'flush'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 253, in compute
    log("cls_name :", cls_name, flush=True)
TypeError: log() got an unexpected keyword argument 'flush'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 510, in <module>
    main()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 498, in main
    test_dataloader('dataset/json/refactor/')   
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 415, in test_dataloader
    test_json_list(data_pars_list)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 453, in test_json_list
    log("Error", f,  e, flush=True)    
TypeError: log() got an unexpected keyword argument 'flush'
