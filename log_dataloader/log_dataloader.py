
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'ce3071beb7006ef52315b8a20bcbe252eb8bee27', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/ce3071beb7006ef52315b8a20bcbe252eb8bee27

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'> 
cls_name : Data

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'> 
cls_name : Data

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz

  URL:  mlmodels.preprocess.text_keras:IMDBDataset {'num_words': 5} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.IMDBDataset'> 
cls_name : IMDBDataset

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  
 #####  get_Data DataLoader  

  ((array([list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       ...,
       list([1, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])],
      dtype=object), array([1, 1, 1, ..., 0, 0, 0]), array([list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       ...,
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
       list([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2]),
       list([1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])],
      dtype=object), array([1, 0, 0, ..., 0, 1, 0])), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fd20ad132f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fd20ad132f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fd27cd630d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fd27cd630d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fd2144f89d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fd2144f89d8> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd22a6478c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd22a6478c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd22a6478c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 297, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 92, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2678784/9912422 [00:00<00:00, 26685876.04it/s]9920512it [00:00, 34623920.89it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 280301.75it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7642585.09it/s]          
0it [00:00, ?it/s]8192it [00:00, 176679.46it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd20d020978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd20d020fd0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd22a647510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd22a647510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fd22a647510> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:07,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:06,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:06,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:45,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:45,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:45,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:44,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:44,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:44,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:19,  6.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  9.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  9.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 12.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 12.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 28.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 43.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 77.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 81.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.40 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.72s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.40 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.72s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.72s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.30 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.72s/ url]
0 examples [00:00, ? examples/s]2020-05-31 06:09:35.350015: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-31 06:09:35.364324: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-31 06:09:35.364517: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55596be24070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-31 06:09:35.364539: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
25 examples [00:00, 248.34 examples/s]121 examples [00:00, 319.26 examples/s]218 examples [00:00, 399.65 examples/s]317 examples [00:00, 486.41 examples/s]415 examples [00:00, 572.21 examples/s]511 examples [00:00, 649.99 examples/s]606 examples [00:00, 717.25 examples/s]703 examples [00:00, 777.11 examples/s]801 examples [00:00, 827.32 examples/s]896 examples [00:01, 859.65 examples/s]992 examples [00:01, 885.11 examples/s]1088 examples [00:01, 905.13 examples/s]1182 examples [00:01, 911.17 examples/s]1276 examples [00:01, 913.89 examples/s]1371 examples [00:01, 923.33 examples/s]1465 examples [00:01, 916.65 examples/s]1558 examples [00:01, 918.50 examples/s]1658 examples [00:01, 939.94 examples/s]1753 examples [00:01, 935.27 examples/s]1848 examples [00:02, 937.65 examples/s]1943 examples [00:02, 939.22 examples/s]2038 examples [00:02, 937.00 examples/s]2132 examples [00:02, 933.19 examples/s]2226 examples [00:02, 931.25 examples/s]2320 examples [00:02, 910.49 examples/s]2412 examples [00:02, 911.69 examples/s]2504 examples [00:02, 906.95 examples/s]2602 examples [00:02, 927.43 examples/s]2695 examples [00:02, 923.30 examples/s]2788 examples [00:03, 889.95 examples/s]2878 examples [00:03, 855.83 examples/s]2968 examples [00:03, 867.32 examples/s]3057 examples [00:03, 872.98 examples/s]3145 examples [00:03, 873.38 examples/s]3233 examples [00:03, 850.27 examples/s]3324 examples [00:03, 866.64 examples/s]3411 examples [00:03, 865.27 examples/s]3501 examples [00:03, 874.72 examples/s]3590 examples [00:03, 876.83 examples/s]3684 examples [00:04, 892.43 examples/s]3774 examples [00:04, 892.94 examples/s]3864 examples [00:04, 889.48 examples/s]3954 examples [00:04, 883.94 examples/s]4043 examples [00:04, 879.41 examples/s]4133 examples [00:04, 870.26 examples/s]4222 examples [00:04, 875.21 examples/s]4310 examples [00:04, 870.09 examples/s]4398 examples [00:04, 853.45 examples/s]4484 examples [00:05, 839.01 examples/s]4574 examples [00:05, 854.25 examples/s]4660 examples [00:05, 841.99 examples/s]4745 examples [00:05, 842.70 examples/s]4832 examples [00:05, 849.55 examples/s]4921 examples [00:05, 860.63 examples/s]5014 examples [00:05, 878.49 examples/s]5103 examples [00:05, 877.27 examples/s]5191 examples [00:05, 876.60 examples/s]5289 examples [00:05, 903.60 examples/s]5382 examples [00:06, 910.39 examples/s]5474 examples [00:06, 891.83 examples/s]5564 examples [00:06, 890.77 examples/s]5654 examples [00:06, 874.13 examples/s]5742 examples [00:06, 843.22 examples/s]5834 examples [00:06, 864.69 examples/s]5929 examples [00:06, 887.63 examples/s]6019 examples [00:06, 881.04 examples/s]6109 examples [00:06, 884.29 examples/s]6198 examples [00:06, 878.73 examples/s]6288 examples [00:07, 882.88 examples/s]6377 examples [00:07, 884.52 examples/s]6466 examples [00:07, 877.69 examples/s]6554 examples [00:07, 866.81 examples/s]6643 examples [00:07, 871.88 examples/s]6731 examples [00:07, 863.88 examples/s]6819 examples [00:07, 868.01 examples/s]6906 examples [00:07, 855.56 examples/s]7003 examples [00:07, 884.96 examples/s]7094 examples [00:07, 889.93 examples/s]7184 examples [00:08, 879.94 examples/s]7273 examples [00:08, 874.55 examples/s]7361 examples [00:08, 863.52 examples/s]7448 examples [00:08, 861.30 examples/s]7541 examples [00:08, 880.07 examples/s]7637 examples [00:08, 902.21 examples/s]7728 examples [00:08, 896.26 examples/s]7818 examples [00:08, 871.06 examples/s]7907 examples [00:08, 875.51 examples/s]8002 examples [00:09, 896.31 examples/s]8094 examples [00:09, 902.33 examples/s]8191 examples [00:09, 920.37 examples/s]8284 examples [00:09, 922.55 examples/s]8377 examples [00:09, 923.52 examples/s]8470 examples [00:09, 911.15 examples/s]8562 examples [00:09, 897.22 examples/s]8656 examples [00:09, 908.81 examples/s]8748 examples [00:09, 889.36 examples/s]8842 examples [00:09, 902.21 examples/s]8935 examples [00:10, 908.34 examples/s]9029 examples [00:10, 915.27 examples/s]9121 examples [00:10, 915.39 examples/s]9213 examples [00:10, 909.48 examples/s]9308 examples [00:10, 918.84 examples/s]9402 examples [00:10, 924.87 examples/s]9495 examples [00:10, 915.86 examples/s]9587 examples [00:10, 908.28 examples/s]9678 examples [00:10, 893.90 examples/s]9768 examples [00:10, 874.02 examples/s]9856 examples [00:11, 861.36 examples/s]9943 examples [00:11, 860.72 examples/s]10030 examples [00:11, 790.49 examples/s]10114 examples [00:11, 802.35 examples/s]10199 examples [00:11, 815.38 examples/s]10284 examples [00:11, 824.94 examples/s]10367 examples [00:11, 818.43 examples/s]10455 examples [00:11, 834.88 examples/s]10546 examples [00:11, 853.84 examples/s]10638 examples [00:12, 871.99 examples/s]10731 examples [00:12, 885.99 examples/s]10821 examples [00:12, 888.99 examples/s]10912 examples [00:12, 894.89 examples/s]11002 examples [00:12, 887.87 examples/s]11096 examples [00:12, 900.30 examples/s]11189 examples [00:12, 906.39 examples/s]11280 examples [00:12, 875.32 examples/s]11368 examples [00:12, 874.00 examples/s]11459 examples [00:12, 882.50 examples/s]11548 examples [00:13, 884.00 examples/s]11643 examples [00:13, 900.40 examples/s]11734 examples [00:13, 849.42 examples/s]11820 examples [00:13, 842.62 examples/s]11906 examples [00:13, 847.03 examples/s]11995 examples [00:13, 859.10 examples/s]12082 examples [00:13, 847.24 examples/s]12167 examples [00:13, 846.58 examples/s]12252 examples [00:13, 839.12 examples/s]12341 examples [00:13, 852.74 examples/s]12434 examples [00:14, 872.64 examples/s]12530 examples [00:14, 894.79 examples/s]12620 examples [00:14, 893.79 examples/s]12712 examples [00:14, 899.03 examples/s]12803 examples [00:14, 900.35 examples/s]12894 examples [00:14, 892.97 examples/s]12984 examples [00:14, 882.41 examples/s]13073 examples [00:14, 871.47 examples/s]13161 examples [00:14, 837.30 examples/s]13259 examples [00:14, 875.32 examples/s]13354 examples [00:15, 895.78 examples/s]13447 examples [00:15, 904.71 examples/s]13538 examples [00:15, 872.91 examples/s]13626 examples [00:15, 851.58 examples/s]13720 examples [00:15, 875.12 examples/s]13810 examples [00:15, 880.52 examples/s]13906 examples [00:15, 900.53 examples/s]14002 examples [00:15, 917.23 examples/s]14100 examples [00:15, 933.21 examples/s]14194 examples [00:16, 932.99 examples/s]14291 examples [00:16, 942.68 examples/s]14390 examples [00:16, 953.98 examples/s]14491 examples [00:16, 969.63 examples/s]14589 examples [00:16, 967.00 examples/s]14686 examples [00:16, 923.98 examples/s]14779 examples [00:16, 890.37 examples/s]14869 examples [00:16, 859.59 examples/s]14956 examples [00:16, 840.27 examples/s]15041 examples [00:16, 831.94 examples/s]15127 examples [00:17, 838.22 examples/s]15224 examples [00:17, 872.06 examples/s]15321 examples [00:17, 897.83 examples/s]15419 examples [00:17, 919.16 examples/s]15514 examples [00:17, 926.89 examples/s]15608 examples [00:17, 924.59 examples/s]15707 examples [00:17, 942.77 examples/s]15803 examples [00:17, 947.52 examples/s]15898 examples [00:17, 944.78 examples/s]15993 examples [00:17, 927.80 examples/s]16086 examples [00:18, 867.58 examples/s]16174 examples [00:18, 867.16 examples/s]16262 examples [00:18, 861.12 examples/s]16355 examples [00:18, 880.31 examples/s]16455 examples [00:18, 912.94 examples/s]16551 examples [00:18, 923.97 examples/s]16646 examples [00:18, 929.81 examples/s]16744 examples [00:18, 942.14 examples/s]16839 examples [00:18, 931.22 examples/s]16933 examples [00:19, 914.33 examples/s]17025 examples [00:19, 902.09 examples/s]17116 examples [00:19, 899.87 examples/s]17207 examples [00:19, 895.11 examples/s]17298 examples [00:19, 898.51 examples/s]17393 examples [00:19, 912.19 examples/s]17489 examples [00:19, 923.26 examples/s]17582 examples [00:19, 894.33 examples/s]17673 examples [00:19, 895.60 examples/s]17767 examples [00:19, 908.10 examples/s]17863 examples [00:20, 921.82 examples/s]17961 examples [00:20, 936.70 examples/s]18059 examples [00:20, 947.23 examples/s]18154 examples [00:20, 943.72 examples/s]18249 examples [00:20, 916.39 examples/s]18341 examples [00:20, 909.51 examples/s]18433 examples [00:20, 866.15 examples/s]18521 examples [00:20, 849.13 examples/s]18616 examples [00:20, 875.31 examples/s]18714 examples [00:21, 903.78 examples/s]18811 examples [00:21, 921.03 examples/s]18904 examples [00:21, 922.41 examples/s]18998 examples [00:21, 925.03 examples/s]19091 examples [00:21, 895.93 examples/s]19183 examples [00:21, 902.29 examples/s]19278 examples [00:21, 914.26 examples/s]19370 examples [00:21, 906.01 examples/s]19461 examples [00:21, 899.58 examples/s]19552 examples [00:21, 897.30 examples/s]19645 examples [00:22, 903.30 examples/s]19736 examples [00:22, 902.94 examples/s]19837 examples [00:22, 930.40 examples/s]19934 examples [00:22, 940.21 examples/s]20029 examples [00:22, 908.76 examples/s]20125 examples [00:22, 922.62 examples/s]20221 examples [00:22, 932.73 examples/s]20315 examples [00:22, 913.57 examples/s]20407 examples [00:22, 893.89 examples/s]20497 examples [00:22, 853.67 examples/s]20583 examples [00:23, 827.68 examples/s]20667 examples [00:23, 828.38 examples/s]20751 examples [00:23, 809.21 examples/s]20845 examples [00:23, 844.22 examples/s]20937 examples [00:23, 864.36 examples/s]21025 examples [00:23, 865.80 examples/s]21112 examples [00:23, 862.43 examples/s]21203 examples [00:23, 875.50 examples/s]21293 examples [00:23, 881.57 examples/s]21390 examples [00:24, 904.65 examples/s]21484 examples [00:24, 914.00 examples/s]21576 examples [00:24, 909.62 examples/s]21668 examples [00:24, 902.94 examples/s]21759 examples [00:24, 895.63 examples/s]21856 examples [00:24, 914.29 examples/s]21948 examples [00:24, 882.09 examples/s]22037 examples [00:24, 854.78 examples/s]22128 examples [00:24, 868.30 examples/s]22227 examples [00:24, 900.87 examples/s]22319 examples [00:25, 905.49 examples/s]22413 examples [00:25, 914.25 examples/s]22508 examples [00:25, 924.23 examples/s]22605 examples [00:25, 935.02 examples/s]22699 examples [00:25, 928.50 examples/s]22792 examples [00:25, 926.96 examples/s]22888 examples [00:25, 933.78 examples/s]22982 examples [00:25, 911.37 examples/s]23074 examples [00:25, 898.52 examples/s]23165 examples [00:25, 891.91 examples/s]23255 examples [00:26, 885.23 examples/s]23344 examples [00:26, 865.78 examples/s]23431 examples [00:26, 842.98 examples/s]23521 examples [00:26, 857.95 examples/s]23616 examples [00:26, 882.07 examples/s]23705 examples [00:26, 864.82 examples/s]23792 examples [00:26, 856.76 examples/s]23878 examples [00:26, 851.27 examples/s]23965 examples [00:26, 853.01 examples/s]24051 examples [00:27, 849.11 examples/s]24137 examples [00:27, 844.88 examples/s]24222 examples [00:27, 838.57 examples/s]24309 examples [00:27, 847.52 examples/s]24401 examples [00:27, 867.01 examples/s]24497 examples [00:27, 890.83 examples/s]24591 examples [00:27, 904.16 examples/s]24682 examples [00:27, 902.94 examples/s]24773 examples [00:27, 900.30 examples/s]24864 examples [00:27, 867.16 examples/s]24954 examples [00:28, 875.81 examples/s]25042 examples [00:28, 873.08 examples/s]25130 examples [00:28, 862.30 examples/s]25218 examples [00:28, 866.80 examples/s]25311 examples [00:28, 882.65 examples/s]25404 examples [00:28, 896.15 examples/s]25494 examples [00:28, 889.94 examples/s]25584 examples [00:28, 869.45 examples/s]25672 examples [00:28, 859.18 examples/s]25759 examples [00:28, 828.75 examples/s]25843 examples [00:29, 829.83 examples/s]25927 examples [00:29, 832.75 examples/s]26019 examples [00:29, 856.55 examples/s]26112 examples [00:29, 875.44 examples/s]26200 examples [00:29, 872.69 examples/s]26288 examples [00:29, 857.11 examples/s]26374 examples [00:29, 850.86 examples/s]26466 examples [00:29, 868.16 examples/s]26559 examples [00:29, 882.88 examples/s]26650 examples [00:29, 888.61 examples/s]26740 examples [00:30, 882.09 examples/s]26829 examples [00:30, 877.77 examples/s]26920 examples [00:30, 884.60 examples/s]27009 examples [00:30, 879.83 examples/s]27098 examples [00:30, 872.06 examples/s]27191 examples [00:30, 887.54 examples/s]27283 examples [00:30, 896.30 examples/s]27382 examples [00:30, 922.36 examples/s]27479 examples [00:30, 934.78 examples/s]27573 examples [00:31, 912.44 examples/s]27665 examples [00:31, 887.90 examples/s]27759 examples [00:31, 900.73 examples/s]27855 examples [00:31, 916.44 examples/s]27947 examples [00:31, 916.58 examples/s]28045 examples [00:31, 932.88 examples/s]28139 examples [00:31, 927.56 examples/s]28232 examples [00:31, 909.26 examples/s]28324 examples [00:31, 907.42 examples/s]28417 examples [00:31, 911.91 examples/s]28515 examples [00:32, 930.08 examples/s]28612 examples [00:32, 941.64 examples/s]28708 examples [00:32, 946.85 examples/s]28805 examples [00:32, 951.18 examples/s]28901 examples [00:32, 943.78 examples/s]28996 examples [00:32, 943.97 examples/s]29092 examples [00:32, 946.48 examples/s]29187 examples [00:32, 905.12 examples/s]29278 examples [00:32, 898.40 examples/s]29371 examples [00:32, 905.58 examples/s]29466 examples [00:33, 916.25 examples/s]29559 examples [00:33, 920.18 examples/s]29652 examples [00:33, 920.19 examples/s]29745 examples [00:33, 896.82 examples/s]29835 examples [00:33, 880.93 examples/s]29924 examples [00:33, 864.45 examples/s]30011 examples [00:33, 826.37 examples/s]30103 examples [00:33, 850.57 examples/s]30198 examples [00:33, 875.86 examples/s]30290 examples [00:34, 887.79 examples/s]30390 examples [00:34, 917.38 examples/s]30483 examples [00:34, 920.73 examples/s]30576 examples [00:34, 909.99 examples/s]30668 examples [00:34, 903.52 examples/s]30759 examples [00:34, 898.84 examples/s]30851 examples [00:34, 904.04 examples/s]30948 examples [00:34, 921.11 examples/s]31041 examples [00:34, 906.84 examples/s]31132 examples [00:34, 906.60 examples/s]31228 examples [00:35, 920.62 examples/s]31321 examples [00:35, 921.28 examples/s]31414 examples [00:35, 912.75 examples/s]31506 examples [00:35, 901.94 examples/s]31597 examples [00:35, 891.35 examples/s]31687 examples [00:35, 893.64 examples/s]31782 examples [00:35, 907.45 examples/s]31875 examples [00:35, 912.27 examples/s]31967 examples [00:35, 911.22 examples/s]32063 examples [00:35, 924.68 examples/s]32158 examples [00:36, 931.82 examples/s]32252 examples [00:36, 886.29 examples/s]32342 examples [00:36, 880.44 examples/s]32433 examples [00:36, 888.70 examples/s]32523 examples [00:36, 888.19 examples/s]32613 examples [00:36, 875.59 examples/s]32701 examples [00:36, 875.03 examples/s]32789 examples [00:36, 868.55 examples/s]32878 examples [00:36, 873.97 examples/s]32975 examples [00:36, 899.35 examples/s]33066 examples [00:37, 882.62 examples/s]33155 examples [00:37, 881.89 examples/s]33244 examples [00:37, 879.81 examples/s]33333 examples [00:37, 872.46 examples/s]33421 examples [00:37, 858.15 examples/s]33514 examples [00:37, 877.47 examples/s]33603 examples [00:37, 880.47 examples/s]33692 examples [00:37, 856.66 examples/s]33780 examples [00:37, 862.58 examples/s]33875 examples [00:38, 885.27 examples/s]33972 examples [00:38, 908.43 examples/s]34064 examples [00:38, 910.50 examples/s]34156 examples [00:38, 902.79 examples/s]34247 examples [00:38, 890.78 examples/s]34337 examples [00:38, 891.49 examples/s]34427 examples [00:38, 889.74 examples/s]34517 examples [00:38, 862.94 examples/s]34604 examples [00:38, 852.46 examples/s]34692 examples [00:38, 858.09 examples/s]34779 examples [00:39, 859.87 examples/s]34866 examples [00:39, 847.57 examples/s]34955 examples [00:39, 858.37 examples/s]35044 examples [00:39, 865.63 examples/s]35131 examples [00:39, 847.79 examples/s]35219 examples [00:39, 855.78 examples/s]35305 examples [00:39, 856.92 examples/s]35393 examples [00:39, 861.74 examples/s]35480 examples [00:39, 861.31 examples/s]35567 examples [00:39, 858.66 examples/s]35653 examples [00:40, 854.25 examples/s]35739 examples [00:40, 847.26 examples/s]35836 examples [00:40, 880.51 examples/s]35932 examples [00:40, 902.05 examples/s]36027 examples [00:40, 915.58 examples/s]36121 examples [00:40, 919.80 examples/s]36214 examples [00:40, 918.04 examples/s]36306 examples [00:40, 909.20 examples/s]36398 examples [00:40, 897.82 examples/s]36488 examples [00:40, 884.05 examples/s]36584 examples [00:41, 903.54 examples/s]36678 examples [00:41, 913.51 examples/s]36773 examples [00:41, 921.36 examples/s]36866 examples [00:41, 901.25 examples/s]36957 examples [00:41, 890.81 examples/s]37047 examples [00:41, 892.34 examples/s]37141 examples [00:41, 903.57 examples/s]37232 examples [00:41, 898.56 examples/s]37322 examples [00:41, 877.28 examples/s]37410 examples [00:42, 865.59 examples/s]37497 examples [00:42, 857.67 examples/s]37583 examples [00:42, 849.77 examples/s]37669 examples [00:42, 846.58 examples/s]37759 examples [00:42, 860.65 examples/s]37846 examples [00:42, 862.97 examples/s]37937 examples [00:42, 874.31 examples/s]38025 examples [00:42, 854.31 examples/s]38111 examples [00:42, 850.12 examples/s]38197 examples [00:42, 816.89 examples/s]38280 examples [00:43, 808.40 examples/s]38367 examples [00:43, 825.00 examples/s]38460 examples [00:43, 852.27 examples/s]38554 examples [00:43, 876.64 examples/s]38643 examples [00:43, 880.22 examples/s]38736 examples [00:43, 892.82 examples/s]38826 examples [00:43, 889.36 examples/s]38916 examples [00:43, 886.23 examples/s]39005 examples [00:43, 870.78 examples/s]39093 examples [00:43, 851.53 examples/s]39179 examples [00:44, 835.60 examples/s]39272 examples [00:44, 861.68 examples/s]39360 examples [00:44, 866.57 examples/s]39450 examples [00:44, 874.95 examples/s]39544 examples [00:44, 891.15 examples/s]39635 examples [00:44, 896.65 examples/s]39727 examples [00:44, 902.68 examples/s]39818 examples [00:44, 875.81 examples/s]39911 examples [00:44, 891.19 examples/s]40001 examples [00:45, 855.73 examples/s]40088 examples [00:45, 848.00 examples/s]40181 examples [00:45, 869.63 examples/s]40271 examples [00:45, 877.52 examples/s]40360 examples [00:45, 871.59 examples/s]40453 examples [00:45, 886.86 examples/s]40546 examples [00:45, 899.15 examples/s]40640 examples [00:45, 909.89 examples/s]40733 examples [00:45, 914.97 examples/s]40825 examples [00:45, 870.58 examples/s]40914 examples [00:46, 874.36 examples/s]41002 examples [00:46, 871.14 examples/s]41090 examples [00:46, 849.41 examples/s]41181 examples [00:46, 865.25 examples/s]41271 examples [00:46, 874.03 examples/s]41359 examples [00:46, 872.93 examples/s]41449 examples [00:46, 879.90 examples/s]41541 examples [00:46, 890.24 examples/s]41631 examples [00:46, 891.74 examples/s]41725 examples [00:46, 904.42 examples/s]41822 examples [00:47, 921.46 examples/s]41918 examples [00:47, 930.60 examples/s]42017 examples [00:47, 946.00 examples/s]42116 examples [00:47, 957.53 examples/s]42215 examples [00:47, 964.84 examples/s]42312 examples [00:47, 938.97 examples/s]42407 examples [00:47, 917.74 examples/s]42500 examples [00:47, 885.63 examples/s]42589 examples [00:47, 885.48 examples/s]42678 examples [00:48, 878.65 examples/s]42768 examples [00:48, 883.01 examples/s]42857 examples [00:48, 879.34 examples/s]42946 examples [00:48, 881.82 examples/s]43035 examples [00:48, 870.75 examples/s]43123 examples [00:48, 865.44 examples/s]43213 examples [00:48, 875.29 examples/s]43303 examples [00:48, 880.10 examples/s]43392 examples [00:48, 857.61 examples/s]43480 examples [00:48, 863.66 examples/s]43567 examples [00:49, 863.77 examples/s]43657 examples [00:49, 873.15 examples/s]43745 examples [00:49, 857.19 examples/s]43834 examples [00:49, 866.66 examples/s]43930 examples [00:49, 892.66 examples/s]44021 examples [00:49, 896.10 examples/s]44115 examples [00:49, 906.57 examples/s]44206 examples [00:49, 869.33 examples/s]44294 examples [00:49, 718.75 examples/s]44378 examples [00:50, 750.50 examples/s]44474 examples [00:50, 802.51 examples/s]44569 examples [00:50, 840.96 examples/s]44662 examples [00:50, 864.82 examples/s]44755 examples [00:50, 883.10 examples/s]44848 examples [00:50, 896.05 examples/s]44945 examples [00:50, 915.34 examples/s]45038 examples [00:50, 907.47 examples/s]45130 examples [00:50, 879.79 examples/s]45219 examples [00:50, 880.37 examples/s]45309 examples [00:51, 884.83 examples/s]45400 examples [00:51, 890.97 examples/s]45500 examples [00:51, 920.27 examples/s]45598 examples [00:51, 935.74 examples/s]45695 examples [00:51, 943.97 examples/s]45793 examples [00:51, 949.98 examples/s]45890 examples [00:51, 953.76 examples/s]45986 examples [00:51, 941.55 examples/s]46081 examples [00:51, 914.56 examples/s]46175 examples [00:51, 919.97 examples/s]46268 examples [00:52, 906.14 examples/s]46359 examples [00:52, 903.62 examples/s]46450 examples [00:52, 881.41 examples/s]46539 examples [00:52, 831.81 examples/s]46623 examples [00:52, 799.56 examples/s]46704 examples [00:52, 759.79 examples/s]46781 examples [00:52, 760.17 examples/s]46868 examples [00:52, 788.08 examples/s]46957 examples [00:52, 813.92 examples/s]47052 examples [00:53, 850.21 examples/s]47142 examples [00:53, 863.69 examples/s]47234 examples [00:53, 878.16 examples/s]47324 examples [00:53, 882.98 examples/s]47416 examples [00:53, 891.72 examples/s]47509 examples [00:53, 902.04 examples/s]47606 examples [00:53, 921.06 examples/s]47699 examples [00:53, 922.30 examples/s]47793 examples [00:53, 925.36 examples/s]47891 examples [00:53, 939.08 examples/s]47986 examples [00:54, 917.07 examples/s]48078 examples [00:54, 900.00 examples/s]48171 examples [00:54, 907.01 examples/s]48262 examples [00:54, 897.35 examples/s]48355 examples [00:54, 904.59 examples/s]48446 examples [00:54, 891.14 examples/s]48540 examples [00:54, 904.47 examples/s]48631 examples [00:54, 902.69 examples/s]48729 examples [00:54, 922.71 examples/s]48829 examples [00:54, 942.89 examples/s]48924 examples [00:55, 921.35 examples/s]49022 examples [00:55, 937.54 examples/s]49121 examples [00:55, 950.96 examples/s]49218 examples [00:55, 954.37 examples/s]49314 examples [00:55, 904.57 examples/s]49406 examples [00:55, 896.76 examples/s]49497 examples [00:55, 883.35 examples/s]49586 examples [00:55, 837.90 examples/s]49675 examples [00:55, 851.17 examples/s]49764 examples [00:56, 860.73 examples/s]49856 examples [00:56, 876.90 examples/s]49951 examples [00:56, 897.57 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7179/50000 [00:00<00:00, 71785.11 examples/s] 38%|███▊      | 18880/50000 [00:00<00:00, 81199.69 examples/s] 61%|██████▏   | 30716/50000 [00:00<00:00, 89641.26 examples/s] 85%|████████▍ | 42371/50000 [00:00<00:00, 96311.88 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 745.46 examples/s]161 examples [00:00, 775.83 examples/s]242 examples [00:00, 784.78 examples/s]328 examples [00:00, 803.94 examples/s]416 examples [00:00, 824.02 examples/s]502 examples [00:00, 834.10 examples/s]598 examples [00:00, 868.10 examples/s]692 examples [00:00, 888.15 examples/s]788 examples [00:00, 905.72 examples/s]877 examples [00:01, 899.74 examples/s]966 examples [00:01, 896.43 examples/s]1055 examples [00:01, 889.70 examples/s]1147 examples [00:01, 897.59 examples/s]1237 examples [00:01, 894.99 examples/s]1327 examples [00:01, 876.35 examples/s]1415 examples [00:01, 655.13 examples/s]1500 examples [00:01, 702.88 examples/s]1589 examples [00:01, 749.56 examples/s]1677 examples [00:02, 783.70 examples/s]1764 examples [00:02, 806.75 examples/s]1852 examples [00:02, 823.51 examples/s]1941 examples [00:02, 840.93 examples/s]2031 examples [00:02, 857.64 examples/s]2119 examples [00:02, 858.36 examples/s]2215 examples [00:02, 886.32 examples/s]2311 examples [00:02, 906.18 examples/s]2405 examples [00:02, 914.43 examples/s]2497 examples [00:02, 913.91 examples/s]2589 examples [00:03, 905.43 examples/s]2680 examples [00:03, 887.05 examples/s]2769 examples [00:03, 871.07 examples/s]2857 examples [00:03, 860.98 examples/s]2944 examples [00:03, 850.41 examples/s]3033 examples [00:03, 858.81 examples/s]3126 examples [00:03, 877.47 examples/s]3221 examples [00:03, 896.55 examples/s]3315 examples [00:03, 907.75 examples/s]3410 examples [00:03, 918.16 examples/s]3502 examples [00:04, 912.83 examples/s]3598 examples [00:04, 925.73 examples/s]3693 examples [00:04, 932.37 examples/s]3787 examples [00:04, 930.79 examples/s]3881 examples [00:04, 918.65 examples/s]3976 examples [00:04, 927.51 examples/s]4069 examples [00:04, 885.10 examples/s]4158 examples [00:04, 852.33 examples/s]4251 examples [00:04, 871.91 examples/s]4347 examples [00:05, 894.03 examples/s]4439 examples [00:05, 900.94 examples/s]4530 examples [00:05, 877.54 examples/s]4619 examples [00:05, 870.76 examples/s]4707 examples [00:05, 856.42 examples/s]4793 examples [00:05, 837.87 examples/s]4882 examples [00:05, 851.74 examples/s]4974 examples [00:05, 870.01 examples/s]5063 examples [00:05, 874.42 examples/s]5151 examples [00:05, 872.26 examples/s]5239 examples [00:06, 868.62 examples/s]5326 examples [00:06, 867.20 examples/s]5422 examples [00:06, 890.65 examples/s]5516 examples [00:06, 903.08 examples/s]5612 examples [00:06, 917.45 examples/s]5704 examples [00:06, 914.44 examples/s]5802 examples [00:06, 931.63 examples/s]5898 examples [00:06, 938.60 examples/s]5996 examples [00:06, 948.19 examples/s]6091 examples [00:06, 946.65 examples/s]6186 examples [00:07, 929.32 examples/s]6280 examples [00:07, 904.70 examples/s]6371 examples [00:07, 898.96 examples/s]6462 examples [00:07, 894.30 examples/s]6557 examples [00:07, 910.05 examples/s]6656 examples [00:07, 930.36 examples/s]6753 examples [00:07, 940.41 examples/s]6853 examples [00:07, 956.76 examples/s]6949 examples [00:07, 925.63 examples/s]7042 examples [00:07, 925.52 examples/s]7138 examples [00:08, 935.27 examples/s]7232 examples [00:08, 913.42 examples/s]7324 examples [00:08, 902.89 examples/s]7415 examples [00:08, 896.41 examples/s]7507 examples [00:08, 901.56 examples/s]7608 examples [00:08, 929.43 examples/s]7704 examples [00:08, 936.39 examples/s]7803 examples [00:08, 950.78 examples/s]7901 examples [00:08, 959.19 examples/s]7998 examples [00:09, 954.32 examples/s]8096 examples [00:09, 959.95 examples/s]8195 examples [00:09, 967.68 examples/s]8293 examples [00:09, 971.02 examples/s]8391 examples [00:09, 954.15 examples/s]8487 examples [00:09, 930.90 examples/s]8581 examples [00:09, 912.32 examples/s]8673 examples [00:09, 897.43 examples/s]8763 examples [00:09, 881.03 examples/s]8852 examples [00:09, 880.32 examples/s]8941 examples [00:10, 875.93 examples/s]9035 examples [00:10, 891.73 examples/s]9129 examples [00:10, 904.24 examples/s]9220 examples [00:10, 869.51 examples/s]9316 examples [00:10, 893.85 examples/s]9408 examples [00:10, 901.39 examples/s]9499 examples [00:10, 897.25 examples/s]9594 examples [00:10, 911.74 examples/s]9686 examples [00:10, 893.72 examples/s]9776 examples [00:10, 856.64 examples/s]9864 examples [00:11, 860.93 examples/s]9953 examples [00:11, 869.12 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteORV3F1/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteORV3F1/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd22a6478c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd22a6478c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd22a6478c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd20b94ff60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd22a5cbbe0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd22a6478c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd22a6478c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd22a6478c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd20dc78400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd20dc782b0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fd201e67620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fd201e67620> 

  function with postional parmater data_info <function split_train_valid at 0x7fd201e67620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fd201e67730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fd201e67730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fd201e67730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2b71d7b7bf086f7e31bb6e6c39745e123b46ee60da8a8203f707e6ce69eb9694
  Stored in directory: /tmp/pip-ephem-wheel-cache-2ng2p0y1/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<141:05:57, 1.70kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<98:59:51, 2.42kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<69:20:26, 3.45kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<48:30:52, 4.93kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<33:51:20, 7.04kB/s].vector_cache/glove.6B.zip:   1%|          | 9.35M/862M [00:05<23:32:33, 10.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:05<16:25:18, 14.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:05<11:25:50, 20.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:05<7:58:04, 29.3kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:05<5:32:42, 41.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:05<3:52:03, 59.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:06<2:41:36, 85.4kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:06<1:52:42, 122kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:06<1:18:30, 174kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:06<54:48, 248kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:06<38:16, 353kB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.1M/862M [00:07<28:37, 470kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:07<20:28, 657kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:09<16:46, 799kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:09<14:53, 900kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:10<11:14, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:10<08:01, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:11<13:57, 955kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:11<11:09, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:12<08:08, 1.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:13<08:44, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:13<07:27, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:14<05:33, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:15<06:58, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:15<06:15, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:16<04:42, 2.80MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:17<06:23, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:17<07:12, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.7M/862M [00:17<05:36, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:18<04:05, 3.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:19<09:55, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:19<08:16, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:19<06:04, 2.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:21<07:17, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:21<07:46, 1.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:21<06:02, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:21<04:22, 2.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:23<12:00, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:23<09:44, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:23<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:25<07:59, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:25<06:54, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:25<05:06, 2.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:27<06:34, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:27<05:53, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:27<04:24, 2.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:29<06:06, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:29<06:53, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:29<05:28, 2.32MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:31<05:51, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:31<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<04:02, 3.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:33<05:47, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:33<06:39, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:33<05:13, 2.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<03:47, 3.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:35<16:00, 781kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:35<12:31, 998kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<09:00, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:37<09:13, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:37<09:01, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:37<06:56, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:39<06:50, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:39<06:03, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<04:32, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:41<06:03, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:41<06:44, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:41<05:20, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:43<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<03:58, 3.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:45<05:37, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:45<06:32, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:45<05:07, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<03:43, 3.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<11:43, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:47<09:28, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:47<06:53, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:49<07:37, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:49<06:34, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<04:53, 2.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<06:15, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<05:35, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<04:12, 2.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:53<05:45, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:53<06:31, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:53<05:04, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<03:42, 3.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<08:18, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:55<07:02, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<05:13, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<06:24, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:57<05:29, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:57<04:14, 2.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<03:05, 3.78MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<49:13, 237kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:59<35:37, 328kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<25:08, 463kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<20:18, 572kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<16:35, 700kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:01<12:11, 951kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<08:38, 1.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<11:09:53, 17.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<7:49:51, 24.6kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<5:28:27, 35.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<3:51:53, 49.5kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<2:43:12, 70.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<1:54:34, 100kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<1:19:59, 143kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<1:41:57, 112kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<1:12:30, 157kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<50:55, 223kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<38:13, 297kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<29:03, 390kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:08<20:52, 543kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<14:40, 768kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<20:45, 543kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<15:39, 719kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<11:12, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<10:26, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<08:16, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:12<04:22, 2.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<24:03, 463kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<19:15, 578kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<14:04, 790kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<09:57, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<40:03, 276kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<29:12, 379kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<20:41, 533kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<16:58, 648kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:18<13:03, 842kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<09:21, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<09:03, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<07:30, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:28, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<06:18, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<05:33, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<04:10, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:24<05:23, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:24<04:43, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<03:33, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<02:37, 4.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<22:04, 486kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<17:44, 604kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:26<12:58, 825kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<09:10, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<30:59, 344kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<22:37, 471kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<16:01, 663kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<11:20, 934kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<28:31, 371kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<21:05, 502kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<15:00, 703kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<12:53, 816kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<11:15, 934kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<08:20, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<06:02, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:34<07:06, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<06:05, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<04:28, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:36<05:30, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:36<06:03, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<04:47, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<03:27, 2.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:38<27:09, 380kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<20:04, 513kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<14:15, 721kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:40<12:18, 832kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:40<10:50, 945kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:40<08:02, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<05:46, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<07:58, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<06:40, 1.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:55, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<05:42, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<06:10, 1.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<04:50, 2.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<03:30, 2.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<34:14, 293kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<24:59, 401kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<17:43, 565kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<14:39, 680kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<12:22, 806kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<09:07, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<06:28, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<11:43, 844kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<09:15, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<06:43, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<06:57, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<06:59, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:52<05:24, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<03:53, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<25:10, 388kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<18:39, 523kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<13:16, 733kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:55<11:29, 844kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<09:04, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<06:35, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<06:48, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<06:50, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<05:13, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<03:47, 2.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:59<06:34, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<05:37, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:59<04:08, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<03:00, 3.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<3:12:46, 49.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<2:16:55, 69.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<1:36:09, 98.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<1:07:19, 140kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<49:43, 189kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<35:47, 263kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<25:13, 372kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<19:44, 474kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<15:49, 591kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<11:33, 808kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<08:10, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<35:07, 264kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<25:31, 363kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<18:02, 513kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<14:42, 627kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<11:14, 819kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<08:05, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<07:44, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<06:22, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<04:39, 1.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<05:20, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<04:42, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<03:31, 2.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:32, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:07, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<03:07, 2.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<04:13, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<03:55, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<02:58, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<04:07, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<03:39, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<02:48, 3.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<02:04, 4.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<18:04, 487kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:21<13:25, 655kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<09:34, 917kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<06:48, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<21:08, 413kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<15:44, 555kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<11:13, 776kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<09:46, 886kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<08:42, 995kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<06:33, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<04:40, 1.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<08:07, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<06:34, 1.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<04:48, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<05:20, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<04:36, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<03:26, 2.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:22, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<03:57, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<02:57, 2.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<03:58, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<03:39, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<02:47, 3.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<03:51, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<03:32, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<02:40, 3.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<04:25, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<03:28, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<02:32, 3.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<05:09, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<04:29, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<03:21, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<04:09, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:41<04:39, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<03:41, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<02:40, 3.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<25:58, 310kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<19:00, 423kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<13:27, 596kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<11:14, 710kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:45<09:33, 835kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<07:06, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<05:02, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<21:46, 363kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<16:04, 492kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<11:25, 689kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<09:45, 803kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<07:39, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:49<05:32, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<05:39, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:46, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<03:32, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<03:45, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<02:49, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:24, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<02:33, 2.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<03:15, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<02:27, 3.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:28, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:58, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<02:18, 3.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<54:31, 136kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<38:47, 191kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<27:19, 271kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<19:06, 385kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:02<27:17, 270kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<19:51, 370kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<14:01, 522kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<11:26, 637kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<09:34, 761kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<07:04, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<05:00, 1.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<54:58, 131kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<39:12, 184kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<27:32, 261kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<20:50, 343kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<15:19, 467kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<10:51, 656kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<09:13, 768kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<07:11, 984kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<05:10, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<05:13, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<05:09, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<03:58, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<02:51, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<22:52, 304kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<16:44, 415kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<11:51, 583kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<09:50, 700kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<08:21, 822kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<04:23, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<35:43, 191kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<25:42, 265kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<18:04, 375kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<14:07, 477kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<11:19, 596kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<08:16, 814kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:20<05:49, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<20:44, 322kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<15:12, 439kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:22<10:45, 618kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<09:01, 732kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<06:59, 944kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<05:02, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<05:02, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<04:56, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<03:46, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<02:42, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:28<05:58, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:28<04:50, 1.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:32, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<03:57, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<03:25, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<02:33, 2.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:15, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<03:37, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<02:51, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<02:03, 3.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<37:24, 168kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<26:50, 233kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<18:52, 330kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<14:33, 426kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<10:50, 571kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<07:43, 798kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<06:47, 903kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<05:23, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<03:54, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<04:08, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<04:10, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:14, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<02:19, 2.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<19:38, 305kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<14:21, 417kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<10:09, 587kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<08:25, 703kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<07:07, 831kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<05:14, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<03:43, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<05:19, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<04:20, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<03:09, 1.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:31, 1.64MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:41, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<02:50, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<02:03, 2.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<03:43, 1.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<02:43, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<03:12, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<03:26, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<02:42, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<01:57, 2.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<19:02, 293kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<13:54, 401kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<09:50, 564kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<08:06, 679kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<06:50, 805kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<05:04, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<03:37, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<04:23, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<03:38, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:41, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<03:06, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<02:02, 2.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<02:39, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<02:59, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<02:21, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<01:43, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<03:21, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<02:54, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<02:08, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<02:39, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<02:25, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<01:48, 2.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:07<02:25, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<02:14, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<01:41, 2.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:09<02:20, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<02:09, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<01:38, 3.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:11<02:16, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<02:06, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<01:35, 3.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:05, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<01:35, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:03, 2.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<02:10, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<02:32, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:02, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<01:28, 3.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<04:09, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<03:19, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<02:26, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:45, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:21<02:25, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<01:48, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:18, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:23<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<02:06, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:25<02:26, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<01:54, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<01:22, 3.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:34, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<02:58, 1.49MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:27<02:10, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<02:30, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:38, 1.64MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:29<02:04, 2.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<01:29, 2.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<31:33, 136kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<22:30, 190kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<15:46, 269kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<11:54, 353kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<09:12, 457kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<06:36, 634kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<04:40, 891kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<04:33, 909kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:37, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<02:46, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<02:48, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<02:09, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:34, 2.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:19, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:02, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:31, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<02:13, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<01:45, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<01:16, 3.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<12:28, 310kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<09:07, 423kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<06:26, 595kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<05:20, 710kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<04:08, 915kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<02:59, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<02:55, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:26, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:47, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:05, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<01:21, 2.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:01, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:34, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<01:10, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<01:40, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<01:32, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:10, 2.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<01:36, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<01:50, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:27, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<01:02, 3.26MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<18:41, 181kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<13:25, 252kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<09:23, 357kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<07:17, 455kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<05:26, 609kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<03:51, 851kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<03:25, 950kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<02:43, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<01:58, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<02:06, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:09, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:40, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:11, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<08:01, 388kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<05:52, 529kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<04:08, 744kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:04<02:54, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<10:48, 282kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<08:12, 370kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<05:51, 517kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<04:05, 731kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<04:41, 633kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<03:35, 825kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<02:34, 1.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<02:26, 1.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<02:19, 1.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<01:15, 2.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<02:16, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<01:53, 1.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<01:22, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:35, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:14<01:23, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:02, 2.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:20, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:16<01:12, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<00:54, 2.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:14, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:24, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<00:47, 3.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<12:49, 200kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<09:13, 277kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<06:27, 392kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<05:02, 496kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<04:03, 615kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<02:56, 845kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<02:03, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:36, 931kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:04, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<01:30, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:34, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:36, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<01:13, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:52, 2.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<04:09, 551kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<03:08, 728kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<02:13, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<02:03, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:37, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:11, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:18, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:22, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:03, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<00:45, 2.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<01:28, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:14, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:54, 2.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:05, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:58, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:43, 2.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<00:57, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:37<01:04, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<00:36, 3.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<04:56, 380kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<03:38, 514kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<02:33, 720kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<02:10, 831kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<01:40, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<00:50, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<07:13, 241kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<05:25, 321kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<03:51, 447kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<02:38, 635kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<06:03, 276kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<04:23, 379kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<03:04, 534kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<02:28, 648kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<01:53, 846kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<01:20, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<01:16, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<01:03, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:51, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:33, 2.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:53<00:41, 2.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:37, 2.20MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<00:27, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:55<00:37, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:43, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:33, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:25, 3.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:57<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:32, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<00:23, 3.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:59<00:32, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:59<00:30, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:22, 3.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:30, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:28, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:28, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:03<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<00:26, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:03<00:18, 3.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<03:20, 295kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:05<02:25, 404kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<01:40, 570kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<01:20, 685kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:07<01:01, 888kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:43, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:40, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:37, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:28, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:19, 2.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:36, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:29, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:20, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:11<00:14, 2.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<01:33, 458kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<01:13, 577kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:13<00:52, 791kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:13<00:34, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:14<03:47, 170kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<02:41, 237kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<01:49, 336kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:20, 432kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:03, 546kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:44, 753kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:30, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:29, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<00:23, 1.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:16, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:16, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:17, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:13, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<00:08, 2.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:57, 389kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:41, 525kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:27, 736kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:21, 847kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:16, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:09, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:09, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:07, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:04, 2.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:08, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:06, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:03, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:02, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:01, 2.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:00, 3.66MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:03, 481kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:02, 642kB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:01:24,  6.53it/s]  0%|          | 630/400000 [00:00<11:54:11,  9.32it/s]  0%|          | 1266/400000 [00:00<8:19:26, 13.31it/s]  0%|          | 1898/400000 [00:00<5:49:22, 18.99it/s]  1%|          | 2533/400000 [00:00<4:04:29, 27.10it/s]  1%|          | 3243/400000 [00:00<2:51:06, 38.64it/s]  1%|          | 3983/400000 [00:00<1:59:49, 55.08it/s]  1%|          | 4637/400000 [00:00<1:24:02, 78.41it/s]  1%|▏         | 5275/400000 [00:00<59:02, 111.42it/s]   1%|▏         | 5915/400000 [00:01<41:34, 158.00it/s]  2%|▏         | 6567/400000 [00:01<29:21, 223.39it/s]  2%|▏         | 7273/400000 [00:01<20:47, 314.86it/s]  2%|▏         | 8022/400000 [00:01<14:47, 441.83it/s]  2%|▏         | 8766/400000 [00:01<10:35, 615.50it/s]  2%|▏         | 9497/400000 [00:01<07:40, 848.63it/s]  3%|▎         | 10223/400000 [00:01<05:37, 1154.43it/s]  3%|▎         | 10997/400000 [00:01<04:10, 1550.01it/s]  3%|▎         | 11744/400000 [00:01<03:10, 2033.42it/s]  3%|▎         | 12481/400000 [00:01<02:29, 2596.46it/s]  3%|▎         | 13253/400000 [00:02<01:59, 3241.47it/s]  4%|▎         | 14003/400000 [00:02<01:38, 3906.55it/s]  4%|▎         | 14773/400000 [00:02<01:24, 4583.51it/s]  4%|▍         | 15527/400000 [00:02<01:14, 5180.30it/s]  4%|▍         | 16279/400000 [00:02<01:07, 5679.29it/s]  4%|▍         | 17025/400000 [00:02<01:03, 6050.43it/s]  4%|▍         | 17762/400000 [00:02<01:01, 6174.14it/s]  5%|▍         | 18472/400000 [00:02<01:01, 6212.43it/s]  5%|▍         | 19158/400000 [00:02<01:01, 6234.56it/s]  5%|▍         | 19887/400000 [00:02<00:58, 6516.07it/s]  5%|▌         | 20638/400000 [00:03<00:55, 6785.14it/s]  5%|▌         | 21389/400000 [00:03<00:54, 6978.53it/s]  6%|▌         | 22162/400000 [00:03<00:52, 7185.55it/s]  6%|▌         | 22937/400000 [00:03<00:51, 7344.18it/s]  6%|▌         | 23697/400000 [00:03<00:50, 7418.85it/s]  6%|▌         | 24461/400000 [00:03<00:50, 7482.90it/s]  6%|▋         | 25216/400000 [00:03<00:50, 7414.20it/s]  6%|▋         | 25962/400000 [00:03<00:52, 7111.14it/s]  7%|▋         | 26679/400000 [00:03<00:53, 6923.69it/s]  7%|▋         | 27377/400000 [00:04<00:55, 6753.56it/s]  7%|▋         | 28057/400000 [00:04<00:55, 6733.75it/s]  7%|▋         | 28799/400000 [00:04<00:53, 6924.56it/s]  7%|▋         | 29528/400000 [00:04<00:52, 7026.26it/s]  8%|▊         | 30299/400000 [00:04<00:51, 7215.69it/s]  8%|▊         | 31068/400000 [00:04<00:50, 7349.71it/s]  8%|▊         | 31817/400000 [00:04<00:49, 7390.33it/s]  8%|▊         | 32581/400000 [00:04<00:49, 7461.53it/s]  8%|▊         | 33329/400000 [00:04<00:50, 7325.27it/s]  9%|▊         | 34086/400000 [00:04<00:49, 7396.76it/s]  9%|▊         | 34847/400000 [00:05<00:48, 7458.15it/s]  9%|▉         | 35602/400000 [00:05<00:48, 7483.37it/s]  9%|▉         | 36358/400000 [00:05<00:48, 7504.89it/s]  9%|▉         | 37116/400000 [00:05<00:48, 7526.90it/s]  9%|▉         | 37870/400000 [00:05<00:49, 7357.71it/s] 10%|▉         | 38607/400000 [00:05<00:49, 7242.50it/s] 10%|▉         | 39333/400000 [00:05<00:52, 6884.23it/s] 10%|█         | 40066/400000 [00:05<00:51, 7012.03it/s] 10%|█         | 40771/400000 [00:05<00:51, 6990.01it/s] 10%|█         | 41528/400000 [00:05<00:50, 7153.90it/s] 11%|█         | 42253/400000 [00:06<00:49, 7181.92it/s] 11%|█         | 43011/400000 [00:06<00:48, 7295.92it/s] 11%|█         | 43793/400000 [00:06<00:47, 7444.57it/s] 11%|█         | 44569/400000 [00:06<00:47, 7534.41it/s] 11%|█▏        | 45336/400000 [00:06<00:46, 7572.27it/s] 12%|█▏        | 46095/400000 [00:06<00:46, 7565.48it/s] 12%|█▏        | 46861/400000 [00:06<00:46, 7590.16it/s] 12%|█▏        | 47634/400000 [00:06<00:46, 7629.21it/s] 12%|█▏        | 48398/400000 [00:06<00:46, 7620.96it/s] 12%|█▏        | 49161/400000 [00:06<00:46, 7610.54it/s] 12%|█▏        | 49926/400000 [00:07<00:45, 7619.76it/s] 13%|█▎        | 50689/400000 [00:07<00:46, 7527.91it/s] 13%|█▎        | 51443/400000 [00:07<00:46, 7471.24it/s] 13%|█▎        | 52203/400000 [00:07<00:46, 7508.91it/s] 13%|█▎        | 52955/400000 [00:07<00:48, 7192.25it/s] 13%|█▎        | 53678/400000 [00:07<00:50, 6921.73it/s] 14%|█▎        | 54375/400000 [00:07<00:50, 6912.00it/s] 14%|█▍        | 55118/400000 [00:07<00:48, 7057.47it/s] 14%|█▍        | 55841/400000 [00:07<00:48, 7107.79it/s] 14%|█▍        | 56554/400000 [00:08<00:48, 7072.33it/s] 14%|█▍        | 57263/400000 [00:08<00:49, 6901.42it/s] 15%|█▍        | 58007/400000 [00:08<00:48, 7053.00it/s] 15%|█▍        | 58772/400000 [00:08<00:47, 7221.73it/s] 15%|█▍        | 59505/400000 [00:08<00:46, 7251.73it/s] 15%|█▌        | 60267/400000 [00:08<00:46, 7355.44it/s] 15%|█▌        | 61035/400000 [00:08<00:45, 7447.39it/s] 15%|█▌        | 61810/400000 [00:08<00:44, 7534.27it/s] 16%|█▌        | 62571/400000 [00:08<00:44, 7554.92it/s] 16%|█▌        | 63328/400000 [00:08<00:45, 7451.34it/s] 16%|█▌        | 64075/400000 [00:09<00:45, 7400.29it/s] 16%|█▌        | 64827/400000 [00:09<00:45, 7435.34it/s] 16%|█▋        | 65573/400000 [00:09<00:44, 7441.97it/s] 17%|█▋        | 66333/400000 [00:09<00:44, 7486.89it/s] 17%|█▋        | 67083/400000 [00:09<00:45, 7351.58it/s] 17%|█▋        | 67819/400000 [00:09<00:45, 7338.48it/s] 17%|█▋        | 68596/400000 [00:09<00:44, 7460.88it/s] 17%|█▋        | 69370/400000 [00:09<00:43, 7541.29it/s] 18%|█▊        | 70136/400000 [00:09<00:43, 7574.63it/s] 18%|█▊        | 70895/400000 [00:09<00:43, 7551.55it/s] 18%|█▊        | 71651/400000 [00:10<00:45, 7157.67it/s] 18%|█▊        | 72372/400000 [00:10<00:45, 7157.34it/s] 18%|█▊        | 73142/400000 [00:10<00:44, 7309.56it/s] 18%|█▊        | 73901/400000 [00:10<00:44, 7389.20it/s] 19%|█▊        | 74643/400000 [00:10<00:44, 7291.20it/s] 19%|█▉        | 75382/400000 [00:10<00:44, 7319.45it/s] 19%|█▉        | 76150/400000 [00:10<00:43, 7421.56it/s] 19%|█▉        | 76894/400000 [00:10<00:43, 7350.61it/s] 19%|█▉        | 77643/400000 [00:10<00:43, 7389.97it/s] 20%|█▉        | 78383/400000 [00:10<00:43, 7375.11it/s] 20%|█▉        | 79133/400000 [00:11<00:43, 7405.99it/s] 20%|█▉        | 79878/400000 [00:11<00:43, 7418.71it/s] 20%|██        | 80621/400000 [00:11<00:43, 7412.80it/s] 20%|██        | 81383/400000 [00:11<00:42, 7472.54it/s] 21%|██        | 82131/400000 [00:11<00:43, 7387.62it/s] 21%|██        | 82884/400000 [00:11<00:42, 7429.74it/s] 21%|██        | 83635/400000 [00:11<00:42, 7452.63it/s] 21%|██        | 84400/400000 [00:11<00:42, 7508.46it/s] 21%|██▏       | 85152/400000 [00:11<00:42, 7449.40it/s] 21%|██▏       | 85898/400000 [00:11<00:42, 7437.00it/s] 22%|██▏       | 86642/400000 [00:12<00:42, 7423.42it/s] 22%|██▏       | 87385/400000 [00:12<00:43, 7213.98it/s] 22%|██▏       | 88112/400000 [00:12<00:43, 7223.82it/s] 22%|██▏       | 88873/400000 [00:12<00:42, 7334.62it/s] 22%|██▏       | 89609/400000 [00:12<00:42, 7340.97it/s] 23%|██▎       | 90366/400000 [00:12<00:41, 7406.10it/s] 23%|██▎       | 91133/400000 [00:12<00:41, 7481.53it/s] 23%|██▎       | 91882/400000 [00:12<00:41, 7432.52it/s] 23%|██▎       | 92637/400000 [00:12<00:41, 7466.78it/s] 23%|██▎       | 93385/400000 [00:13<00:42, 7233.60it/s] 24%|██▎       | 94111/400000 [00:13<00:44, 6893.42it/s] 24%|██▎       | 94806/400000 [00:13<00:45, 6733.36it/s] 24%|██▍       | 95484/400000 [00:13<00:45, 6656.60it/s] 24%|██▍       | 96153/400000 [00:13<00:46, 6595.18it/s] 24%|██▍       | 96879/400000 [00:13<00:44, 6780.36it/s] 24%|██▍       | 97640/400000 [00:13<00:43, 7009.15it/s] 25%|██▍       | 98365/400000 [00:13<00:42, 7077.49it/s] 25%|██▍       | 99076/400000 [00:13<00:45, 6659.79it/s] 25%|██▍       | 99750/400000 [00:13<00:46, 6519.94it/s] 25%|██▌       | 100460/400000 [00:14<00:44, 6681.08it/s] 25%|██▌       | 101199/400000 [00:14<00:43, 6878.48it/s] 25%|██▌       | 101976/400000 [00:14<00:41, 7123.59it/s] 26%|██▌       | 102723/400000 [00:14<00:41, 7220.84it/s] 26%|██▌       | 103450/400000 [00:14<00:41, 7132.56it/s] 26%|██▌       | 104170/400000 [00:14<00:41, 7151.71it/s] 26%|██▌       | 104897/400000 [00:14<00:41, 7186.06it/s] 26%|██▋       | 105654/400000 [00:14<00:40, 7296.34it/s] 27%|██▋       | 106402/400000 [00:14<00:39, 7348.94it/s] 27%|██▋       | 107139/400000 [00:14<00:40, 7265.76it/s] 27%|██▋       | 107895/400000 [00:15<00:39, 7351.14it/s] 27%|██▋       | 108658/400000 [00:15<00:39, 7432.49it/s] 27%|██▋       | 109428/400000 [00:15<00:38, 7510.70it/s] 28%|██▊       | 110180/400000 [00:15<00:38, 7470.72it/s] 28%|██▊       | 110929/400000 [00:15<00:38, 7475.95it/s] 28%|██▊       | 111678/400000 [00:15<00:39, 7386.17it/s] 28%|██▊       | 112443/400000 [00:15<00:38, 7461.93it/s] 28%|██▊       | 113193/400000 [00:15<00:38, 7471.53it/s] 28%|██▊       | 113957/400000 [00:15<00:38, 7516.56it/s] 29%|██▊       | 114710/400000 [00:15<00:38, 7506.67it/s] 29%|██▉       | 115461/400000 [00:16<00:39, 7278.84it/s] 29%|██▉       | 116191/400000 [00:16<00:39, 7269.57it/s] 29%|██▉       | 116926/400000 [00:16<00:38, 7291.63it/s] 29%|██▉       | 117669/400000 [00:16<00:38, 7330.33it/s] 30%|██▉       | 118429/400000 [00:16<00:38, 7407.43it/s] 30%|██▉       | 119171/400000 [00:16<00:39, 7191.54it/s] 30%|██▉       | 119892/400000 [00:16<00:40, 6916.11it/s] 30%|███       | 120588/400000 [00:16<00:41, 6787.16it/s] 30%|███       | 121270/400000 [00:16<00:41, 6681.10it/s] 30%|███       | 121974/400000 [00:17<00:40, 6782.35it/s] 31%|███       | 122714/400000 [00:17<00:39, 6955.97it/s] 31%|███       | 123442/400000 [00:17<00:39, 7048.07it/s] 31%|███       | 124149/400000 [00:17<00:39, 6997.77it/s] 31%|███       | 124851/400000 [00:17<00:39, 6899.18it/s] 31%|███▏      | 125543/400000 [00:17<00:40, 6755.19it/s] 32%|███▏      | 126221/400000 [00:17<00:41, 6628.72it/s] 32%|███▏      | 126897/400000 [00:17<00:40, 6667.50it/s] 32%|███▏      | 127658/400000 [00:17<00:39, 6922.65it/s] 32%|███▏      | 128408/400000 [00:17<00:38, 7083.05it/s] 32%|███▏      | 129120/400000 [00:18<00:38, 7043.76it/s] 32%|███▏      | 129843/400000 [00:18<00:38, 7096.84it/s] 33%|███▎      | 130583/400000 [00:18<00:37, 7185.07it/s] 33%|███▎      | 131323/400000 [00:18<00:37, 7246.13it/s] 33%|███▎      | 132058/400000 [00:18<00:36, 7274.69it/s] 33%|███▎      | 132800/400000 [00:18<00:36, 7317.46it/s] 33%|███▎      | 133547/400000 [00:18<00:36, 7361.35it/s] 34%|███▎      | 134307/400000 [00:18<00:35, 7429.58it/s] 34%|███▍      | 135052/400000 [00:18<00:35, 7433.63it/s] 34%|███▍      | 135796/400000 [00:18<00:35, 7414.22it/s] 34%|███▍      | 136538/400000 [00:19<00:37, 7090.99it/s] 34%|███▍      | 137251/400000 [00:19<00:38, 6754.92it/s] 34%|███▍      | 137933/400000 [00:19<00:39, 6641.31it/s] 35%|███▍      | 138621/400000 [00:19<00:38, 6708.88it/s] 35%|███▍      | 139355/400000 [00:19<00:37, 6884.68it/s] 35%|███▌      | 140116/400000 [00:19<00:36, 7086.08it/s] 35%|███▌      | 140849/400000 [00:19<00:36, 7154.62it/s] 35%|███▌      | 141587/400000 [00:19<00:35, 7219.59it/s] 36%|███▌      | 142340/400000 [00:19<00:35, 7309.15it/s] 36%|███▌      | 143086/400000 [00:20<00:34, 7351.32it/s] 36%|███▌      | 143835/400000 [00:20<00:34, 7391.80it/s] 36%|███▌      | 144576/400000 [00:20<00:34, 7335.22it/s] 36%|███▋      | 145341/400000 [00:20<00:34, 7426.75it/s] 37%|███▋      | 146085/400000 [00:20<00:34, 7415.06it/s] 37%|███▋      | 146828/400000 [00:20<00:34, 7410.89it/s] 37%|███▋      | 147570/400000 [00:20<00:34, 7286.83it/s] 37%|███▋      | 148300/400000 [00:20<00:34, 7273.90it/s] 37%|███▋      | 149028/400000 [00:20<00:34, 7272.07it/s] 37%|███▋      | 149763/400000 [00:20<00:34, 7293.45it/s] 38%|███▊      | 150527/400000 [00:21<00:33, 7392.26it/s] 38%|███▊      | 151303/400000 [00:21<00:33, 7497.95it/s] 38%|███▊      | 152067/400000 [00:21<00:32, 7537.13it/s] 38%|███▊      | 152822/400000 [00:21<00:32, 7519.89it/s] 38%|███▊      | 153588/400000 [00:21<00:32, 7560.22it/s] 39%|███▊      | 154345/400000 [00:21<00:33, 7316.63it/s] 39%|███▉      | 155079/400000 [00:21<00:34, 7154.01it/s] 39%|███▉      | 155835/400000 [00:21<00:33, 7269.55it/s] 39%|███▉      | 156596/400000 [00:21<00:33, 7367.79it/s] 39%|███▉      | 157351/400000 [00:21<00:32, 7420.87it/s] 40%|███▉      | 158095/400000 [00:22<00:32, 7421.91it/s] 40%|███▉      | 158839/400000 [00:22<00:32, 7362.39it/s] 40%|███▉      | 159576/400000 [00:22<00:34, 6971.97it/s] 40%|████      | 160279/400000 [00:22<00:35, 6811.00it/s] 40%|████      | 160965/400000 [00:22<00:35, 6697.88it/s] 40%|████      | 161639/400000 [00:22<00:35, 6673.58it/s] 41%|████      | 162337/400000 [00:22<00:35, 6762.63it/s] 41%|████      | 163028/400000 [00:22<00:34, 6805.28it/s] 41%|████      | 163792/400000 [00:22<00:33, 7035.33it/s] 41%|████      | 164565/400000 [00:22<00:32, 7229.41it/s] 41%|████▏     | 165334/400000 [00:23<00:31, 7361.57it/s] 42%|████▏     | 166086/400000 [00:23<00:31, 7407.96it/s] 42%|████▏     | 166829/400000 [00:23<00:31, 7305.47it/s] 42%|████▏     | 167562/400000 [00:23<00:31, 7292.71it/s] 42%|████▏     | 168293/400000 [00:23<00:32, 7143.57it/s] 42%|████▏     | 169009/400000 [00:23<00:33, 6975.54it/s] 42%|████▏     | 169709/400000 [00:23<00:33, 6827.98it/s] 43%|████▎     | 170394/400000 [00:23<00:34, 6598.80it/s] 43%|████▎     | 171083/400000 [00:23<00:34, 6681.42it/s] 43%|████▎     | 171823/400000 [00:24<00:33, 6880.64it/s] 43%|████▎     | 172592/400000 [00:24<00:32, 7103.15it/s] 43%|████▎     | 173369/400000 [00:24<00:31, 7289.98it/s] 44%|████▎     | 174103/400000 [00:24<00:31, 7117.53it/s] 44%|████▎     | 174819/400000 [00:24<00:32, 6952.16it/s] 44%|████▍     | 175518/400000 [00:24<00:33, 6728.47it/s] 44%|████▍     | 176195/400000 [00:24<00:33, 6665.82it/s] 44%|████▍     | 176911/400000 [00:24<00:32, 6805.23it/s] 44%|████▍     | 177670/400000 [00:24<00:31, 7021.42it/s] 45%|████▍     | 178448/400000 [00:24<00:30, 7232.09it/s] 45%|████▍     | 179176/400000 [00:25<00:31, 6992.57it/s] 45%|████▍     | 179880/400000 [00:25<00:32, 6757.84it/s] 45%|████▌     | 180615/400000 [00:25<00:31, 6924.39it/s] 45%|████▌     | 181380/400000 [00:25<00:30, 7126.90it/s] 46%|████▌     | 182129/400000 [00:25<00:30, 7229.33it/s] 46%|████▌     | 182883/400000 [00:25<00:29, 7318.52it/s] 46%|████▌     | 183620/400000 [00:25<00:29, 7332.41it/s] 46%|████▌     | 184356/400000 [00:25<00:29, 7241.74it/s] 46%|████▋     | 185082/400000 [00:25<00:30, 7019.65it/s] 46%|████▋     | 185787/400000 [00:25<00:31, 6818.53it/s] 47%|████▋     | 186472/400000 [00:26<00:31, 6775.74it/s] 47%|████▋     | 187224/400000 [00:26<00:30, 6982.42it/s] 47%|████▋     | 187926/400000 [00:26<00:30, 6976.10it/s] 47%|████▋     | 188664/400000 [00:26<00:29, 7090.61it/s] 47%|████▋     | 189423/400000 [00:26<00:29, 7233.21it/s] 48%|████▊     | 190206/400000 [00:26<00:28, 7400.28it/s] 48%|████▊     | 190949/400000 [00:26<00:28, 7404.31it/s] 48%|████▊     | 191700/400000 [00:26<00:28, 7434.88it/s] 48%|████▊     | 192476/400000 [00:26<00:27, 7528.22it/s] 48%|████▊     | 193230/400000 [00:27<00:28, 7368.33it/s] 48%|████▊     | 193969/400000 [00:27<00:28, 7322.35it/s] 49%|████▊     | 194718/400000 [00:27<00:27, 7370.68it/s] 49%|████▉     | 195456/400000 [00:27<00:27, 7322.27it/s] 49%|████▉     | 196189/400000 [00:27<00:28, 7066.51it/s] 49%|████▉     | 196899/400000 [00:27<00:29, 6890.28it/s] 49%|████▉     | 197591/400000 [00:27<00:29, 6881.60it/s] 50%|████▉     | 198282/400000 [00:27<00:29, 6809.03it/s] 50%|████▉     | 198965/400000 [00:27<00:30, 6694.14it/s] 50%|████▉     | 199636/400000 [00:27<00:30, 6632.64it/s] 50%|█████     | 200301/400000 [00:28<00:30, 6510.60it/s] 50%|█████     | 200960/400000 [00:28<00:30, 6533.48it/s] 50%|█████     | 201680/400000 [00:28<00:29, 6719.55it/s] 51%|█████     | 202413/400000 [00:28<00:28, 6889.16it/s] 51%|█████     | 203152/400000 [00:28<00:28, 7029.88it/s] 51%|█████     | 203890/400000 [00:28<00:27, 7129.91it/s] 51%|█████     | 204635/400000 [00:28<00:27, 7222.38it/s] 51%|█████▏    | 205359/400000 [00:28<00:27, 7107.97it/s] 52%|█████▏    | 206072/400000 [00:28<00:28, 6840.74it/s] 52%|█████▏    | 206760/400000 [00:28<00:28, 6688.01it/s] 52%|█████▏    | 207463/400000 [00:29<00:28, 6783.87it/s] 52%|█████▏    | 208228/400000 [00:29<00:27, 7021.23it/s] 52%|█████▏    | 208988/400000 [00:29<00:26, 7183.97it/s] 52%|█████▏    | 209724/400000 [00:29<00:26, 7233.72it/s] 53%|█████▎    | 210476/400000 [00:29<00:25, 7317.27it/s] 53%|█████▎    | 211218/400000 [00:29<00:25, 7347.02it/s] 53%|█████▎    | 211983/400000 [00:29<00:25, 7433.38it/s] 53%|█████▎    | 212742/400000 [00:29<00:25, 7478.06it/s] 53%|█████▎    | 213491/400000 [00:29<00:25, 7433.80it/s] 54%|█████▎    | 214236/400000 [00:29<00:25, 7258.36it/s] 54%|█████▎    | 214974/400000 [00:30<00:25, 7292.72it/s] 54%|█████▍    | 215751/400000 [00:30<00:24, 7429.61it/s] 54%|█████▍    | 216522/400000 [00:30<00:24, 7508.16it/s] 54%|█████▍    | 217274/400000 [00:30<00:24, 7391.85it/s] 55%|█████▍    | 218019/400000 [00:30<00:24, 7407.13it/s] 55%|█████▍    | 218768/400000 [00:30<00:24, 7430.08it/s] 55%|█████▍    | 219512/400000 [00:30<00:24, 7427.24it/s] 55%|█████▌    | 220267/400000 [00:30<00:24, 7460.72it/s] 55%|█████▌    | 221018/400000 [00:30<00:23, 7474.41it/s] 55%|█████▌    | 221776/400000 [00:31<00:23, 7504.89it/s] 56%|█████▌    | 222537/400000 [00:31<00:23, 7535.16it/s] 56%|█████▌    | 223291/400000 [00:31<00:23, 7499.61it/s] 56%|█████▌    | 224059/400000 [00:31<00:23, 7551.80it/s] 56%|█████▌    | 224815/400000 [00:31<00:24, 7278.50it/s] 56%|█████▋    | 225546/400000 [00:31<00:24, 7029.54it/s] 57%|█████▋    | 226253/400000 [00:31<00:25, 6853.51it/s] 57%|█████▋    | 226942/400000 [00:31<00:25, 6716.84it/s] 57%|█████▋    | 227686/400000 [00:31<00:24, 6917.07it/s] 57%|█████▋    | 228390/400000 [00:31<00:24, 6952.35it/s] 57%|█████▋    | 229088/400000 [00:32<00:25, 6766.50it/s] 57%|█████▋    | 229839/400000 [00:32<00:24, 6973.43it/s] 58%|█████▊    | 230600/400000 [00:32<00:23, 7151.82it/s] 58%|█████▊    | 231319/400000 [00:32<00:24, 7001.10it/s] 58%|█████▊    | 232023/400000 [00:32<00:24, 6967.11it/s] 58%|█████▊    | 232751/400000 [00:32<00:23, 7057.56it/s] 58%|█████▊    | 233459/400000 [00:32<00:23, 6969.04it/s] 59%|█████▊    | 234195/400000 [00:32<00:23, 7079.33it/s] 59%|█████▊    | 234955/400000 [00:32<00:22, 7227.64it/s] 59%|█████▉    | 235680/400000 [00:32<00:23, 7117.65it/s] 59%|█████▉    | 236442/400000 [00:33<00:22, 7260.33it/s] 59%|█████▉    | 237189/400000 [00:33<00:22, 7321.36it/s] 59%|█████▉    | 237939/400000 [00:33<00:21, 7371.79it/s] 60%|█████▉    | 238678/400000 [00:33<00:21, 7367.36it/s] 60%|█████▉    | 239439/400000 [00:33<00:21, 7437.49it/s] 60%|██████    | 240184/400000 [00:33<00:21, 7279.67it/s] 60%|██████    | 240918/400000 [00:33<00:21, 7297.19it/s] 60%|██████    | 241678/400000 [00:33<00:21, 7383.22it/s] 61%|██████    | 242434/400000 [00:33<00:21, 7435.05it/s] 61%|██████    | 243179/400000 [00:33<00:21, 7202.82it/s] 61%|██████    | 243902/400000 [00:34<00:22, 6951.59it/s] 61%|██████    | 244642/400000 [00:34<00:21, 7079.06it/s] 61%|██████▏   | 245430/400000 [00:34<00:21, 7299.75it/s] 62%|██████▏   | 246202/400000 [00:34<00:20, 7420.20it/s] 62%|██████▏   | 246973/400000 [00:34<00:20, 7503.62it/s] 62%|██████▏   | 247752/400000 [00:34<00:20, 7587.07it/s] 62%|██████▏   | 248513/400000 [00:34<00:20, 7538.09it/s] 62%|██████▏   | 249300/400000 [00:34<00:19, 7633.43it/s] 63%|██████▎   | 250065/400000 [00:34<00:19, 7617.96it/s] 63%|██████▎   | 250833/400000 [00:35<00:19, 7635.88it/s] 63%|██████▎   | 251598/400000 [00:35<00:19, 7456.40it/s] 63%|██████▎   | 252346/400000 [00:35<00:20, 7138.27it/s] 63%|██████▎   | 253064/400000 [00:35<00:21, 6853.10it/s] 63%|██████▎   | 253755/400000 [00:35<00:21, 6791.00it/s] 64%|██████▎   | 254443/400000 [00:35<00:21, 6815.97it/s] 64%|██████▍   | 255189/400000 [00:35<00:20, 6996.00it/s] 64%|██████▍   | 255958/400000 [00:35<00:20, 7187.75it/s] 64%|██████▍   | 256716/400000 [00:35<00:19, 7298.74it/s] 64%|██████▍   | 257449/400000 [00:35<00:19, 7213.50it/s] 65%|██████▍   | 258194/400000 [00:36<00:19, 7280.10it/s] 65%|██████▍   | 258966/400000 [00:36<00:19, 7404.46it/s] 65%|██████▍   | 259724/400000 [00:36<00:18, 7454.79it/s] 65%|██████▌   | 260471/400000 [00:36<00:18, 7458.77it/s] 65%|██████▌   | 261218/400000 [00:36<00:18, 7389.27it/s] 65%|██████▌   | 261985/400000 [00:36<00:18, 7470.88it/s] 66%|██████▌   | 262774/400000 [00:36<00:18, 7590.34it/s] 66%|██████▌   | 263542/400000 [00:36<00:17, 7615.70it/s] 66%|██████▌   | 264305/400000 [00:36<00:18, 7392.80it/s] 66%|██████▋   | 265047/400000 [00:36<00:18, 7245.55it/s] 66%|██████▋   | 265774/400000 [00:37<00:19, 6990.74it/s] 67%|██████▋   | 266505/400000 [00:37<00:18, 7082.66it/s] 67%|██████▋   | 267272/400000 [00:37<00:18, 7249.08it/s] 67%|██████▋   | 268009/400000 [00:37<00:18, 7284.21it/s] 67%|██████▋   | 268740/400000 [00:37<00:18, 7289.35it/s] 67%|██████▋   | 269492/400000 [00:37<00:17, 7354.82it/s] 68%|██████▊   | 270235/400000 [00:37<00:17, 7374.97it/s] 68%|██████▊   | 271033/400000 [00:37<00:17, 7545.54it/s] 68%|██████▊   | 271790/400000 [00:37<00:17, 7381.85it/s] 68%|██████▊   | 272533/400000 [00:37<00:17, 7395.21it/s] 68%|██████▊   | 273283/400000 [00:38<00:17, 7425.09it/s] 69%|██████▊   | 274035/400000 [00:38<00:16, 7452.10it/s] 69%|██████▊   | 274781/400000 [00:38<00:16, 7376.91it/s] 69%|██████▉   | 275520/400000 [00:38<00:17, 7113.12it/s] 69%|██████▉   | 276234/400000 [00:38<00:17, 7084.34it/s] 69%|██████▉   | 277009/400000 [00:38<00:16, 7269.69it/s] 69%|██████▉   | 277771/400000 [00:38<00:16, 7370.93it/s] 70%|██████▉   | 278511/400000 [00:38<00:17, 7038.28it/s] 70%|██████▉   | 279220/400000 [00:38<00:17, 6868.87it/s] 70%|██████▉   | 279912/400000 [00:39<00:17, 6695.63it/s] 70%|███████   | 280586/400000 [00:39<00:17, 6678.63it/s] 70%|███████   | 281351/400000 [00:39<00:17, 6941.82it/s] 71%|███████   | 282108/400000 [00:39<00:16, 7118.35it/s] 71%|███████   | 282836/400000 [00:39<00:16, 7165.41it/s] 71%|███████   | 283556/400000 [00:39<00:16, 7035.71it/s] 71%|███████   | 284263/400000 [00:39<00:16, 6875.92it/s] 71%|███████   | 284954/400000 [00:39<00:17, 6736.52it/s] 71%|███████▏  | 285684/400000 [00:39<00:16, 6895.89it/s] 72%|███████▏  | 286441/400000 [00:39<00:16, 7083.08it/s] 72%|███████▏  | 287153/400000 [00:40<00:15, 7083.56it/s] 72%|███████▏  | 287864/400000 [00:40<00:15, 7021.49it/s] 72%|███████▏  | 288621/400000 [00:40<00:15, 7175.56it/s] 72%|███████▏  | 289349/400000 [00:40<00:15, 7205.72it/s] 73%|███████▎  | 290109/400000 [00:40<00:15, 7317.97it/s] 73%|███████▎  | 290866/400000 [00:40<00:14, 7391.78it/s] 73%|███████▎  | 291607/400000 [00:40<00:14, 7395.81it/s] 73%|███████▎  | 292378/400000 [00:40<00:14, 7485.68it/s] 73%|███████▎  | 293152/400000 [00:40<00:14, 7558.01it/s] 73%|███████▎  | 293909/400000 [00:40<00:14, 7536.51it/s] 74%|███████▎  | 294664/400000 [00:41<00:14, 7509.38it/s] 74%|███████▍  | 295416/400000 [00:41<00:14, 7240.93it/s] 74%|███████▍  | 296169/400000 [00:41<00:14, 7323.46it/s] 74%|███████▍  | 296956/400000 [00:41<00:13, 7478.37it/s] 74%|███████▍  | 297713/400000 [00:41<00:13, 7505.21it/s] 75%|███████▍  | 298466/400000 [00:41<00:13, 7509.44it/s] 75%|███████▍  | 299218/400000 [00:41<00:13, 7294.48it/s] 75%|███████▍  | 299950/400000 [00:41<00:13, 7241.89it/s] 75%|███████▌  | 300718/400000 [00:41<00:13, 7365.68it/s] 75%|███████▌  | 301482/400000 [00:42<00:13, 7443.47it/s] 76%|███████▌  | 302255/400000 [00:42<00:12, 7523.38it/s] 76%|███████▌  | 303049/400000 [00:42<00:12, 7642.00it/s] 76%|███████▌  | 303815/400000 [00:42<00:12, 7626.92it/s] 76%|███████▌  | 304588/400000 [00:42<00:12, 7656.89it/s] 76%|███████▋  | 305382/400000 [00:42<00:12, 7738.76it/s] 77%|███████▋  | 306157/400000 [00:42<00:12, 7509.11it/s] 77%|███████▋  | 306912/400000 [00:42<00:12, 7521.20it/s] 77%|███████▋  | 307695/400000 [00:42<00:12, 7610.62it/s] 77%|███████▋  | 308458/400000 [00:42<00:12, 7610.52it/s] 77%|███████▋  | 309220/400000 [00:43<00:12, 7559.17it/s] 77%|███████▋  | 309977/400000 [00:43<00:11, 7559.11it/s] 78%|███████▊  | 310734/400000 [00:43<00:12, 7375.97it/s] 78%|███████▊  | 311473/400000 [00:43<00:12, 7298.55it/s] 78%|███████▊  | 312204/400000 [00:43<00:12, 7277.27it/s] 78%|███████▊  | 312933/400000 [00:43<00:12, 7209.03it/s] 78%|███████▊  | 313665/400000 [00:43<00:11, 7240.41it/s] 79%|███████▊  | 314411/400000 [00:43<00:11, 7304.43it/s] 79%|███████▉  | 315161/400000 [00:43<00:11, 7361.49it/s] 79%|███████▉  | 315898/400000 [00:43<00:11, 7337.18it/s] 79%|███████▉  | 316633/400000 [00:44<00:11, 7307.00it/s] 79%|███████▉  | 317364/400000 [00:44<00:11, 7023.74it/s] 80%|███████▉  | 318069/400000 [00:44<00:11, 6916.72it/s] 80%|███████▉  | 318815/400000 [00:44<00:11, 7071.13it/s] 80%|███████▉  | 319551/400000 [00:44<00:11, 7153.66it/s] 80%|████████  | 320300/400000 [00:44<00:10, 7249.73it/s] 80%|████████  | 321027/400000 [00:44<00:10, 7189.70it/s] 80%|████████  | 321783/400000 [00:44<00:10, 7294.48it/s] 81%|████████  | 322532/400000 [00:44<00:10, 7350.10it/s] 81%|████████  | 323268/400000 [00:44<00:10, 7075.39it/s] 81%|████████  | 324004/400000 [00:45<00:10, 7153.79it/s] 81%|████████  | 324736/400000 [00:45<00:10, 7201.72it/s] 81%|████████▏ | 325458/400000 [00:45<00:10, 7143.35it/s] 82%|████████▏ | 326186/400000 [00:45<00:10, 7182.70it/s] 82%|████████▏ | 326932/400000 [00:45<00:10, 7262.81it/s] 82%|████████▏ | 327682/400000 [00:45<00:09, 7332.17it/s] 82%|████████▏ | 328416/400000 [00:45<00:09, 7203.82it/s] 82%|████████▏ | 329167/400000 [00:45<00:09, 7289.83it/s] 82%|████████▏ | 329912/400000 [00:45<00:09, 7336.14it/s] 83%|████████▎ | 330656/400000 [00:45<00:09, 7365.70it/s] 83%|████████▎ | 331394/400000 [00:46<00:09, 7328.24it/s] 83%|████████▎ | 332129/400000 [00:46<00:09, 7332.14it/s] 83%|████████▎ | 332863/400000 [00:46<00:09, 7326.36it/s] 83%|████████▎ | 333596/400000 [00:46<00:09, 7322.49it/s] 84%|████████▎ | 334329/400000 [00:46<00:08, 7310.50it/s] 84%|████████▍ | 335061/400000 [00:46<00:08, 7267.42it/s] 84%|████████▍ | 335788/400000 [00:46<00:08, 7220.67it/s] 84%|████████▍ | 336535/400000 [00:46<00:08, 7293.70it/s] 84%|████████▍ | 337288/400000 [00:46<00:08, 7360.82it/s] 85%|████████▍ | 338025/400000 [00:46<00:08, 7257.54it/s] 85%|████████▍ | 338775/400000 [00:47<00:08, 7328.35it/s] 85%|████████▍ | 339511/400000 [00:47<00:08, 7337.66it/s] 85%|████████▌ | 340253/400000 [00:47<00:08, 7361.31it/s] 85%|████████▌ | 340990/400000 [00:47<00:08, 7361.35it/s] 85%|████████▌ | 341727/400000 [00:47<00:07, 7362.61it/s] 86%|████████▌ | 342464/400000 [00:47<00:07, 7272.19it/s] 86%|████████▌ | 343192/400000 [00:47<00:07, 7209.98it/s] 86%|████████▌ | 343914/400000 [00:47<00:07, 7175.51it/s] 86%|████████▌ | 344632/400000 [00:47<00:08, 6817.66it/s] 86%|████████▋ | 345318/400000 [00:48<00:08, 6544.50it/s] 86%|████████▋ | 345978/400000 [00:48<00:08, 6484.18it/s] 87%|████████▋ | 346631/400000 [00:48<00:08, 6296.49it/s] 87%|████████▋ | 347361/400000 [00:48<00:08, 6566.47it/s] 87%|████████▋ | 348113/400000 [00:48<00:07, 6824.81it/s] 87%|████████▋ | 348840/400000 [00:48<00:07, 6951.61it/s] 87%|████████▋ | 349541/400000 [00:48<00:07, 6650.40it/s] 88%|████████▊ | 350226/400000 [00:48<00:07, 6706.37it/s] 88%|████████▊ | 350945/400000 [00:48<00:07, 6843.77it/s] 88%|████████▊ | 351695/400000 [00:48<00:06, 7026.15it/s] 88%|████████▊ | 352437/400000 [00:49<00:06, 7137.30it/s] 88%|████████▊ | 353154/400000 [00:49<00:06, 7075.91it/s] 88%|████████▊ | 353864/400000 [00:49<00:06, 6757.15it/s] 89%|████████▊ | 354545/400000 [00:49<00:06, 6679.86it/s] 89%|████████▉ | 355267/400000 [00:49<00:06, 6833.00it/s] 89%|████████▉ | 356041/400000 [00:49<00:06, 7080.38it/s] 89%|████████▉ | 356762/400000 [00:49<00:06, 7116.60it/s] 89%|████████▉ | 357477/400000 [00:49<00:06, 6912.29it/s] 90%|████████▉ | 358172/400000 [00:49<00:06, 6798.57it/s] 90%|████████▉ | 358855/400000 [00:50<00:06, 6682.42it/s] 90%|████████▉ | 359526/400000 [00:50<00:06, 6544.16it/s] 90%|█████████ | 360245/400000 [00:50<00:05, 6724.72it/s] 90%|█████████ | 361028/400000 [00:50<00:05, 7005.29it/s] 90%|█████████ | 361769/400000 [00:50<00:05, 7120.12it/s] 91%|█████████ | 362486/400000 [00:50<00:05, 6929.46it/s] 91%|█████████ | 363183/400000 [00:50<00:05, 6781.39it/s] 91%|█████████ | 363865/400000 [00:50<00:05, 6584.57it/s] 91%|█████████ | 364583/400000 [00:50<00:05, 6752.31it/s] 91%|█████████▏| 365314/400000 [00:50<00:05, 6909.84it/s] 92%|█████████▏| 366065/400000 [00:51<00:04, 7078.05it/s] 92%|█████████▏| 366822/400000 [00:51<00:04, 7218.10it/s] 92%|█████████▏| 367571/400000 [00:51<00:04, 7295.86it/s] 92%|█████████▏| 368303/400000 [00:51<00:04, 7100.90it/s] 92%|█████████▏| 369016/400000 [00:51<00:04, 6897.76it/s] 92%|█████████▏| 369710/400000 [00:51<00:04, 6750.81it/s] 93%|█████████▎| 370389/400000 [00:51<00:04, 6629.19it/s] 93%|█████████▎| 371055/400000 [00:51<00:04, 6477.98it/s] 93%|█████████▎| 371706/400000 [00:51<00:04, 6429.87it/s] 93%|█████████▎| 372351/400000 [00:52<00:04, 6302.27it/s] 93%|█████████▎| 372984/400000 [00:52<00:04, 6050.87it/s] 93%|█████████▎| 373607/400000 [00:52<00:04, 6100.78it/s] 94%|█████████▎| 374354/400000 [00:52<00:03, 6455.23it/s] 94%|█████████▍| 375119/400000 [00:52<00:03, 6772.13it/s] 94%|█████████▍| 375884/400000 [00:52<00:03, 7012.06it/s] 94%|█████████▍| 376616/400000 [00:52<00:03, 7093.16it/s] 94%|█████████▍| 377376/400000 [00:52<00:03, 7236.49it/s] 95%|█████████▍| 378105/400000 [00:52<00:03, 7069.22it/s] 95%|█████████▍| 378817/400000 [00:52<00:03, 6890.81it/s] 95%|█████████▍| 379511/400000 [00:53<00:03, 6819.41it/s] 95%|█████████▌| 380255/400000 [00:53<00:02, 6993.35it/s] 95%|█████████▌| 381004/400000 [00:53<00:02, 7134.44it/s] 95%|█████████▌| 381777/400000 [00:53<00:02, 7302.33it/s] 96%|█████████▌| 382541/400000 [00:53<00:02, 7398.21it/s] 96%|█████████▌| 383302/400000 [00:53<00:02, 7458.79it/s] 96%|█████████▌| 384050/400000 [00:53<00:02, 7455.15it/s] 96%|█████████▌| 384809/400000 [00:53<00:02, 7492.15it/s] 96%|█████████▋| 385560/400000 [00:53<00:01, 7439.28it/s] 97%|█████████▋| 386305/400000 [00:53<00:01, 7111.18it/s] 97%|█████████▋| 387020/400000 [00:54<00:01, 6866.28it/s] 97%|█████████▋| 387711/400000 [00:54<00:01, 6743.40it/s] 97%|█████████▋| 388410/400000 [00:54<00:01, 6813.60it/s] 97%|█████████▋| 389122/400000 [00:54<00:01, 6902.38it/s] 97%|█████████▋| 389862/400000 [00:54<00:01, 7043.90it/s] 98%|█████████▊| 390628/400000 [00:54<00:01, 7217.86it/s] 98%|█████████▊| 391366/400000 [00:54<00:01, 7265.65it/s] 98%|█████████▊| 392095/400000 [00:54<00:01, 7052.09it/s] 98%|█████████▊| 392803/400000 [00:54<00:01, 6879.92it/s] 98%|█████████▊| 393494/400000 [00:55<00:00, 6717.56it/s] 99%|█████████▊| 394169/400000 [00:55<00:00, 6665.79it/s] 99%|█████████▊| 394838/400000 [00:55<00:00, 6580.29it/s] 99%|█████████▉| 395579/400000 [00:55<00:00, 6807.51it/s] 99%|█████████▉| 396342/400000 [00:55<00:00, 7027.91it/s] 99%|█████████▉| 397115/400000 [00:55<00:00, 7223.72it/s] 99%|█████████▉| 397842/400000 [00:55<00:00, 7087.92it/s]100%|█████████▉| 398555/400000 [00:55<00:00, 6905.40it/s]100%|█████████▉| 399250/400000 [00:55<00:00, 6725.99it/s]100%|█████████▉| 399927/400000 [00:55<00:00, 6636.98it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7147.53it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fd201e920f0>, <torchtext.data.dataset.TabularDataset object at 0x7fd201e92240>, <torchtext.vocab.Vocab object at 0x7fd201e92160>), {}) 

