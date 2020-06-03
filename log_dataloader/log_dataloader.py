
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5e4770d378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5e4770d378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5eb975e0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5eb975e0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5e50ef29d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5e50ef29d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e670428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e670428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e670428c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 25%|██▍       | 2465792/9912422 [00:00<00:00, 24202151.62it/s] 96%|█████████▋| 9560064/9912422 [00:00<00:00, 30133265.05it/s]9920512it [00:00, 30458442.03it/s]                             
0it [00:00, ?it/s]32768it [00:00, 503634.94it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7825698.92it/s]          
0it [00:00, ?it/s]8192it [00:00, 114095.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e4a713fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e4a71bf28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5e67042510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5e67042510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5e67042510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:48,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:48,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:48,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:47,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:47,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:33,  4.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:33,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:33,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:33,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:32,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:32,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:31,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:22,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:22,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 12.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 12.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 16.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:06, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:06, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:06, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:06, 16.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 21.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 67.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 67.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 72.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 73.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.75 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.75 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.74 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.29s/ url]
0 examples [00:00, ? examples/s]2020-06-03 00:09:17.361589: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-03 00:09:17.374357: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-03 00:09:17.374514: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e02a9d18c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-03 00:09:17.374534: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
46 examples [00:00, 457.22 examples/s]136 examples [00:00, 536.03 examples/s]234 examples [00:00, 619.03 examples/s]334 examples [00:00, 698.10 examples/s]428 examples [00:00, 754.97 examples/s]524 examples [00:00, 805.32 examples/s]616 examples [00:00, 836.16 examples/s]718 examples [00:00, 882.76 examples/s]816 examples [00:00, 908.19 examples/s]914 examples [00:01, 928.57 examples/s]1015 examples [00:01, 945.69 examples/s]1111 examples [00:01, 930.14 examples/s]1205 examples [00:01, 929.30 examples/s]1305 examples [00:01, 947.56 examples/s]1401 examples [00:01, 946.74 examples/s]1496 examples [00:01, 937.81 examples/s]1591 examples [00:01, 939.58 examples/s]1686 examples [00:01, 924.01 examples/s]1783 examples [00:01, 936.06 examples/s]1877 examples [00:02, 927.97 examples/s]1970 examples [00:02, 911.17 examples/s]2062 examples [00:02, 909.75 examples/s]2158 examples [00:02, 923.10 examples/s]2251 examples [00:02, 922.98 examples/s]2347 examples [00:02, 933.06 examples/s]2449 examples [00:02, 953.41 examples/s]2545 examples [00:02, 951.48 examples/s]2641 examples [00:02, 934.05 examples/s]2737 examples [00:02, 940.87 examples/s]2844 examples [00:03, 975.36 examples/s]2948 examples [00:03, 992.68 examples/s]3048 examples [00:03, 985.98 examples/s]3153 examples [00:03, 1001.33 examples/s]3255 examples [00:03, 1004.79 examples/s]3357 examples [00:03, 1007.42 examples/s]3460 examples [00:03, 1011.22 examples/s]3562 examples [00:03, 986.19 examples/s] 3661 examples [00:03, 962.63 examples/s]3761 examples [00:03, 972.02 examples/s]3863 examples [00:04, 985.83 examples/s]3962 examples [00:04, 981.01 examples/s]4061 examples [00:04, 946.25 examples/s]4156 examples [00:04, 921.92 examples/s]4249 examples [00:04, 915.88 examples/s]4345 examples [00:04, 928.65 examples/s]4440 examples [00:04, 933.85 examples/s]4536 examples [00:04, 940.65 examples/s]4631 examples [00:04, 903.15 examples/s]4735 examples [00:05, 940.12 examples/s]4834 examples [00:05, 953.87 examples/s]4930 examples [00:05, 947.11 examples/s]5037 examples [00:05, 978.90 examples/s]5140 examples [00:05, 992.09 examples/s]5243 examples [00:05, 1000.39 examples/s]5344 examples [00:05, 993.98 examples/s] 5444 examples [00:05, 991.09 examples/s]5545 examples [00:05, 995.53 examples/s]5645 examples [00:05, 992.51 examples/s]5745 examples [00:06, 976.08 examples/s]5843 examples [00:06, 960.97 examples/s]5940 examples [00:06, 937.72 examples/s]6034 examples [00:06, 921.10 examples/s]6127 examples [00:06, 899.94 examples/s]6219 examples [00:06, 903.84 examples/s]6310 examples [00:06, 888.22 examples/s]6400 examples [00:06, 885.40 examples/s]6494 examples [00:06, 898.60 examples/s]6587 examples [00:06, 905.92 examples/s]6683 examples [00:07, 918.61 examples/s]6775 examples [00:07, 918.04 examples/s]6867 examples [00:07, 896.57 examples/s]6957 examples [00:07, 865.05 examples/s]7047 examples [00:07, 874.40 examples/s]7140 examples [00:07, 888.35 examples/s]7233 examples [00:07, 898.64 examples/s]7324 examples [00:07, 877.56 examples/s]7413 examples [00:07, 863.25 examples/s]7500 examples [00:08, 863.20 examples/s]7587 examples [00:08, 864.68 examples/s]7676 examples [00:08, 872.06 examples/s]7764 examples [00:08, 849.16 examples/s]7854 examples [00:08, 863.41 examples/s]7950 examples [00:08, 890.25 examples/s]8043 examples [00:08, 901.72 examples/s]8140 examples [00:08, 918.81 examples/s]8234 examples [00:08, 923.14 examples/s]8332 examples [00:08, 937.17 examples/s]8442 examples [00:09, 979.89 examples/s]8541 examples [00:09, 962.23 examples/s]8638 examples [00:09, 936.35 examples/s]8736 examples [00:09, 946.90 examples/s]8832 examples [00:09, 914.78 examples/s]8935 examples [00:09, 944.33 examples/s]9031 examples [00:09, 910.29 examples/s]9123 examples [00:09, 911.82 examples/s]9215 examples [00:09, 903.85 examples/s]9311 examples [00:09, 919.99 examples/s]9418 examples [00:10, 959.52 examples/s]9522 examples [00:10, 981.48 examples/s]9621 examples [00:10, 982.93 examples/s]9720 examples [00:10, 984.10 examples/s]9820 examples [00:10, 988.04 examples/s]9920 examples [00:10, 960.71 examples/s]10017 examples [00:10, 903.17 examples/s]10119 examples [00:10, 934.28 examples/s]10221 examples [00:10, 956.94 examples/s]10318 examples [00:11, 937.04 examples/s]10413 examples [00:11, 913.50 examples/s]10507 examples [00:11, 920.09 examples/s]10600 examples [00:11, 910.76 examples/s]10692 examples [00:11, 912.51 examples/s]10784 examples [00:11, 907.04 examples/s]10877 examples [00:11, 912.65 examples/s]10970 examples [00:11, 915.15 examples/s]11062 examples [00:11, 909.35 examples/s]11155 examples [00:11, 912.95 examples/s]11249 examples [00:12, 919.91 examples/s]11342 examples [00:12, 922.62 examples/s]11435 examples [00:12, 924.45 examples/s]11528 examples [00:12, 880.64 examples/s]11620 examples [00:12, 890.72 examples/s]11715 examples [00:12, 905.77 examples/s]11810 examples [00:12, 916.52 examples/s]11902 examples [00:12, 910.31 examples/s]11994 examples [00:12, 912.09 examples/s]12096 examples [00:12, 941.02 examples/s]12191 examples [00:13, 932.28 examples/s]12286 examples [00:13, 937.11 examples/s]12380 examples [00:13, 934.28 examples/s]12474 examples [00:13, 913.00 examples/s]12566 examples [00:13, 904.03 examples/s]12663 examples [00:13, 921.39 examples/s]12756 examples [00:13, 902.67 examples/s]12847 examples [00:13, 901.79 examples/s]12938 examples [00:13, 892.71 examples/s]13036 examples [00:14, 914.43 examples/s]13128 examples [00:14, 886.90 examples/s]13227 examples [00:14, 913.04 examples/s]13323 examples [00:14, 924.05 examples/s]13416 examples [00:14, 908.27 examples/s]13511 examples [00:14, 920.30 examples/s]13612 examples [00:14, 944.70 examples/s]13713 examples [00:14, 962.05 examples/s]13814 examples [00:14, 973.82 examples/s]13913 examples [00:14, 975.68 examples/s]14016 examples [00:15, 989.78 examples/s]14117 examples [00:15, 995.45 examples/s]14218 examples [00:15, 998.27 examples/s]14324 examples [00:15, 1013.34 examples/s]14429 examples [00:15, 1022.04 examples/s]14532 examples [00:15, 1018.64 examples/s]14634 examples [00:15, 1013.37 examples/s]14741 examples [00:15, 1028.59 examples/s]14844 examples [00:15, 1020.04 examples/s]14947 examples [00:15, 1019.22 examples/s]15049 examples [00:16, 1014.26 examples/s]15151 examples [00:16, 1003.78 examples/s]15252 examples [00:16, 991.22 examples/s] 15352 examples [00:16, 989.83 examples/s]15455 examples [00:16, 1000.65 examples/s]15556 examples [00:16, 1001.46 examples/s]15657 examples [00:16, 1000.49 examples/s]15758 examples [00:16, 985.95 examples/s] 15863 examples [00:16, 1002.72 examples/s]15964 examples [00:16, 968.29 examples/s] 16062 examples [00:17, 936.78 examples/s]16157 examples [00:17, 907.63 examples/s]16251 examples [00:17, 915.64 examples/s]16344 examples [00:17, 919.56 examples/s]16437 examples [00:17, 918.68 examples/s]16534 examples [00:17, 933.37 examples/s]16628 examples [00:17, 932.57 examples/s]16722 examples [00:17, 933.66 examples/s]16816 examples [00:17, 925.33 examples/s]16909 examples [00:18, 913.73 examples/s]17010 examples [00:18, 938.99 examples/s]17108 examples [00:18, 949.85 examples/s]17204 examples [00:18, 934.06 examples/s]17298 examples [00:18, 919.78 examples/s]17391 examples [00:18, 921.24 examples/s]17484 examples [00:18, 917.63 examples/s]17578 examples [00:18, 920.80 examples/s]17671 examples [00:18, 922.36 examples/s]17764 examples [00:18, 919.63 examples/s]17856 examples [00:19, 885.54 examples/s]17945 examples [00:19, 861.83 examples/s]18032 examples [00:19, 859.82 examples/s]18137 examples [00:19, 907.88 examples/s]18229 examples [00:19, 892.26 examples/s]18324 examples [00:19, 907.57 examples/s]18421 examples [00:19, 924.37 examples/s]18524 examples [00:19, 952.37 examples/s]18629 examples [00:19, 979.39 examples/s]18728 examples [00:19, 959.69 examples/s]18829 examples [00:20, 974.12 examples/s]18927 examples [00:20, 973.00 examples/s]19025 examples [00:20, 950.65 examples/s]19121 examples [00:20, 942.08 examples/s]19217 examples [00:20, 946.48 examples/s]19321 examples [00:20, 971.08 examples/s]19425 examples [00:20, 989.86 examples/s]19531 examples [00:20, 1007.05 examples/s]19632 examples [00:20, 1006.25 examples/s]19733 examples [00:20, 995.32 examples/s] 19833 examples [00:21, 981.84 examples/s]19932 examples [00:21, 974.61 examples/s]20030 examples [00:21, 900.01 examples/s]20125 examples [00:21, 911.50 examples/s]20218 examples [00:21, 908.72 examples/s]20314 examples [00:21, 921.45 examples/s]20409 examples [00:21, 929.10 examples/s]20512 examples [00:21, 953.75 examples/s]20617 examples [00:21, 978.10 examples/s]20716 examples [00:22, 964.66 examples/s]20821 examples [00:22, 988.33 examples/s]20921 examples [00:22, 973.17 examples/s]21019 examples [00:22, 946.50 examples/s]21115 examples [00:22, 931.13 examples/s]21209 examples [00:22, 928.98 examples/s]21306 examples [00:22, 939.72 examples/s]21401 examples [00:22, 938.02 examples/s]21505 examples [00:22, 965.55 examples/s]21617 examples [00:22, 1004.98 examples/s]21720 examples [00:23, 1009.79 examples/s]21828 examples [00:23, 1028.78 examples/s]21932 examples [00:23, 1028.00 examples/s]22036 examples [00:23, 1013.82 examples/s]22138 examples [00:23, 989.91 examples/s] 22238 examples [00:23, 962.59 examples/s]22342 examples [00:23, 983.17 examples/s]22452 examples [00:23, 1014.33 examples/s]22554 examples [00:23, 1013.54 examples/s]22658 examples [00:24, 1020.13 examples/s]22761 examples [00:24, 1007.78 examples/s]22865 examples [00:24, 1017.00 examples/s]22967 examples [00:24, 991.28 examples/s] 23067 examples [00:24, 913.00 examples/s]23166 examples [00:24, 933.61 examples/s]23270 examples [00:24, 963.06 examples/s]23368 examples [00:24, 952.52 examples/s]23477 examples [00:24, 988.17 examples/s]23579 examples [00:24, 995.52 examples/s]23680 examples [00:25, 996.56 examples/s]23781 examples [00:25, 988.17 examples/s]23881 examples [00:25, 960.10 examples/s]23979 examples [00:25, 961.58 examples/s]24076 examples [00:25, 942.47 examples/s]24171 examples [00:25, 941.82 examples/s]24266 examples [00:25, 943.00 examples/s]24366 examples [00:25, 957.91 examples/s]24472 examples [00:25, 986.05 examples/s]24571 examples [00:25, 985.13 examples/s]24670 examples [00:26, 971.69 examples/s]24768 examples [00:26, 959.68 examples/s]24865 examples [00:26, 957.25 examples/s]24961 examples [00:26, 903.71 examples/s]25055 examples [00:26, 912.60 examples/s]25147 examples [00:26, 910.41 examples/s]25243 examples [00:26, 924.24 examples/s]25337 examples [00:26, 928.57 examples/s]25433 examples [00:26, 935.44 examples/s]25527 examples [00:27, 928.44 examples/s]25620 examples [00:27, 915.74 examples/s]25714 examples [00:27, 921.66 examples/s]25807 examples [00:27, 920.97 examples/s]25903 examples [00:27, 931.64 examples/s]26000 examples [00:27, 941.84 examples/s]26095 examples [00:27, 936.61 examples/s]26189 examples [00:27, 913.29 examples/s]26281 examples [00:27, 910.25 examples/s]26373 examples [00:27, 894.16 examples/s]26463 examples [00:28, 887.73 examples/s]26554 examples [00:28, 893.70 examples/s]26649 examples [00:28, 906.86 examples/s]26740 examples [00:28, 905.77 examples/s]26834 examples [00:28, 913.81 examples/s]26934 examples [00:28, 933.95 examples/s]27030 examples [00:28, 937.39 examples/s]27126 examples [00:28, 939.12 examples/s]27233 examples [00:28, 972.94 examples/s]27338 examples [00:28, 993.84 examples/s]27446 examples [00:29, 1017.71 examples/s]27549 examples [00:29, 995.54 examples/s] 27649 examples [00:29, 986.88 examples/s]27748 examples [00:29, 975.13 examples/s]27846 examples [00:29, 939.42 examples/s]27948 examples [00:29, 961.75 examples/s]28045 examples [00:29, 942.75 examples/s]28146 examples [00:29, 961.94 examples/s]28243 examples [00:29, 950.44 examples/s]28342 examples [00:30, 960.06 examples/s]28444 examples [00:30, 975.20 examples/s]28549 examples [00:30, 995.25 examples/s]28657 examples [00:30, 1019.24 examples/s]28762 examples [00:30, 1026.25 examples/s]28866 examples [00:30, 1028.80 examples/s]28970 examples [00:30, 1023.63 examples/s]29073 examples [00:30, 1018.54 examples/s]29175 examples [00:30, 994.00 examples/s] 29276 examples [00:30, 996.22 examples/s]29376 examples [00:31, 987.71 examples/s]29475 examples [00:31, 969.53 examples/s]29573 examples [00:31, 956.94 examples/s]29669 examples [00:31, 952.20 examples/s]29765 examples [00:31, 947.10 examples/s]29860 examples [00:31, 933.39 examples/s]29954 examples [00:31, 900.43 examples/s]30045 examples [00:31, 844.16 examples/s]30152 examples [00:31, 900.99 examples/s]30255 examples [00:31, 934.06 examples/s]30362 examples [00:32, 968.85 examples/s]30461 examples [00:32, 960.68 examples/s]30559 examples [00:32, 948.63 examples/s]30655 examples [00:32, 940.60 examples/s]30750 examples [00:32, 918.57 examples/s]30845 examples [00:32, 927.54 examples/s]30939 examples [00:32, 899.28 examples/s]31036 examples [00:32, 917.18 examples/s]31135 examples [00:32, 935.75 examples/s]31229 examples [00:33, 908.38 examples/s]31321 examples [00:33, 883.64 examples/s]31410 examples [00:33, 850.58 examples/s]31496 examples [00:33, 849.09 examples/s]31582 examples [00:33, 845.85 examples/s]31670 examples [00:33, 854.21 examples/s]31756 examples [00:33, 848.93 examples/s]31842 examples [00:33, 830.12 examples/s]31931 examples [00:33, 845.65 examples/s]32025 examples [00:33, 869.97 examples/s]32121 examples [00:34, 894.98 examples/s]32212 examples [00:34, 897.38 examples/s]32303 examples [00:34, 892.98 examples/s]32393 examples [00:34, 891.14 examples/s]32483 examples [00:34, 869.29 examples/s]32577 examples [00:34, 888.37 examples/s]32674 examples [00:34, 908.92 examples/s]32776 examples [00:34, 939.49 examples/s]32884 examples [00:34, 977.27 examples/s]32986 examples [00:34, 985.00 examples/s]33086 examples [00:35, 985.74 examples/s]33185 examples [00:35, 971.93 examples/s]33290 examples [00:35, 992.72 examples/s]33398 examples [00:35, 1016.02 examples/s]33503 examples [00:35, 1024.93 examples/s]33608 examples [00:35, 1029.37 examples/s]33712 examples [00:35, 1025.24 examples/s]33815 examples [00:35, 982.93 examples/s] 33915 examples [00:35, 987.14 examples/s]34015 examples [00:36, 965.29 examples/s]34112 examples [00:36, 963.00 examples/s]34209 examples [00:36, 945.05 examples/s]34304 examples [00:36, 936.40 examples/s]34398 examples [00:36, 910.61 examples/s]34500 examples [00:36, 938.77 examples/s]34595 examples [00:36, 924.53 examples/s]34688 examples [00:36, 922.77 examples/s]34782 examples [00:36, 925.69 examples/s]34876 examples [00:36, 927.83 examples/s]34970 examples [00:37, 929.31 examples/s]35064 examples [00:37, 921.18 examples/s]35157 examples [00:37, 907.16 examples/s]35248 examples [00:37, 873.22 examples/s]35338 examples [00:37, 880.72 examples/s]35432 examples [00:37, 896.69 examples/s]35526 examples [00:37, 907.84 examples/s]35617 examples [00:37, 897.07 examples/s]35709 examples [00:37, 903.62 examples/s]35803 examples [00:37, 913.90 examples/s]35900 examples [00:38, 929.14 examples/s]35998 examples [00:38, 942.29 examples/s]36102 examples [00:38, 968.08 examples/s]36209 examples [00:38, 995.09 examples/s]36309 examples [00:38, 995.93 examples/s]36414 examples [00:38, 1011.15 examples/s]36518 examples [00:38, 1017.94 examples/s]36620 examples [00:38, 988.04 examples/s] 36726 examples [00:38, 1008.14 examples/s]36828 examples [00:39, 990.36 examples/s] 36928 examples [00:39, 957.26 examples/s]37026 examples [00:39, 961.95 examples/s]37123 examples [00:39, 956.81 examples/s]37225 examples [00:39, 971.67 examples/s]37323 examples [00:39, 967.29 examples/s]37422 examples [00:39, 972.40 examples/s]37527 examples [00:39, 991.94 examples/s]37627 examples [00:39, 991.40 examples/s]37728 examples [00:39, 995.37 examples/s]37828 examples [00:40, 977.14 examples/s]37928 examples [00:40, 983.13 examples/s]38027 examples [00:40, 952.78 examples/s]38129 examples [00:40, 968.88 examples/s]38227 examples [00:40, 970.44 examples/s]38325 examples [00:40, 972.30 examples/s]38427 examples [00:40, 984.65 examples/s]38535 examples [00:40, 1011.29 examples/s]38637 examples [00:40, 994.59 examples/s] 38737 examples [00:40, 980.04 examples/s]38836 examples [00:41, 980.43 examples/s]38936 examples [00:41, 985.97 examples/s]39035 examples [00:41, 967.98 examples/s]39132 examples [00:41, 923.92 examples/s]39225 examples [00:41, 905.61 examples/s]39317 examples [00:41, 903.51 examples/s]39412 examples [00:41, 914.85 examples/s]39511 examples [00:41, 934.80 examples/s]39607 examples [00:41, 940.56 examples/s]39713 examples [00:42, 972.67 examples/s]39811 examples [00:42, 973.72 examples/s]39916 examples [00:42, 993.39 examples/s]40016 examples [00:42, 912.87 examples/s]40109 examples [00:42, 915.01 examples/s]40205 examples [00:42, 926.81 examples/s]40301 examples [00:42, 936.32 examples/s]40396 examples [00:42, 938.84 examples/s]40497 examples [00:42, 956.09 examples/s]40601 examples [00:42, 979.25 examples/s]40700 examples [00:43, 968.99 examples/s]40800 examples [00:43, 976.63 examples/s]40903 examples [00:43, 990.32 examples/s]41003 examples [00:43, 986.19 examples/s]41103 examples [00:43, 989.17 examples/s]41203 examples [00:43, 979.36 examples/s]41305 examples [00:43, 988.66 examples/s]41410 examples [00:43, 1004.81 examples/s]41511 examples [00:43, 974.89 examples/s] 41609 examples [00:43, 971.66 examples/s]41707 examples [00:44, 962.05 examples/s]41804 examples [00:44, 953.78 examples/s]41900 examples [00:44, 939.81 examples/s]41995 examples [00:44, 914.37 examples/s]42087 examples [00:44, 885.04 examples/s]42185 examples [00:44, 910.63 examples/s]42282 examples [00:44, 926.32 examples/s]42380 examples [00:44, 940.76 examples/s]42477 examples [00:44, 947.88 examples/s]42579 examples [00:45, 966.99 examples/s]42685 examples [00:45, 992.77 examples/s]42785 examples [00:45, 959.84 examples/s]42886 examples [00:45, 971.86 examples/s]42986 examples [00:45, 980.09 examples/s]43085 examples [00:45, 976.40 examples/s]43183 examples [00:45, 976.61 examples/s]43281 examples [00:45, 970.15 examples/s]43383 examples [00:45, 984.31 examples/s]43482 examples [00:45, 966.92 examples/s]43582 examples [00:46, 976.33 examples/s]43680 examples [00:46, 946.04 examples/s]43775 examples [00:46, 915.86 examples/s]43871 examples [00:46, 925.95 examples/s]43966 examples [00:46, 928.23 examples/s]44062 examples [00:46, 937.13 examples/s]44158 examples [00:46, 943.72 examples/s]44259 examples [00:46, 960.83 examples/s]44356 examples [00:46, 958.51 examples/s]44458 examples [00:46, 974.75 examples/s]44557 examples [00:47, 978.64 examples/s]44655 examples [00:47, 967.05 examples/s]44752 examples [00:47, 943.46 examples/s]44847 examples [00:47, 942.48 examples/s]44942 examples [00:47, 930.03 examples/s]45039 examples [00:47, 939.94 examples/s]45134 examples [00:47, 936.12 examples/s]45228 examples [00:47, 930.40 examples/s]45326 examples [00:47, 943.17 examples/s]45421 examples [00:47, 943.33 examples/s]45517 examples [00:48, 947.90 examples/s]45615 examples [00:48, 955.92 examples/s]45714 examples [00:48, 965.34 examples/s]45820 examples [00:48, 990.21 examples/s]45920 examples [00:48, 977.28 examples/s]46018 examples [00:48, 973.34 examples/s]46116 examples [00:48, 959.17 examples/s]46219 examples [00:48, 978.02 examples/s]46318 examples [00:48, 979.13 examples/s]46417 examples [00:49, 975.22 examples/s]46520 examples [00:49, 988.24 examples/s]46619 examples [00:49, 982.67 examples/s]46718 examples [00:49, 957.55 examples/s]46814 examples [00:49, 939.45 examples/s]46909 examples [00:49, 898.32 examples/s]47002 examples [00:49, 906.22 examples/s]47098 examples [00:49, 921.37 examples/s]47201 examples [00:49, 948.85 examples/s]47297 examples [00:49, 948.52 examples/s]47399 examples [00:50, 966.42 examples/s]47501 examples [00:50, 981.71 examples/s]47600 examples [00:50, 971.52 examples/s]47698 examples [00:50, 924.65 examples/s]47795 examples [00:50, 936.28 examples/s]47890 examples [00:50, 938.93 examples/s]47985 examples [00:50, 911.21 examples/s]48081 examples [00:50, 923.67 examples/s]48176 examples [00:50, 929.87 examples/s]48270 examples [00:50, 921.90 examples/s]48363 examples [00:51, 921.33 examples/s]48459 examples [00:51, 930.79 examples/s]48553 examples [00:51, 897.41 examples/s]48644 examples [00:51, 884.31 examples/s]48735 examples [00:51, 889.15 examples/s]48825 examples [00:51, 883.19 examples/s]48914 examples [00:51, 875.28 examples/s]49013 examples [00:51, 904.92 examples/s]49108 examples [00:51, 916.44 examples/s]49200 examples [00:52, 907.86 examples/s]49292 examples [00:52, 908.72 examples/s]49384 examples [00:52, 910.76 examples/s]49477 examples [00:52, 914.47 examples/s]49569 examples [00:52, 914.30 examples/s]49661 examples [00:52, 891.89 examples/s]49751 examples [00:52, 889.34 examples/s]49842 examples [00:52, 894.32 examples/s]49935 examples [00:52, 903.55 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7435/50000 [00:00<00:00, 74346.70 examples/s] 39%|███▊      | 19257/50000 [00:00<00:00, 83660.47 examples/s] 64%|██████▎   | 31841/50000 [00:00<00:00, 93012.58 examples/s] 90%|█████████ | 45099/50000 [00:00<00:00, 102158.61 examples/s]                                                                0 examples [00:00, ? examples/s]68 examples [00:00, 675.80 examples/s]161 examples [00:00, 735.78 examples/s]260 examples [00:00, 796.74 examples/s]355 examples [00:00, 836.51 examples/s]455 examples [00:00, 878.08 examples/s]548 examples [00:00, 892.60 examples/s]655 examples [00:00, 938.47 examples/s]746 examples [00:00, 764.99 examples/s]834 examples [00:00, 795.38 examples/s]924 examples [00:01, 822.62 examples/s]1008 examples [00:01, 819.02 examples/s]1095 examples [00:01, 833.35 examples/s]1183 examples [00:01, 844.41 examples/s]1276 examples [00:01, 866.06 examples/s]1367 examples [00:01, 878.69 examples/s]1465 examples [00:01, 906.31 examples/s]1562 examples [00:01, 921.80 examples/s]1655 examples [00:01, 923.69 examples/s]1748 examples [00:01, 920.98 examples/s]1841 examples [00:02, 904.44 examples/s]1934 examples [00:02, 911.74 examples/s]2026 examples [00:02, 911.08 examples/s]2122 examples [00:02, 923.72 examples/s]2217 examples [00:02, 931.04 examples/s]2311 examples [00:02, 906.17 examples/s]2404 examples [00:02, 912.82 examples/s]2496 examples [00:02, 906.69 examples/s]2591 examples [00:02, 919.24 examples/s]2689 examples [00:03, 936.50 examples/s]2785 examples [00:03, 940.77 examples/s]2881 examples [00:03, 945.66 examples/s]2976 examples [00:03, 934.11 examples/s]3070 examples [00:03, 891.21 examples/s]3161 examples [00:03, 896.76 examples/s]3253 examples [00:03, 902.11 examples/s]3347 examples [00:03, 912.04 examples/s]3440 examples [00:03, 913.76 examples/s]3535 examples [00:03, 922.32 examples/s]3628 examples [00:04, 906.78 examples/s]3719 examples [00:04, 885.83 examples/s]3813 examples [00:04, 898.82 examples/s]3904 examples [00:04, 887.43 examples/s]3995 examples [00:04, 891.89 examples/s]4087 examples [00:04, 898.06 examples/s]4181 examples [00:04, 910.13 examples/s]4273 examples [00:04, 908.95 examples/s]4364 examples [00:04, 909.22 examples/s]4456 examples [00:04, 911.65 examples/s]4556 examples [00:05, 935.41 examples/s]4650 examples [00:05, 926.01 examples/s]4744 examples [00:05, 929.46 examples/s]4838 examples [00:05, 918.28 examples/s]4931 examples [00:05, 921.30 examples/s]5027 examples [00:05, 930.29 examples/s]5122 examples [00:05, 934.68 examples/s]5217 examples [00:05, 938.99 examples/s]5316 examples [00:05, 952.00 examples/s]5412 examples [00:05, 951.52 examples/s]5508 examples [00:06, 937.72 examples/s]5602 examples [00:06, 920.93 examples/s]5695 examples [00:06, 914.65 examples/s]5794 examples [00:06, 935.72 examples/s]5888 examples [00:06, 873.32 examples/s]5984 examples [00:06, 895.64 examples/s]6077 examples [00:06, 905.36 examples/s]6176 examples [00:06, 926.89 examples/s]6277 examples [00:06, 948.15 examples/s]6377 examples [00:07, 960.97 examples/s]6474 examples [00:07, 932.99 examples/s]6568 examples [00:07, 928.28 examples/s]6667 examples [00:07, 944.46 examples/s]6773 examples [00:07, 974.39 examples/s]6875 examples [00:07, 987.22 examples/s]6975 examples [00:07, 972.64 examples/s]7073 examples [00:07, 954.94 examples/s]7173 examples [00:07, 967.02 examples/s]7270 examples [00:07, 965.76 examples/s]7367 examples [00:08, 964.90 examples/s]7470 examples [00:08, 983.01 examples/s]7569 examples [00:08, 961.87 examples/s]7666 examples [00:08, 955.35 examples/s]7762 examples [00:08, 951.03 examples/s]7858 examples [00:08, 943.41 examples/s]7953 examples [00:08, 942.42 examples/s]8049 examples [00:08, 945.98 examples/s]8144 examples [00:08, 927.27 examples/s]8237 examples [00:08, 913.60 examples/s]8333 examples [00:09, 925.11 examples/s]8431 examples [00:09, 938.83 examples/s]8526 examples [00:09, 921.30 examples/s]8619 examples [00:09, 920.18 examples/s]8714 examples [00:09, 928.09 examples/s]8810 examples [00:09, 937.34 examples/s]8916 examples [00:09, 970.83 examples/s]9014 examples [00:09, 968.08 examples/s]9122 examples [00:09, 998.94 examples/s]9228 examples [00:09, 1015.07 examples/s]9332 examples [00:10, 1021.05 examples/s]9435 examples [00:10, 985.60 examples/s] 9536 examples [00:10, 990.86 examples/s]9642 examples [00:10, 1010.57 examples/s]9748 examples [00:10, 1023.39 examples/s]9851 examples [00:10, 1018.65 examples/s]9954 examples [00:10, 1005.06 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQ1N7I7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQ1N7I7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e670428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e670428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e670428c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ed34e5da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e48613358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e670428c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e670428c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e670428c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e48311da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e483117f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5e44f666a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5e44f666a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f5e44f666a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5e44f667b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5e44f667b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5e44f667b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=309660cda117e809e18ef7495ba50b011865ff73c6f4da9e7561fc4f0f6b51da
  Stored in directory: /tmp/pip-ephem-wheel-cache-h8uditar/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<121:55:40, 1.96kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<85:34:53, 2.80kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<59:57:02, 3.99kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:04<41:56:42, 5.70kB/s].vector_cache/glove.6B.zip:   0%|          | 3.30M/862M [00:04<29:16:56, 8.15kB/s].vector_cache/glove.6B.zip:   1%|          | 6.47M/862M [00:04<20:25:29, 11.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:04<14:13:30, 16.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:04<9:54:40, 23.7kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.9M/862M [00:05<6:53:58, 33.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:05<4:48:34, 48.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.2M/862M [00:05<3:21:01, 69.1kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:05<2:20:08, 98.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.7M/862M [00:05<1:37:39, 141kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:05<1:08:08, 201kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.4M/862M [00:05<47:30, 287kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:05<33:12, 408kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:06<23:46, 568kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:08<18:28, 727kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:08<16:46, 800kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:08<12:34, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:08<09:03, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:10<09:36, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:10<08:06, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:10<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:12<07:19, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:12<07:49, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:12<06:04, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:12<04:23, 3.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:14<20:47, 636kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:14<15:53, 832kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:14<11:23, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:16<11:03, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:16<09:04, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:16<06:40, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:18<07:45, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:18<08:07, 1.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:18<06:20, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:20<06:32, 1.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:20<05:55, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:20<04:26, 2.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:21<06:06, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:22<06:54, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:22<05:25, 2.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<03:55, 3.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:23<22:45, 566kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:24<17:13, 747kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:24<12:22, 1.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:25<11:38, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<10:47, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:26<08:08, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:27<07:45, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:27<06:45, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:28<05:00, 2.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<06:28, 1.96MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<07:13, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:42, 2.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<04:08, 3.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<1:33:01, 136kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<1:06:23, 190kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:32<46:39, 270kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<35:30, 353kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<26:07, 480kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<18:34, 674kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<15:55, 783kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<12:24, 1.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<08:56, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<09:12, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<07:41, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<06:53, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<06:04, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<04:30, 2.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<06:04, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<06:45, 1.81MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:41<03:57, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<06:05, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<05:31, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<04:07, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<05:15, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<03:59, 3.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:38, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<03:54, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<04:56, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<03:43, 3.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<02:45, 4.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<46:52, 254kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<34:00, 350kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:51<24:00, 495kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<19:35, 605kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<14:52, 796kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<10:41, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<10:15, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<08:23, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<06:07, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<07:03, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<06:08, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<04:35, 2.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<05:57, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<06:32, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<05:04, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<03:43, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<07:06, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<06:11, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<04:35, 2.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<05:54, 1.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<06:29, 1.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<05:02, 2.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<03:47, 3.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<05:41, 2.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<05:09, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<03:53, 2.93MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:23, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<06:06, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<04:46, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<03:26, 3.28MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<57:50, 195kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<41:39, 271kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<29:21, 384kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<23:06, 486kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<17:19, 648kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<12:23, 904kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<11:17, 990kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<10:09, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<07:40, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<07:09, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<06:08, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:35, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<05:46, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<05:12, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<03:52, 2.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<05:16, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<04:50, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<03:39, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<05:07, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<04:32, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:21<03:20, 3.24MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<06:17, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<05:31, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<04:08, 2.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<05:24, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<04:53, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<03:38, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<05:05, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<04:29, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<03:54, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<02:51, 3.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<13:12, 804kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<14:32, 730kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<11:34, 918kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<08:25, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<10:07, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<12:55, 816kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<10:20, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<07:30, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<05:28, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<10:48, 970kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<12:46, 821kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<10:08, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<07:24, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:33<05:23, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<12:46, 815kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<13:36, 765kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<10:40, 975kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<07:46, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<05:38, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<14:49, 698kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<14:40, 705kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<11:21, 910kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<08:12, 1.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<05:55, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:38<18:58, 542kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<17:32, 585kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<13:19, 770kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:38<09:33, 1.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<06:53, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<38:20, 266kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<31:04, 328kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<22:46, 448kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<16:11, 628kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<11:30, 881kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<25:17, 401kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<21:35, 469kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:42<16:06, 629kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<11:29, 879kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<08:10, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<21:02, 478kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<18:30, 544kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:44<13:46, 731kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<09:53, 1.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<07:05, 1.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<58:35, 171kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<44:46, 223kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:46<32:17, 309kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<22:48, 437kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<18:04, 549kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<16:40, 595kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<12:39, 783kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<09:04, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<06:29, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<17:09, 575kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<15:44, 627kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<11:56, 826kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<08:34, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:50<06:09, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<14:02, 697kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<13:30, 725kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<10:15, 953kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<07:26, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<05:20, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<20:24, 476kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<17:41, 550kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<13:17, 731kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<09:31, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<08:50, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<09:50, 980kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<07:49, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<05:40, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<04:09, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<09:39, 993kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<10:23, 922kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<08:11, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<05:56, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:58<04:18, 2.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:00<11:02, 862kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:00<11:19, 839kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<08:42, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<06:19, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<04:34, 2.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<17:59, 525kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<16:10, 584kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<12:12, 773kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<08:44, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<06:16, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<13:24, 700kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<12:43, 737kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<09:44, 962kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:04<07:02, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<05:07, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<09:35, 970kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<10:13, 910kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<07:54, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<05:49, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<04:14, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<06:47, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<08:15, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<06:32, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:08<04:49, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<03:33, 2.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<08:35, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<09:30, 963kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<07:31, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<05:28, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<03:59, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<09:59, 911kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<10:24, 874kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<08:03, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<05:49, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<04:18, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<07:17, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<08:31, 1.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<06:44, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<04:57, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<03:38, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<06:45, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<08:08, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<06:30, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<04:47, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<03:30, 2.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<1:20:34, 110kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<59:32, 149kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<42:27, 209kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<29:51, 297kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<22:52, 386kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<19:21, 456kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<14:21, 614kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<10:12, 860kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<07:15, 1.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<39:17, 223kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<30:21, 288kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<21:55, 399kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<15:28, 563kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<11:01, 788kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<12:40, 685kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<11:53, 730kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<09:04, 956kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<06:31, 1.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<06:41, 1.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<07:29, 1.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<05:53, 1.46MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<04:29, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<03:14, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<07:53, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<08:32, 1.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<06:34, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<04:51, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:30, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<09:27, 897kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<09:21, 905kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<07:13, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<05:13, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:31<05:57, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<06:43, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<05:22, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<03:54, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<02:52, 2.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<27:28, 303kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<21:47, 383kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<15:46, 528kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<11:14, 740kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<07:57, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<15:54, 520kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<13:32, 611kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<09:58, 828kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<07:12, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<05:09, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<12:26, 659kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<10:57, 748kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<08:13, 996kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<05:53, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<07:05, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<07:06, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<05:26, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<04:02, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<04:27, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<05:12, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<04:04, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<03:04, 2.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<03:49, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:40, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<03:46, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<02:45, 2.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<05:05, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<05:38, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<04:28, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<03:16, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<05:07, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<05:24, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<04:13, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<03:04, 2.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<05:54, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<05:52, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:32, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:50<03:16, 2.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<08:11, 942kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<07:37, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<05:44, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<06:36, 1.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<06:12, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<04:43, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<03:23, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<16:44, 452kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<13:31, 560kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<09:53, 764kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<07:00, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<09:20, 804kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<07:45, 966kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<05:47, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<04:08, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<06:26, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<06:21, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<04:50, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:27, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<06:21, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<06:10, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<04:40, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<05:01, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<04:42, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<03:32, 2.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<02:32, 2.84MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<20:26, 354kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<17:36, 411kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<13:06, 551kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<09:21, 770kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<08:00, 894kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<07:07, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<05:18, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<07:15, 978kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<06:34, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<04:58, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<03:33, 1.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<17:22, 404kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<13:39, 514kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<09:55, 706kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<06:59, 995kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<20:09, 345kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<15:34, 446kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<11:15, 616kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<07:54, 871kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<19:42, 349kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<14:47, 466kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<10:32, 651kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<08:36, 791kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<07:18, 932kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<05:26, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<04:53, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<04:41, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<03:35, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:21<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<03:51, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<03:00, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<03:11, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<03:28, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<02:42, 2.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<01:56, 3.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<15:38, 418kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<12:09, 538kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<11:25, 572kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<08:40, 746kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<07:18, 885kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<05:24, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<04:49, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<04:34, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:28, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:37, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<02:50, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<03:00, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<03:20, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<02:37, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<02:51, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:12, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<02:32, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<02:46, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<03:06, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<01:47, 3.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<03:39, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<03:32, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<02:02, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<03:18, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<03:29, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<02:43, 2.20MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:52, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<03:07, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<02:28, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<02:41, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<02:58, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<02:19, 2.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<01:41, 3.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<04:24, 1.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<04:10, 1.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<03:09, 1.83MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<02:15, 2.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<12:08, 471kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<09:36, 595kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<06:58, 817kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<05:46, 977kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<05:06, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<03:50, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<03:35, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<04:37, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<03:46, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<02:46, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<03:14, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<03:18, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<02:31, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<01:49, 2.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<05:02, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<04:24, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<03:20, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:25, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<03:28, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<03:29, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:38, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<01:54, 2.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<04:11, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<03:58, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<02:59, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:08, 2.44MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<04:36, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<04:12, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<03:11, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:03, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:07, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:23, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<01:43, 2.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<06:53, 741kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<05:49, 876kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<04:18, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:08<03:37, 1.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<01:59, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:10<03:32, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<02:41, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<01:55, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<08:11, 597kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<06:39, 735kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<04:50, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<03:27, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<04:09, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<03:50, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<02:54, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<02:48, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:45, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:01, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:15, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:25, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:36, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<02:02, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<01:27, 3.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<13:52, 333kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<10:34, 436kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<07:36, 605kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<05:19, 856kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<08:55, 510kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<07:06, 640kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<05:10, 876kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<04:19, 1.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<03:54, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:54, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:37, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<03:07, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<02:17, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:28, 1.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<01:53, 2.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:22, 3.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<03:40, 1.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<03:18, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<02:28, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<01:45, 2.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<07:47, 540kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<06:12, 676kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<04:31, 925kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:49, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<04:07, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:34<03:12, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<02:23, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<01:42, 2.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<05:51, 695kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<04:50, 840kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<03:31, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:32, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<03:00, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:49, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<02:09, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<02:09, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<02:12, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<01:42, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<01:14, 3.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<03:55, 986kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<03:26, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<02:34, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<05:53, 644kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<04:36, 822kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<03:19, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<02:20, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<09:51, 378kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<07:33, 493kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<05:24, 686kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<03:47, 967kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<04:40, 782kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<03:53, 940kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<02:51, 1.27MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<02:35, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<02:26, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:50, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<01:19, 2.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<03:39, 961kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<03:10, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<02:21, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:13, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:09, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:39, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<01:43, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:55<01:47, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<01:22, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:56<00:59, 3.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<06:17, 527kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<04:57, 668kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<03:35, 916kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<03:02, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<03:15, 996kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<02:33, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<01:50, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:08, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:03, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:34, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:36, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:30, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:08, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:23, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:21, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:01, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:18, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:06, 2.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:15, 2.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:22, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:04, 2.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:51, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:30, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:06, 2.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:30, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:31, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:10, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:20, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:03, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:10, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:44, 1.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:26, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:02, 2.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:25, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:26, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:06, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:11, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:15, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<00:58, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:05, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:36, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<01:18, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<00:56, 2.51MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:14, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<01:15, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<00:58, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:03, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:32, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<00:55, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:15, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<00:58, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:01, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:28, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:11, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<00:51, 2.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:06, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:07, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:51, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:59, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:46, 2.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:51, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:56, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<00:43, 2.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:49, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:53, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<00:41, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:29, 3.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<03:33, 505kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<02:47, 643kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<02:00, 883kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<01:23, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<10:10, 170kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<07:24, 233kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<05:12, 328kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<03:49, 434kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<03:15, 509kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<02:23, 691kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<01:42, 955kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:27, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:40, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:57, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<00:55, 1.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:42, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:43, 2.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:01, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<00:49, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<00:35, 2.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:42, 1.97MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:43, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:53<00:33, 2.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<00:36, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:39, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:30, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:21, 3.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<04:50, 257kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<03:35, 346kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:57<02:32, 484kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<01:42, 687kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<35:41, 33.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<25:06, 46.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<17:22, 66.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<11:51, 95.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<08:40, 128kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<06:12, 178kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<04:24, 249kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<03:01, 354kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<02:20, 443kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<01:49, 569kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:17, 790kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:03<00:53, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<01:01, 952kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:52, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:39, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:34, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<00:33, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:24, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:25, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:18, 2.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:08<00:13, 3.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:23, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:17, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:18, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:24, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:20, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:15, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:16, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:18, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:14, 2.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:10, 3.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:23, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:24, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:16<00:18, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:12, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:27, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:25, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:19, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:12, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:25, 980kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:23, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:17, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:11, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:14, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:14, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:07, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:13, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:13, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:09, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:05, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:14, 882kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:12, 985kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:08, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:11, 738kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:09, 862kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:06, 1.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:06, 622kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:05, 730kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:03, 981kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 815kB/s] .vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<25:53:54,  4.29it/s]  0%|          | 868/400000 [00:00<18:05:36,  6.13it/s]  0%|          | 1701/400000 [00:00<12:38:34,  8.75it/s]  1%|          | 2455/400000 [00:00<8:50:15, 12.50it/s]   1%|          | 3195/400000 [00:00<6:10:45, 17.84it/s]  1%|          | 3918/400000 [00:00<4:19:20, 25.46it/s]  1%|          | 4646/400000 [00:00<3:01:28, 36.31it/s]  1%|▏         | 5457/400000 [00:00<2:07:00, 51.77it/s]  2%|▏         | 6221/400000 [00:01<1:28:59, 73.75it/s]  2%|▏         | 6952/400000 [00:01<1:02:26, 104.90it/s]  2%|▏         | 7739/400000 [00:01<43:52, 149.00it/s]    2%|▏         | 8561/400000 [00:01<30:53, 211.22it/s]  2%|▏         | 9428/400000 [00:01<21:47, 298.62it/s]  3%|▎         | 10225/400000 [00:01<15:29, 419.50it/s]  3%|▎         | 11010/400000 [00:01<11:05, 584.94it/s]  3%|▎         | 11785/400000 [00:01<07:59, 809.42it/s]  3%|▎         | 12605/400000 [00:01<05:49, 1109.35it/s]  3%|▎         | 13391/400000 [00:01<04:18, 1493.28it/s]  4%|▎         | 14174/400000 [00:02<03:16, 1966.78it/s]  4%|▎         | 14968/400000 [00:02<02:31, 2539.73it/s]  4%|▍         | 15749/400000 [00:02<02:01, 3155.50it/s]  4%|▍         | 16514/400000 [00:02<01:40, 3810.34it/s]  4%|▍         | 17345/400000 [00:02<01:24, 4548.93it/s]  5%|▍         | 18124/400000 [00:02<01:13, 5177.01it/s]  5%|▍         | 18899/400000 [00:02<01:06, 5696.11it/s]  5%|▍         | 19665/400000 [00:02<01:02, 6109.31it/s]  5%|▌         | 20422/400000 [00:02<00:59, 6400.23it/s]  5%|▌         | 21168/400000 [00:02<00:57, 6590.34it/s]  5%|▌         | 21924/400000 [00:03<00:55, 6852.73it/s]  6%|▌         | 22665/400000 [00:03<00:54, 6925.09it/s]  6%|▌         | 23397/400000 [00:03<00:53, 7001.96it/s]  6%|▌         | 24150/400000 [00:03<00:52, 7150.59it/s]  6%|▌         | 24886/400000 [00:03<00:53, 6949.61it/s]  6%|▋         | 25647/400000 [00:03<00:52, 7133.80it/s]  7%|▋         | 26460/400000 [00:03<00:50, 7405.54it/s]  7%|▋         | 27316/400000 [00:03<00:48, 7715.34it/s]  7%|▋         | 28099/400000 [00:03<00:48, 7646.80it/s]  7%|▋         | 28872/400000 [00:04<00:49, 7507.41it/s]  7%|▋         | 29629/400000 [00:04<00:50, 7405.52it/s]  8%|▊         | 30447/400000 [00:04<00:48, 7620.97it/s]  8%|▊         | 31241/400000 [00:04<00:47, 7712.65it/s]  8%|▊         | 32075/400000 [00:04<00:46, 7842.37it/s]  8%|▊         | 32883/400000 [00:04<00:46, 7908.47it/s]  8%|▊         | 33677/400000 [00:04<00:49, 7422.24it/s]  9%|▊         | 34427/400000 [00:04<00:50, 7284.60it/s]  9%|▉         | 35235/400000 [00:04<00:48, 7504.53it/s]  9%|▉         | 36028/400000 [00:04<00:47, 7625.61it/s]  9%|▉         | 36796/400000 [00:05<00:47, 7570.38it/s]  9%|▉         | 37557/400000 [00:05<00:47, 7562.57it/s] 10%|▉         | 38348/400000 [00:05<00:47, 7661.67it/s] 10%|▉         | 39127/400000 [00:05<00:46, 7699.10it/s] 10%|▉         | 39932/400000 [00:05<00:46, 7799.41it/s] 10%|█         | 40714/400000 [00:05<00:46, 7774.15it/s] 10%|█         | 41493/400000 [00:05<00:46, 7639.70it/s] 11%|█         | 42259/400000 [00:05<00:47, 7602.38it/s] 11%|█         | 43021/400000 [00:05<00:47, 7561.80it/s] 11%|█         | 43806/400000 [00:05<00:46, 7643.22it/s] 11%|█         | 44571/400000 [00:06<00:47, 7528.98it/s] 11%|█▏        | 45352/400000 [00:06<00:46, 7609.53it/s] 12%|█▏        | 46114/400000 [00:06<00:46, 7561.58it/s] 12%|█▏        | 46912/400000 [00:06<00:45, 7681.80it/s] 12%|█▏        | 47682/400000 [00:06<00:45, 7682.04it/s] 12%|█▏        | 48451/400000 [00:06<00:46, 7530.69it/s] 12%|█▏        | 49206/400000 [00:06<00:47, 7335.79it/s] 12%|█▏        | 49944/400000 [00:06<00:47, 7348.43it/s] 13%|█▎        | 50681/400000 [00:06<00:47, 7334.18it/s] 13%|█▎        | 51453/400000 [00:06<00:46, 7443.82it/s] 13%|█▎        | 52213/400000 [00:07<00:46, 7488.69it/s] 13%|█▎        | 52979/400000 [00:07<00:46, 7537.36it/s] 13%|█▎        | 53757/400000 [00:07<00:45, 7606.50it/s] 14%|█▎        | 54567/400000 [00:07<00:44, 7746.61it/s] 14%|█▍        | 55343/400000 [00:07<00:45, 7655.08it/s] 14%|█▍        | 56110/400000 [00:07<00:45, 7479.38it/s] 14%|█▍        | 56873/400000 [00:07<00:45, 7522.62it/s] 14%|█▍        | 57690/400000 [00:07<00:44, 7705.07it/s] 15%|█▍        | 58552/400000 [00:07<00:42, 7956.75it/s] 15%|█▍        | 59390/400000 [00:08<00:42, 8077.74it/s] 15%|█▌        | 60201/400000 [00:08<00:42, 7915.94it/s] 15%|█▌        | 61045/400000 [00:08<00:42, 8065.05it/s] 15%|█▌        | 61877/400000 [00:08<00:41, 8138.71it/s] 16%|█▌        | 62693/400000 [00:08<00:41, 8090.35it/s] 16%|█▌        | 63558/400000 [00:08<00:40, 8248.67it/s] 16%|█▌        | 64385/400000 [00:08<00:42, 7968.67it/s] 16%|█▋        | 65186/400000 [00:08<00:43, 7738.62it/s] 16%|█▋        | 65964/400000 [00:08<00:43, 7748.92it/s] 17%|█▋        | 66836/400000 [00:08<00:41, 8015.15it/s] 17%|█▋        | 67681/400000 [00:09<00:40, 8137.84it/s] 17%|█▋        | 68499/400000 [00:09<00:41, 7974.86it/s] 17%|█▋        | 69325/400000 [00:09<00:41, 8057.50it/s] 18%|█▊        | 70135/400000 [00:09<00:40, 8067.98it/s] 18%|█▊        | 70987/400000 [00:09<00:40, 8196.29it/s] 18%|█▊        | 71841/400000 [00:09<00:39, 8296.19it/s] 18%|█▊        | 72673/400000 [00:09<00:40, 8082.29it/s] 18%|█▊        | 73540/400000 [00:09<00:39, 8248.54it/s] 19%|█▊        | 74368/400000 [00:09<00:39, 8145.06it/s] 19%|█▉        | 75185/400000 [00:09<00:40, 7999.26it/s] 19%|█▉        | 75987/400000 [00:10<00:41, 7877.89it/s] 19%|█▉        | 76777/400000 [00:10<00:42, 7593.87it/s] 19%|█▉        | 77563/400000 [00:10<00:42, 7671.07it/s] 20%|█▉        | 78347/400000 [00:10<00:41, 7719.18it/s] 20%|█▉        | 79156/400000 [00:10<00:41, 7824.13it/s] 20%|█▉        | 79964/400000 [00:10<00:40, 7897.34it/s] 20%|██        | 80756/400000 [00:10<00:41, 7630.73it/s] 20%|██        | 81522/400000 [00:10<00:41, 7602.12it/s] 21%|██        | 82285/400000 [00:10<00:42, 7514.14it/s] 21%|██        | 83057/400000 [00:11<00:41, 7573.75it/s] 21%|██        | 83816/400000 [00:11<00:43, 7346.80it/s] 21%|██        | 84554/400000 [00:11<00:43, 7288.18it/s] 21%|██▏       | 85308/400000 [00:11<00:42, 7356.87it/s] 22%|██▏       | 86068/400000 [00:11<00:42, 7427.51it/s] 22%|██▏       | 86812/400000 [00:11<00:42, 7396.73it/s] 22%|██▏       | 87599/400000 [00:11<00:41, 7530.72it/s] 22%|██▏       | 88354/400000 [00:11<00:41, 7500.15it/s] 22%|██▏       | 89155/400000 [00:11<00:40, 7644.30it/s] 22%|██▏       | 89960/400000 [00:11<00:39, 7757.54it/s] 23%|██▎       | 90816/400000 [00:12<00:38, 7980.13it/s] 23%|██▎       | 91665/400000 [00:12<00:37, 8125.10it/s] 23%|██▎       | 92487/400000 [00:12<00:37, 8153.00it/s] 23%|██▎       | 93304/400000 [00:12<00:38, 8033.14it/s] 24%|██▎       | 94109/400000 [00:12<00:38, 7882.51it/s] 24%|██▎       | 94923/400000 [00:12<00:38, 7956.22it/s] 24%|██▍       | 95721/400000 [00:12<00:38, 7805.16it/s] 24%|██▍       | 96504/400000 [00:12<00:40, 7535.41it/s] 24%|██▍       | 97266/400000 [00:12<00:40, 7560.27it/s] 25%|██▍       | 98068/400000 [00:12<00:39, 7690.27it/s] 25%|██▍       | 98900/400000 [00:13<00:38, 7868.40it/s] 25%|██▍       | 99735/400000 [00:13<00:37, 8004.93it/s] 25%|██▌       | 100538/400000 [00:13<00:38, 7850.54it/s] 25%|██▌       | 101326/400000 [00:13<00:38, 7787.75it/s] 26%|██▌       | 102179/400000 [00:13<00:37, 7995.58it/s] 26%|██▌       | 102982/400000 [00:13<00:37, 7849.68it/s] 26%|██▌       | 103770/400000 [00:13<00:38, 7622.19it/s] 26%|██▌       | 104538/400000 [00:13<00:38, 7637.98it/s] 26%|██▋       | 105304/400000 [00:13<00:38, 7626.14it/s] 27%|██▋       | 106162/400000 [00:13<00:37, 7887.50it/s] 27%|██▋       | 106966/400000 [00:14<00:36, 7932.10it/s] 27%|██▋       | 107806/400000 [00:14<00:36, 8065.94it/s] 27%|██▋       | 108615/400000 [00:14<00:37, 7874.98it/s] 27%|██▋       | 109453/400000 [00:14<00:36, 8018.43it/s] 28%|██▊       | 110289/400000 [00:14<00:35, 8117.12it/s] 28%|██▊       | 111103/400000 [00:14<00:36, 7967.19it/s] 28%|██▊       | 111902/400000 [00:14<00:36, 7811.33it/s] 28%|██▊       | 112695/400000 [00:14<00:36, 7845.73it/s] 28%|██▊       | 113513/400000 [00:14<00:36, 7942.54it/s] 29%|██▊       | 114309/400000 [00:15<00:36, 7769.18it/s] 29%|██▉       | 115088/400000 [00:15<00:37, 7646.68it/s] 29%|██▉       | 115941/400000 [00:15<00:36, 7890.49it/s] 29%|██▉       | 116740/400000 [00:15<00:35, 7919.80it/s] 29%|██▉       | 117642/400000 [00:15<00:34, 8219.01it/s] 30%|██▉       | 118496/400000 [00:15<00:33, 8312.39it/s] 30%|██▉       | 119340/400000 [00:15<00:33, 8348.84it/s] 30%|███       | 120178/400000 [00:15<00:33, 8304.82it/s] 30%|███       | 121011/400000 [00:15<00:34, 8032.94it/s] 30%|███       | 121883/400000 [00:15<00:33, 8225.36it/s] 31%|███       | 122709/400000 [00:16<00:33, 8176.10it/s] 31%|███       | 123530/400000 [00:16<00:33, 8155.43it/s] 31%|███       | 124348/400000 [00:16<00:34, 8082.49it/s] 31%|███▏      | 125158/400000 [00:16<00:34, 7959.37it/s] 31%|███▏      | 125956/400000 [00:16<00:34, 7867.88it/s] 32%|███▏      | 126759/400000 [00:16<00:34, 7913.82it/s] 32%|███▏      | 127552/400000 [00:16<00:34, 7870.26it/s] 32%|███▏      | 128340/400000 [00:16<00:34, 7762.39it/s] 32%|███▏      | 129118/400000 [00:16<00:35, 7593.03it/s] 32%|███▏      | 129899/400000 [00:16<00:35, 7655.75it/s] 33%|███▎      | 130680/400000 [00:17<00:34, 7699.22it/s] 33%|███▎      | 131515/400000 [00:17<00:34, 7882.81it/s] 33%|███▎      | 132399/400000 [00:17<00:32, 8144.76it/s] 33%|███▎      | 133217/400000 [00:17<00:33, 8055.50it/s] 34%|███▎      | 134046/400000 [00:17<00:32, 8123.36it/s] 34%|███▎      | 134861/400000 [00:17<00:33, 7957.78it/s] 34%|███▍      | 135659/400000 [00:17<00:33, 7841.71it/s] 34%|███▍      | 136446/400000 [00:17<00:33, 7834.30it/s] 34%|███▍      | 137231/400000 [00:17<00:33, 7737.11it/s] 35%|███▍      | 138053/400000 [00:17<00:33, 7871.67it/s] 35%|███▍      | 138853/400000 [00:18<00:33, 7906.65it/s] 35%|███▍      | 139654/400000 [00:18<00:32, 7937.04it/s] 35%|███▌      | 140492/400000 [00:18<00:32, 8062.12it/s] 35%|███▌      | 141300/400000 [00:18<00:32, 7889.25it/s] 36%|███▌      | 142133/400000 [00:18<00:32, 8013.49it/s] 36%|███▌      | 142936/400000 [00:18<00:32, 7968.45it/s] 36%|███▌      | 143734/400000 [00:18<00:32, 7885.73it/s] 36%|███▌      | 144587/400000 [00:18<00:31, 8067.29it/s] 36%|███▋      | 145398/400000 [00:18<00:31, 8079.60it/s] 37%|███▋      | 146254/400000 [00:18<00:30, 8216.11it/s] 37%|███▋      | 147121/400000 [00:19<00:30, 8345.18it/s] 37%|███▋      | 147957/400000 [00:19<00:30, 8336.21it/s] 37%|███▋      | 148814/400000 [00:19<00:29, 8403.01it/s] 37%|███▋      | 149656/400000 [00:19<00:31, 8048.35it/s] 38%|███▊      | 150465/400000 [00:19<00:31, 7969.92it/s] 38%|███▊      | 151265/400000 [00:19<00:31, 7907.32it/s] 38%|███▊      | 152104/400000 [00:19<00:30, 8046.17it/s] 38%|███▊      | 152969/400000 [00:19<00:30, 8216.82it/s] 38%|███▊      | 153794/400000 [00:19<00:30, 8106.06it/s] 39%|███▊      | 154607/400000 [00:20<00:30, 7923.87it/s] 39%|███▉      | 155484/400000 [00:20<00:29, 8158.59it/s] 39%|███▉      | 156304/400000 [00:20<00:30, 8023.08it/s] 39%|███▉      | 157110/400000 [00:20<00:30, 7996.04it/s] 39%|███▉      | 157912/400000 [00:20<00:30, 7902.94it/s] 40%|███▉      | 158704/400000 [00:20<00:30, 7876.57it/s] 40%|███▉      | 159506/400000 [00:20<00:30, 7919.01it/s] 40%|████      | 160299/400000 [00:20<00:30, 7903.54it/s] 40%|████      | 161090/400000 [00:20<00:30, 7797.59it/s] 40%|████      | 161871/400000 [00:20<00:31, 7612.81it/s] 41%|████      | 162634/400000 [00:21<00:31, 7565.34it/s] 41%|████      | 163392/400000 [00:21<00:31, 7562.87it/s] 41%|████      | 164150/400000 [00:21<00:31, 7513.42it/s] 41%|████      | 164927/400000 [00:21<00:30, 7587.94it/s] 41%|████▏     | 165687/400000 [00:21<00:31, 7538.33it/s] 42%|████▏     | 166445/400000 [00:21<00:30, 7548.80it/s] 42%|████▏     | 167201/400000 [00:21<00:30, 7549.40it/s] 42%|████▏     | 167957/400000 [00:21<00:31, 7275.45it/s] 42%|████▏     | 168687/400000 [00:21<00:31, 7237.07it/s] 42%|████▏     | 169457/400000 [00:21<00:31, 7367.44it/s] 43%|████▎     | 170238/400000 [00:22<00:30, 7492.87it/s] 43%|████▎     | 171031/400000 [00:22<00:30, 7616.93it/s] 43%|████▎     | 171795/400000 [00:22<00:30, 7471.62it/s] 43%|████▎     | 172544/400000 [00:22<00:30, 7443.33it/s] 43%|████▎     | 173329/400000 [00:22<00:29, 7560.62it/s] 44%|████▎     | 174087/400000 [00:22<00:30, 7481.80it/s] 44%|████▎     | 174844/400000 [00:22<00:30, 7504.41it/s] 44%|████▍     | 175665/400000 [00:22<00:29, 7700.67it/s] 44%|████▍     | 176489/400000 [00:22<00:28, 7850.16it/s] 44%|████▍     | 177295/400000 [00:22<00:28, 7911.51it/s] 45%|████▍     | 178088/400000 [00:23<00:28, 7880.37it/s] 45%|████▍     | 178930/400000 [00:23<00:27, 8033.01it/s] 45%|████▍     | 179769/400000 [00:23<00:27, 8065.51it/s] 45%|████▌     | 180603/400000 [00:23<00:26, 8144.09it/s] 45%|████▌     | 181419/400000 [00:23<00:27, 8052.91it/s] 46%|████▌     | 182261/400000 [00:23<00:26, 8159.42it/s] 46%|████▌     | 183078/400000 [00:23<00:27, 7988.69it/s] 46%|████▌     | 183879/400000 [00:23<00:27, 7941.82it/s] 46%|████▌     | 184675/400000 [00:23<00:27, 7893.13it/s] 46%|████▋     | 185466/400000 [00:24<00:27, 7695.49it/s] 47%|████▋     | 186238/400000 [00:24<00:27, 7659.48it/s] 47%|████▋     | 187058/400000 [00:24<00:27, 7812.93it/s] 47%|████▋     | 187841/400000 [00:24<00:27, 7803.90it/s] 47%|████▋     | 188623/400000 [00:24<00:27, 7801.52it/s] 47%|████▋     | 189417/400000 [00:24<00:26, 7841.99it/s] 48%|████▊     | 190259/400000 [00:24<00:26, 8005.25it/s] 48%|████▊     | 191087/400000 [00:24<00:25, 8084.09it/s] 48%|████▊     | 191897/400000 [00:24<00:25, 8071.83it/s] 48%|████▊     | 192705/400000 [00:24<00:26, 7966.86it/s] 48%|████▊     | 193503/400000 [00:25<00:26, 7935.47it/s] 49%|████▊     | 194346/400000 [00:25<00:25, 8075.28it/s] 49%|████▉     | 195170/400000 [00:25<00:25, 8122.38it/s] 49%|████▉     | 196005/400000 [00:25<00:24, 8185.87it/s] 49%|████▉     | 196825/400000 [00:25<00:25, 7982.87it/s] 49%|████▉     | 197625/400000 [00:25<00:25, 7788.54it/s] 50%|████▉     | 198463/400000 [00:25<00:25, 7956.26it/s] 50%|████▉     | 199262/400000 [00:25<00:25, 7859.71it/s] 50%|█████     | 200116/400000 [00:25<00:24, 8050.30it/s] 50%|█████     | 200965/400000 [00:25<00:24, 8175.07it/s] 50%|█████     | 201785/400000 [00:26<00:25, 7834.27it/s] 51%|█████     | 202574/400000 [00:26<00:25, 7749.67it/s] 51%|█████     | 203395/400000 [00:26<00:24, 7881.44it/s] 51%|█████     | 204222/400000 [00:26<00:24, 7993.87it/s] 51%|█████▏    | 205024/400000 [00:26<00:24, 7977.21it/s] 51%|█████▏    | 205824/400000 [00:26<00:24, 7788.14it/s] 52%|█████▏    | 206618/400000 [00:26<00:24, 7831.65it/s] 52%|█████▏    | 207403/400000 [00:26<00:24, 7835.23it/s] 52%|█████▏    | 208188/400000 [00:26<00:24, 7787.01it/s] 52%|█████▏    | 208968/400000 [00:26<00:24, 7777.07it/s] 52%|█████▏    | 209752/400000 [00:27<00:24, 7793.93it/s] 53%|█████▎    | 210570/400000 [00:27<00:23, 7903.41it/s] 53%|█████▎    | 211423/400000 [00:27<00:23, 8080.87it/s] 53%|█████▎    | 212242/400000 [00:27<00:23, 8112.61it/s] 53%|█████▎    | 213055/400000 [00:27<00:23, 7797.41it/s] 53%|█████▎    | 213839/400000 [00:27<00:24, 7703.85it/s] 54%|█████▎    | 214612/400000 [00:27<00:24, 7540.31it/s] 54%|█████▍    | 215404/400000 [00:27<00:24, 7648.06it/s] 54%|█████▍    | 216269/400000 [00:27<00:23, 7920.92it/s] 54%|█████▍    | 217109/400000 [00:28<00:22, 8055.81it/s] 54%|█████▍    | 217918/400000 [00:28<00:22, 7952.56it/s] 55%|█████▍    | 218716/400000 [00:28<00:23, 7841.30it/s] 55%|█████▍    | 219572/400000 [00:28<00:22, 8043.49it/s] 55%|█████▌    | 220415/400000 [00:28<00:22, 8154.52it/s] 55%|█████▌    | 221247/400000 [00:28<00:21, 8203.01it/s] 56%|█████▌    | 222070/400000 [00:28<00:22, 8055.31it/s] 56%|█████▌    | 222878/400000 [00:28<00:22, 7823.54it/s] 56%|█████▌    | 223664/400000 [00:28<00:22, 7801.48it/s] 56%|█████▌    | 224500/400000 [00:28<00:22, 7959.42it/s] 56%|█████▋    | 225331/400000 [00:29<00:21, 8060.75it/s] 57%|█████▋    | 226139/400000 [00:29<00:21, 8016.36it/s] 57%|█████▋    | 226942/400000 [00:29<00:21, 7930.31it/s] 57%|█████▋    | 227763/400000 [00:29<00:21, 8009.98it/s] 57%|█████▋    | 228570/400000 [00:29<00:21, 8025.80it/s] 57%|█████▋    | 229374/400000 [00:29<00:21, 7992.10it/s] 58%|█████▊    | 230174/400000 [00:29<00:21, 7963.05it/s] 58%|█████▊    | 230998/400000 [00:29<00:21, 8041.54it/s] 58%|█████▊    | 231839/400000 [00:29<00:20, 8148.02it/s] 58%|█████▊    | 232699/400000 [00:29<00:20, 8275.53it/s] 58%|█████▊    | 233528/400000 [00:30<00:20, 8036.01it/s] 59%|█████▊    | 234334/400000 [00:30<00:21, 7848.17it/s] 59%|█████▉    | 235164/400000 [00:30<00:20, 7977.58it/s] 59%|█████▉    | 236068/400000 [00:30<00:19, 8268.57it/s] 59%|█████▉    | 236945/400000 [00:30<00:19, 8411.35it/s] 59%|█████▉    | 237790/400000 [00:30<00:19, 8314.24it/s] 60%|█████▉    | 238625/400000 [00:30<00:19, 8115.41it/s] 60%|█████▉    | 239440/400000 [00:30<00:19, 8105.13it/s] 60%|██████    | 240336/400000 [00:30<00:19, 8341.68it/s] 60%|██████    | 241198/400000 [00:30<00:18, 8422.87it/s] 61%|██████    | 242043/400000 [00:31<00:19, 8307.19it/s] 61%|██████    | 242876/400000 [00:31<00:19, 8204.22it/s] 61%|██████    | 243699/400000 [00:31<00:19, 8190.24it/s] 61%|██████    | 244532/400000 [00:31<00:18, 8228.07it/s] 61%|██████▏   | 245415/400000 [00:31<00:18, 8399.72it/s] 62%|██████▏   | 246257/400000 [00:31<00:18, 8296.62it/s] 62%|██████▏   | 247168/400000 [00:31<00:17, 8523.54it/s] 62%|██████▏   | 248023/400000 [00:31<00:17, 8508.71it/s] 62%|██████▏   | 248876/400000 [00:31<00:18, 8363.32it/s] 62%|██████▏   | 249715/400000 [00:32<00:18, 8248.61it/s] 63%|██████▎   | 250542/400000 [00:32<00:18, 8032.79it/s] 63%|██████▎   | 251362/400000 [00:32<00:18, 8079.51it/s] 63%|██████▎   | 252172/400000 [00:32<00:18, 8015.15it/s] 63%|██████▎   | 253030/400000 [00:32<00:17, 8174.56it/s] 63%|██████▎   | 253880/400000 [00:32<00:17, 8266.70it/s] 64%|██████▎   | 254709/400000 [00:32<00:18, 8030.32it/s] 64%|██████▍   | 255515/400000 [00:32<00:18, 7935.40it/s] 64%|██████▍   | 256331/400000 [00:32<00:17, 7999.11it/s] 64%|██████▍   | 257165/400000 [00:32<00:17, 8097.90it/s] 65%|██████▍   | 258015/400000 [00:33<00:17, 8213.02it/s] 65%|██████▍   | 258838/400000 [00:33<00:17, 8208.95it/s] 65%|██████▍   | 259660/400000 [00:33<00:17, 8170.56it/s] 65%|██████▌   | 260507/400000 [00:33<00:16, 8255.95it/s] 65%|██████▌   | 261334/400000 [00:33<00:16, 8238.19it/s] 66%|██████▌   | 262159/400000 [00:33<00:16, 8191.79it/s] 66%|██████▌   | 262979/400000 [00:33<00:17, 7960.93it/s] 66%|██████▌   | 263777/400000 [00:33<00:17, 7928.53it/s] 66%|██████▌   | 264619/400000 [00:33<00:16, 8068.20it/s] 66%|██████▋   | 265465/400000 [00:33<00:16, 8179.63it/s] 67%|██████▋   | 266318/400000 [00:34<00:16, 8281.39it/s] 67%|██████▋   | 267148/400000 [00:34<00:16, 8125.90it/s] 67%|██████▋   | 267986/400000 [00:34<00:16, 8200.07it/s] 67%|██████▋   | 268808/400000 [00:34<00:15, 8201.83it/s] 67%|██████▋   | 269630/400000 [00:34<00:15, 8164.51it/s] 68%|██████▊   | 270448/400000 [00:34<00:16, 8064.74it/s] 68%|██████▊   | 271256/400000 [00:34<00:16, 7873.85it/s] 68%|██████▊   | 272053/400000 [00:34<00:16, 7900.72it/s] 68%|██████▊   | 272845/400000 [00:34<00:16, 7870.16it/s] 68%|██████▊   | 273634/400000 [00:34<00:16, 7874.69it/s] 69%|██████▊   | 274427/400000 [00:35<00:15, 7889.98it/s] 69%|██████▉   | 275217/400000 [00:35<00:15, 7818.62it/s] 69%|██████▉   | 276034/400000 [00:35<00:15, 7918.61it/s] 69%|██████▉   | 276891/400000 [00:35<00:15, 8100.95it/s] 69%|██████▉   | 277734/400000 [00:35<00:14, 8194.25it/s] 70%|██████▉   | 278570/400000 [00:35<00:14, 8242.72it/s] 70%|██████▉   | 279417/400000 [00:35<00:14, 8307.63it/s] 70%|███████   | 280249/400000 [00:35<00:14, 8189.32it/s] 70%|███████   | 281069/400000 [00:35<00:14, 7958.20it/s] 70%|███████   | 281929/400000 [00:35<00:14, 8137.64it/s] 71%|███████   | 282746/400000 [00:36<00:14, 8128.40it/s] 71%|███████   | 283561/400000 [00:36<00:14, 7885.81it/s] 71%|███████   | 284404/400000 [00:36<00:14, 8040.24it/s] 71%|███████▏  | 285321/400000 [00:36<00:13, 8347.19it/s] 72%|███████▏  | 286234/400000 [00:36<00:13, 8566.39it/s] 72%|███████▏  | 287096/400000 [00:36<00:13, 8460.43it/s] 72%|███████▏  | 287946/400000 [00:36<00:13, 8349.55it/s] 72%|███████▏  | 288784/400000 [00:36<00:13, 8208.50it/s] 72%|███████▏  | 289629/400000 [00:36<00:13, 8278.88it/s] 73%|███████▎  | 290547/400000 [00:37<00:12, 8528.03it/s] 73%|███████▎  | 291404/400000 [00:37<00:12, 8414.61it/s] 73%|███████▎  | 292257/400000 [00:37<00:12, 8446.64it/s] 73%|███████▎  | 293161/400000 [00:37<00:12, 8615.33it/s] 74%|███████▎  | 294067/400000 [00:37<00:12, 8741.57it/s] 74%|███████▎  | 294944/400000 [00:37<00:12, 8519.90it/s] 74%|███████▍  | 295799/400000 [00:37<00:12, 8256.93it/s] 74%|███████▍  | 296629/400000 [00:37<00:12, 8101.20it/s] 74%|███████▍  | 297446/400000 [00:37<00:12, 8119.36it/s] 75%|███████▍  | 298264/400000 [00:37<00:12, 8137.11it/s] 75%|███████▍  | 299080/400000 [00:38<00:12, 8047.11it/s] 75%|███████▍  | 299887/400000 [00:38<00:12, 7952.01it/s] 75%|███████▌  | 300684/400000 [00:38<00:12, 7835.12it/s] 75%|███████▌  | 301561/400000 [00:38<00:12, 8093.55it/s] 76%|███████▌  | 302424/400000 [00:38<00:11, 8246.64it/s] 76%|███████▌  | 303318/400000 [00:38<00:11, 8443.05it/s] 76%|███████▌  | 304166/400000 [00:38<00:11, 8177.52it/s] 76%|███████▌  | 304993/400000 [00:38<00:11, 8204.93it/s] 76%|███████▋  | 305911/400000 [00:38<00:11, 8474.56it/s] 77%|███████▋  | 306802/400000 [00:38<00:10, 8598.36it/s] 77%|███████▋  | 307707/400000 [00:39<00:10, 8727.89it/s] 77%|███████▋  | 308587/400000 [00:39<00:10, 8746.86it/s] 77%|███████▋  | 309464/400000 [00:39<00:10, 8519.81it/s] 78%|███████▊  | 310319/400000 [00:39<00:10, 8295.66it/s] 78%|███████▊  | 311152/400000 [00:39<00:10, 8287.23it/s] 78%|███████▊  | 312076/400000 [00:39<00:10, 8550.42it/s] 78%|███████▊  | 312935/400000 [00:39<00:10, 8278.92it/s] 78%|███████▊  | 313768/400000 [00:39<00:10, 8186.87it/s] 79%|███████▊  | 314591/400000 [00:39<00:10, 8150.64it/s] 79%|███████▉  | 315440/400000 [00:40<00:10, 8246.65it/s] 79%|███████▉  | 316267/400000 [00:40<00:10, 8161.45it/s] 79%|███████▉  | 317085/400000 [00:40<00:10, 7988.26it/s] 79%|███████▉  | 317894/400000 [00:40<00:10, 8016.51it/s] 80%|███████▉  | 318727/400000 [00:40<00:10, 8107.35it/s] 80%|███████▉  | 319539/400000 [00:40<00:10, 7902.22it/s] 80%|████████  | 320338/400000 [00:40<00:10, 7924.62it/s] 80%|████████  | 321132/400000 [00:40<00:10, 7731.72it/s] 80%|████████  | 321933/400000 [00:40<00:09, 7811.98it/s] 81%|████████  | 322739/400000 [00:40<00:09, 7883.49it/s] 81%|████████  | 323529/400000 [00:41<00:09, 7784.27it/s] 81%|████████  | 324309/400000 [00:41<00:09, 7744.17it/s] 81%|████████▏ | 325098/400000 [00:41<00:09, 7786.77it/s] 81%|████████▏ | 325878/400000 [00:41<00:09, 7766.32it/s] 82%|████████▏ | 326656/400000 [00:41<00:09, 7753.46it/s] 82%|████████▏ | 327480/400000 [00:41<00:09, 7891.43it/s] 82%|████████▏ | 328331/400000 [00:41<00:08, 8066.73it/s] 82%|████████▏ | 329184/400000 [00:41<00:08, 8197.91it/s] 83%|████████▎ | 330006/400000 [00:41<00:08, 8052.04it/s] 83%|████████▎ | 330859/400000 [00:41<00:08, 8188.05it/s] 83%|████████▎ | 331704/400000 [00:42<00:08, 8261.72it/s] 83%|████████▎ | 332553/400000 [00:42<00:08, 8328.78it/s] 83%|████████▎ | 333387/400000 [00:42<00:08, 8161.77it/s] 84%|████████▎ | 334205/400000 [00:42<00:08, 7754.28it/s] 84%|████████▍ | 335006/400000 [00:42<00:08, 7828.02it/s] 84%|████████▍ | 335793/400000 [00:42<00:08, 7766.63it/s] 84%|████████▍ | 336573/400000 [00:42<00:08, 7718.10it/s] 84%|████████▍ | 337347/400000 [00:42<00:08, 7720.02it/s] 85%|████████▍ | 338121/400000 [00:42<00:08, 7667.65it/s] 85%|████████▍ | 338906/400000 [00:42<00:07, 7719.19it/s] 85%|████████▍ | 339679/400000 [00:43<00:07, 7596.09it/s] 85%|████████▌ | 340487/400000 [00:43<00:07, 7733.62it/s] 85%|████████▌ | 341262/400000 [00:43<00:07, 7641.74it/s] 86%|████████▌ | 342034/400000 [00:43<00:07, 7664.38it/s] 86%|████████▌ | 342802/400000 [00:43<00:07, 7522.64it/s] 86%|████████▌ | 343556/400000 [00:43<00:07, 7509.77it/s] 86%|████████▌ | 344329/400000 [00:43<00:07, 7572.52it/s] 86%|████████▋ | 345087/400000 [00:43<00:07, 7521.39it/s] 86%|████████▋ | 345850/400000 [00:43<00:07, 7553.52it/s] 87%|████████▋ | 346606/400000 [00:44<00:07, 7500.06it/s] 87%|████████▋ | 347357/400000 [00:44<00:07, 7488.45it/s] 87%|████████▋ | 348126/400000 [00:44<00:06, 7545.71it/s] 87%|████████▋ | 348881/400000 [00:44<00:06, 7536.58it/s] 87%|████████▋ | 349635/400000 [00:44<00:06, 7279.21it/s] 88%|████████▊ | 350405/400000 [00:44<00:06, 7397.69it/s] 88%|████████▊ | 351147/400000 [00:44<00:06, 7362.40it/s] 88%|████████▊ | 351899/400000 [00:44<00:06, 7408.67it/s] 88%|████████▊ | 352666/400000 [00:44<00:06, 7484.51it/s] 88%|████████▊ | 353435/400000 [00:44<00:06, 7544.53it/s] 89%|████████▊ | 354197/400000 [00:45<00:06, 7565.16it/s] 89%|████████▊ | 354955/400000 [00:45<00:06, 7479.76it/s] 89%|████████▉ | 355720/400000 [00:45<00:05, 7528.29it/s] 89%|████████▉ | 356474/400000 [00:45<00:05, 7512.38it/s] 89%|████████▉ | 357261/400000 [00:45<00:05, 7613.96it/s] 90%|████████▉ | 358035/400000 [00:45<00:05, 7648.89it/s] 90%|████████▉ | 358846/400000 [00:45<00:05, 7779.16it/s] 90%|████████▉ | 359653/400000 [00:45<00:05, 7863.58it/s] 90%|█████████ | 360441/400000 [00:45<00:05, 7798.90it/s] 90%|█████████ | 361235/400000 [00:45<00:04, 7836.43it/s] 91%|█████████ | 362020/400000 [00:46<00:04, 7818.41it/s] 91%|█████████ | 362803/400000 [00:46<00:04, 7737.23it/s] 91%|█████████ | 363578/400000 [00:46<00:04, 7440.81it/s] 91%|█████████ | 364340/400000 [00:46<00:04, 7492.08it/s] 91%|█████████▏| 365111/400000 [00:46<00:04, 7554.30it/s] 91%|█████████▏| 365871/400000 [00:46<00:04, 7565.80it/s] 92%|█████████▏| 366641/400000 [00:46<00:04, 7605.30it/s] 92%|█████████▏| 367439/400000 [00:46<00:04, 7712.60it/s] 92%|█████████▏| 368212/400000 [00:46<00:04, 7583.36it/s] 92%|█████████▏| 368990/400000 [00:46<00:04, 7638.78it/s] 92%|█████████▏| 369755/400000 [00:47<00:03, 7631.73it/s] 93%|█████████▎| 370519/400000 [00:47<00:03, 7560.55it/s] 93%|█████████▎| 371294/400000 [00:47<00:03, 7614.24it/s] 93%|█████████▎| 372056/400000 [00:47<00:03, 7542.43it/s] 93%|█████████▎| 372817/400000 [00:47<00:03, 7560.70it/s] 93%|█████████▎| 373605/400000 [00:47<00:03, 7651.35it/s] 94%|█████████▎| 374397/400000 [00:47<00:03, 7727.90it/s] 94%|█████████▍| 375171/400000 [00:47<00:03, 7695.53it/s] 94%|█████████▍| 375955/400000 [00:47<00:03, 7737.68it/s] 94%|█████████▍| 376768/400000 [00:47<00:02, 7849.40it/s] 94%|█████████▍| 377567/400000 [00:48<00:02, 7890.01it/s] 95%|█████████▍| 378389/400000 [00:48<00:02, 7984.40it/s] 95%|█████████▍| 379213/400000 [00:48<00:02, 8058.83it/s] 95%|█████████▌| 380020/400000 [00:48<00:02, 7896.32it/s] 95%|█████████▌| 380811/400000 [00:48<00:02, 7849.03it/s] 95%|█████████▌| 381603/400000 [00:48<00:02, 7868.77it/s] 96%|█████████▌| 382413/400000 [00:48<00:02, 7935.57it/s] 96%|█████████▌| 383227/400000 [00:48<00:02, 7994.57it/s] 96%|█████████▌| 384027/400000 [00:48<00:02, 7859.73it/s] 96%|█████████▌| 384817/400000 [00:48<00:01, 7869.56it/s] 96%|█████████▋| 385605/400000 [00:49<00:01, 7816.81it/s] 97%|█████████▋| 386445/400000 [00:49<00:01, 7981.50it/s] 97%|█████████▋| 387253/400000 [00:49<00:01, 8008.00it/s] 97%|█████████▋| 388055/400000 [00:49<00:01, 7923.39it/s] 97%|█████████▋| 388869/400000 [00:49<00:01, 7983.73it/s] 97%|█████████▋| 389669/400000 [00:49<00:01, 7966.03it/s] 98%|█████████▊| 390467/400000 [00:49<00:01, 7823.73it/s] 98%|█████████▊| 391267/400000 [00:49<00:01, 7874.54it/s] 98%|█████████▊| 392056/400000 [00:49<00:01, 7829.37it/s] 98%|█████████▊| 392904/400000 [00:50<00:00, 8011.81it/s] 98%|█████████▊| 393707/400000 [00:50<00:00, 7906.72it/s] 99%|█████████▊| 394505/400000 [00:50<00:00, 7928.13it/s] 99%|█████████▉| 395300/400000 [00:50<00:00, 7933.86it/s] 99%|█████████▉| 396095/400000 [00:50<00:00, 7790.48it/s] 99%|█████████▉| 396876/400000 [00:50<00:00, 7763.19it/s] 99%|█████████▉| 397654/400000 [00:50<00:00, 7742.58it/s]100%|█████████▉| 398468/400000 [00:50<00:00, 7856.79it/s]100%|█████████▉| 399291/400000 [00:50<00:00, 7962.13it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7857.15it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5e450520f0>, <torchtext.data.dataset.TabularDataset object at 0x7f5e45052240>, <torchtext.vocab.Vocab object at 0x7f5e45052160>), {}) 

