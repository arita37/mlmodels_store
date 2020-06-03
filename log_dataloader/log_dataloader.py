
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f94b2715378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f94b2715378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f95247660d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f95247660d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f94bbefa9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f94bbefa9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f94d204a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f94d204a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f94d204a8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138239.92it/s] 84%|████████▍ | 8331264/9912422 [00:00<00:08, 197344.95it/s]9920512it [00:00, 43151623.27it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 262090.51it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8870292.93it/s]          
0it [00:00, ?it/s]8192it [00:00, 194484.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94b361cfd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94b3624f28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f94d204a510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f94d204a510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f94d204a510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:18,  2.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:18,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:17,  2.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:56,  2.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:56,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:21,  6.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:21,  6.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:15,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:15,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:15,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:15,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:01<00:11, 12.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:11, 12.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:08, 15.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:08, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 20.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:06, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 20.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 26.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 33.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 33.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 40.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 40.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:01, 48.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 48.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 55.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 55.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 61.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 61.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:00, 66.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 66.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 70.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 70.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 73.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 76.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 76.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 78.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 78.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 78.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.23s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 80.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.23s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.23s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.99 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.23s/ url]
0 examples [00:00, ? examples/s]2020-06-03 06:09:52.274116: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-03 06:09:52.287181: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-03 06:09:52.287378: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55811889cd30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-03 06:09:52.287397: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
36 examples [00:00, 357.06 examples/s]125 examples [00:00, 435.06 examples/s]216 examples [00:00, 515.64 examples/s]304 examples [00:00, 587.44 examples/s]392 examples [00:00, 651.08 examples/s]473 examples [00:00, 690.15 examples/s]558 examples [00:00, 730.31 examples/s]645 examples [00:00, 766.25 examples/s]738 examples [00:00, 806.73 examples/s]825 examples [00:01, 822.97 examples/s]913 examples [00:01, 837.57 examples/s]1000 examples [00:01, 840.69 examples/s]1088 examples [00:01, 851.60 examples/s]1175 examples [00:01, 856.63 examples/s]1262 examples [00:01, 841.58 examples/s]1356 examples [00:01, 867.19 examples/s]1445 examples [00:01, 872.82 examples/s]1538 examples [00:01, 887.53 examples/s]1628 examples [00:01, 889.51 examples/s]1718 examples [00:02, 887.11 examples/s]1807 examples [00:02, 881.06 examples/s]1897 examples [00:02, 885.01 examples/s]1987 examples [00:02, 889.44 examples/s]2080 examples [00:02, 898.67 examples/s]2170 examples [00:02, 887.43 examples/s]2259 examples [00:02, 881.57 examples/s]2348 examples [00:02, 877.15 examples/s]2437 examples [00:02, 878.74 examples/s]2525 examples [00:02, 876.66 examples/s]2617 examples [00:03, 888.00 examples/s]2706 examples [00:03, 878.32 examples/s]2794 examples [00:03, 874.59 examples/s]2886 examples [00:03, 886.62 examples/s]2976 examples [00:03, 889.48 examples/s]3067 examples [00:03, 894.38 examples/s]3158 examples [00:03, 897.22 examples/s]3251 examples [00:03, 904.13 examples/s]3342 examples [00:03, 900.38 examples/s]3433 examples [00:03, 902.50 examples/s]3524 examples [00:04, 872.93 examples/s]3612 examples [00:04, 850.39 examples/s]3702 examples [00:04, 863.50 examples/s]3793 examples [00:04, 874.20 examples/s]3881 examples [00:04, 851.46 examples/s]3970 examples [00:04, 861.03 examples/s]4057 examples [00:04, 861.45 examples/s]4146 examples [00:04, 868.22 examples/s]4236 examples [00:04, 875.24 examples/s]4327 examples [00:04, 884.55 examples/s]4420 examples [00:05, 896.50 examples/s]4510 examples [00:05, 895.48 examples/s]4600 examples [00:05, 882.84 examples/s]4689 examples [00:05, 878.25 examples/s]4777 examples [00:05, 876.79 examples/s]4865 examples [00:05, 870.15 examples/s]4953 examples [00:05, 866.27 examples/s]5040 examples [00:05, 836.89 examples/s]5131 examples [00:05, 856.78 examples/s]5223 examples [00:06, 871.98 examples/s]5313 examples [00:06, 879.01 examples/s]5406 examples [00:06, 891.75 examples/s]5496 examples [00:06, 891.60 examples/s]5586 examples [00:06, 884.93 examples/s]5675 examples [00:06, 882.29 examples/s]5764 examples [00:06, 871.61 examples/s]5855 examples [00:06, 880.44 examples/s]5946 examples [00:06, 888.56 examples/s]6035 examples [00:06, 884.74 examples/s]6124 examples [00:07, 859.61 examples/s]6211 examples [00:07, 859.20 examples/s]6300 examples [00:07, 866.21 examples/s]6391 examples [00:07, 877.44 examples/s]6479 examples [00:07, 872.22 examples/s]6567 examples [00:07, 861.02 examples/s]6654 examples [00:07, 851.96 examples/s]6743 examples [00:07, 861.83 examples/s]6832 examples [00:07, 867.80 examples/s]6919 examples [00:07, 846.87 examples/s]7006 examples [00:08, 852.19 examples/s]7092 examples [00:08, 848.15 examples/s]7177 examples [00:08, 847.40 examples/s]7263 examples [00:08, 849.95 examples/s]7355 examples [00:08, 868.48 examples/s]7445 examples [00:08, 876.05 examples/s]7533 examples [00:08, 866.60 examples/s]7624 examples [00:08, 878.20 examples/s]7712 examples [00:08, 868.19 examples/s]7799 examples [00:08, 857.45 examples/s]7891 examples [00:09, 873.48 examples/s]7979 examples [00:09, 843.83 examples/s]8064 examples [00:09, 835.42 examples/s]8149 examples [00:09, 837.35 examples/s]8243 examples [00:09, 865.07 examples/s]8334 examples [00:09, 875.47 examples/s]8423 examples [00:09, 878.02 examples/s]8515 examples [00:09, 888.56 examples/s]8608 examples [00:09, 899.37 examples/s]8699 examples [00:10, 884.93 examples/s]8793 examples [00:10, 898.71 examples/s]8884 examples [00:10, 867.55 examples/s]8975 examples [00:10, 878.20 examples/s]9064 examples [00:10, 870.38 examples/s]9153 examples [00:10, 875.99 examples/s]9241 examples [00:10, 846.36 examples/s]9326 examples [00:10, 840.00 examples/s]9415 examples [00:10, 852.74 examples/s]9501 examples [00:10, 825.13 examples/s]9584 examples [00:11, 817.48 examples/s]9671 examples [00:11, 831.24 examples/s]9761 examples [00:11, 849.50 examples/s]9853 examples [00:11, 867.97 examples/s]9942 examples [00:11, 872.17 examples/s]10030 examples [00:11, 839.51 examples/s]10115 examples [00:11, 832.47 examples/s]10199 examples [00:11, 830.14 examples/s]10286 examples [00:11, 839.29 examples/s]10371 examples [00:11, 833.37 examples/s]10455 examples [00:12, 833.06 examples/s]10539 examples [00:12, 817.71 examples/s]10624 examples [00:12, 826.90 examples/s]10710 examples [00:12, 836.14 examples/s]10799 examples [00:12, 851.49 examples/s]10885 examples [00:12, 845.56 examples/s]10973 examples [00:12, 854.94 examples/s]11062 examples [00:12, 863.45 examples/s]11151 examples [00:12, 869.86 examples/s]11241 examples [00:13, 876.56 examples/s]11329 examples [00:13, 872.84 examples/s]11417 examples [00:13, 866.87 examples/s]11504 examples [00:13, 840.16 examples/s]11592 examples [00:13, 848.98 examples/s]11678 examples [00:13, 850.83 examples/s]11764 examples [00:13, 853.32 examples/s]11850 examples [00:13, 844.31 examples/s]11942 examples [00:13, 864.06 examples/s]12034 examples [00:13, 878.94 examples/s]12123 examples [00:14, 865.65 examples/s]12210 examples [00:14, 845.52 examples/s]12301 examples [00:14, 862.80 examples/s]12388 examples [00:14, 860.01 examples/s]12476 examples [00:14, 865.91 examples/s]12567 examples [00:14, 876.81 examples/s]12657 examples [00:14, 882.92 examples/s]12746 examples [00:14, 883.58 examples/s]12835 examples [00:14, 884.09 examples/s]12929 examples [00:14, 898.79 examples/s]13020 examples [00:15, 901.35 examples/s]13111 examples [00:15, 890.49 examples/s]13201 examples [00:15, 889.65 examples/s]13291 examples [00:15, 891.05 examples/s]13382 examples [00:15, 894.49 examples/s]13473 examples [00:15, 898.59 examples/s]13563 examples [00:15, 891.25 examples/s]13653 examples [00:15, 884.64 examples/s]13743 examples [00:15, 888.52 examples/s]13832 examples [00:15, 876.21 examples/s]13920 examples [00:16, 850.47 examples/s]14006 examples [00:16, 846.55 examples/s]14093 examples [00:16, 852.44 examples/s]14180 examples [00:16, 854.99 examples/s]14267 examples [00:16, 858.96 examples/s]14353 examples [00:16, 859.10 examples/s]14440 examples [00:16, 860.01 examples/s]14530 examples [00:16, 870.43 examples/s]14619 examples [00:16, 874.21 examples/s]14710 examples [00:16, 881.89 examples/s]14804 examples [00:17, 898.20 examples/s]14896 examples [00:17, 901.56 examples/s]14987 examples [00:17, 886.30 examples/s]15078 examples [00:17, 891.55 examples/s]15168 examples [00:17, 892.94 examples/s]15258 examples [00:17, 888.88 examples/s]15348 examples [00:17, 890.31 examples/s]15439 examples [00:17, 895.30 examples/s]15531 examples [00:17, 902.30 examples/s]15622 examples [00:18, 890.38 examples/s]15714 examples [00:18, 897.35 examples/s]15805 examples [00:18, 900.77 examples/s]15896 examples [00:18, 899.61 examples/s]15986 examples [00:18, 894.18 examples/s]16076 examples [00:18, 895.72 examples/s]16166 examples [00:18, 895.85 examples/s]16259 examples [00:18, 905.48 examples/s]16352 examples [00:18, 912.22 examples/s]16444 examples [00:18, 899.46 examples/s]16535 examples [00:19, 876.82 examples/s]16627 examples [00:19, 889.07 examples/s]16717 examples [00:19, 883.35 examples/s]16808 examples [00:19, 889.50 examples/s]16898 examples [00:19, 886.84 examples/s]16990 examples [00:19, 894.10 examples/s]17080 examples [00:19, 888.92 examples/s]17170 examples [00:19, 890.02 examples/s]17260 examples [00:19, 876.13 examples/s]17354 examples [00:19, 893.20 examples/s]17444 examples [00:20, 886.74 examples/s]17535 examples [00:20, 892.81 examples/s]17625 examples [00:20, 890.91 examples/s]17718 examples [00:20, 899.88 examples/s]17809 examples [00:20, 900.63 examples/s]17903 examples [00:20, 909.36 examples/s]17994 examples [00:20, 886.38 examples/s]18087 examples [00:20, 897.69 examples/s]18177 examples [00:20, 890.42 examples/s]18267 examples [00:20, 887.51 examples/s]18357 examples [00:21, 890.88 examples/s]18447 examples [00:21, 864.07 examples/s]18534 examples [00:21, 854.85 examples/s]18620 examples [00:21, 855.95 examples/s]18709 examples [00:21, 865.46 examples/s]18797 examples [00:21, 866.97 examples/s]18884 examples [00:21, 867.52 examples/s]18973 examples [00:21, 871.80 examples/s]19063 examples [00:21, 879.85 examples/s]19152 examples [00:21, 874.61 examples/s]19243 examples [00:22, 883.03 examples/s]19334 examples [00:22, 890.19 examples/s]19424 examples [00:22, 880.29 examples/s]19513 examples [00:22, 857.03 examples/s]19602 examples [00:22, 865.25 examples/s]19695 examples [00:22, 883.67 examples/s]19786 examples [00:22, 890.20 examples/s]19876 examples [00:22, 878.22 examples/s]19964 examples [00:22, 862.64 examples/s]20051 examples [00:23, 825.03 examples/s]20138 examples [00:23, 836.89 examples/s]20227 examples [00:23, 851.59 examples/s]20316 examples [00:23, 862.15 examples/s]20408 examples [00:23, 876.72 examples/s]20499 examples [00:23, 886.18 examples/s]20592 examples [00:23, 897.81 examples/s]20682 examples [00:23, 894.45 examples/s]20772 examples [00:23, 885.87 examples/s]20861 examples [00:23, 877.09 examples/s]20952 examples [00:24, 884.19 examples/s]21047 examples [00:24, 900.79 examples/s]21140 examples [00:24, 908.04 examples/s]21233 examples [00:24, 911.97 examples/s]21325 examples [00:24, 908.44 examples/s]21416 examples [00:24, 901.59 examples/s]21507 examples [00:24, 901.91 examples/s]21598 examples [00:24, 899.94 examples/s]21689 examples [00:24, 886.43 examples/s]21778 examples [00:24, 868.79 examples/s]21866 examples [00:25, 863.95 examples/s]21953 examples [00:25, 850.77 examples/s]22041 examples [00:25, 856.81 examples/s]22132 examples [00:25, 869.94 examples/s]22224 examples [00:25, 883.54 examples/s]22317 examples [00:25, 895.08 examples/s]22407 examples [00:25, 896.38 examples/s]22497 examples [00:25, 897.44 examples/s]22589 examples [00:25, 903.22 examples/s]22680 examples [00:25, 878.96 examples/s]22769 examples [00:26, 869.96 examples/s]22857 examples [00:26, 871.63 examples/s]22945 examples [00:26, 868.47 examples/s]23032 examples [00:26, 847.72 examples/s]23117 examples [00:26, 809.05 examples/s]23206 examples [00:26, 831.13 examples/s]23298 examples [00:26, 853.76 examples/s]23388 examples [00:26, 866.84 examples/s]23480 examples [00:26, 880.04 examples/s]23570 examples [00:27, 883.66 examples/s]23659 examples [00:27, 876.20 examples/s]23747 examples [00:27, 872.10 examples/s]23835 examples [00:27, 849.87 examples/s]23921 examples [00:27, 839.28 examples/s]24006 examples [00:27, 839.27 examples/s]24093 examples [00:27, 846.93 examples/s]24183 examples [00:27, 861.09 examples/s]24271 examples [00:27, 865.86 examples/s]24361 examples [00:27, 872.80 examples/s]24451 examples [00:28, 874.69 examples/s]24539 examples [00:28, 874.08 examples/s]24627 examples [00:28, 863.02 examples/s]24714 examples [00:28, 863.72 examples/s]24801 examples [00:28, 863.60 examples/s]24888 examples [00:28, 848.93 examples/s]24977 examples [00:28, 858.61 examples/s]25070 examples [00:28, 876.38 examples/s]25158 examples [00:28, 873.96 examples/s]25246 examples [00:28, 870.91 examples/s]25334 examples [00:29, 836.62 examples/s]25419 examples [00:29, 814.74 examples/s]25502 examples [00:29, 817.09 examples/s]25586 examples [00:29, 822.48 examples/s]25669 examples [00:29, 804.29 examples/s]25756 examples [00:29, 820.25 examples/s]25844 examples [00:29, 835.29 examples/s]25929 examples [00:29, 839.36 examples/s]26019 examples [00:29, 853.91 examples/s]26107 examples [00:30, 860.53 examples/s]26197 examples [00:30, 869.58 examples/s]26287 examples [00:30, 877.57 examples/s]26375 examples [00:30, 873.70 examples/s]26466 examples [00:30, 882.44 examples/s]26559 examples [00:30, 894.93 examples/s]26649 examples [00:30, 895.21 examples/s]26739 examples [00:30, 887.54 examples/s]26833 examples [00:30, 899.93 examples/s]26924 examples [00:30, 901.66 examples/s]27015 examples [00:31, 894.40 examples/s]27105 examples [00:31, 893.83 examples/s]27195 examples [00:31, 878.09 examples/s]27283 examples [00:31, 869.20 examples/s]27374 examples [00:31, 879.55 examples/s]27463 examples [00:31, 880.85 examples/s]27552 examples [00:31, 870.38 examples/s]27640 examples [00:31, 843.00 examples/s]27725 examples [00:31, 837.53 examples/s]27810 examples [00:31, 840.99 examples/s]27901 examples [00:32, 859.06 examples/s]27988 examples [00:32, 852.41 examples/s]28074 examples [00:32, 845.22 examples/s]28159 examples [00:32, 831.70 examples/s]28249 examples [00:32, 850.44 examples/s]28335 examples [00:32, 845.83 examples/s]28424 examples [00:32, 856.80 examples/s]28517 examples [00:32, 876.40 examples/s]28605 examples [00:32, 876.86 examples/s]28693 examples [00:32, 870.59 examples/s]28783 examples [00:33, 879.10 examples/s]28876 examples [00:33, 892.93 examples/s]28972 examples [00:33, 909.90 examples/s]29065 examples [00:33, 914.16 examples/s]29157 examples [00:33, 912.76 examples/s]29249 examples [00:33, 913.51 examples/s]29344 examples [00:33, 923.83 examples/s]29437 examples [00:33, 917.63 examples/s]29529 examples [00:33, 899.98 examples/s]29620 examples [00:33, 885.84 examples/s]29709 examples [00:34, 877.86 examples/s]29797 examples [00:34, 877.36 examples/s]29888 examples [00:34, 884.48 examples/s]29977 examples [00:34, 874.71 examples/s]30065 examples [00:34, 847.06 examples/s]30150 examples [00:34, 839.97 examples/s]30238 examples [00:34, 849.99 examples/s]30331 examples [00:34, 872.43 examples/s]30422 examples [00:34, 881.48 examples/s]30511 examples [00:35, 877.03 examples/s]30599 examples [00:35, 874.18 examples/s]30688 examples [00:35, 878.08 examples/s]30776 examples [00:35, 877.39 examples/s]30864 examples [00:35, 862.66 examples/s]30955 examples [00:35, 874.70 examples/s]31043 examples [00:35, 843.18 examples/s]31130 examples [00:35, 849.36 examples/s]31219 examples [00:35, 860.86 examples/s]31309 examples [00:35, 870.26 examples/s]31399 examples [00:36, 876.84 examples/s]31488 examples [00:36, 879.00 examples/s]31581 examples [00:36, 893.59 examples/s]31674 examples [00:36, 903.57 examples/s]31770 examples [00:36, 918.46 examples/s]31866 examples [00:36, 930.15 examples/s]31960 examples [00:36, 920.11 examples/s]32053 examples [00:36, 904.40 examples/s]32144 examples [00:36, 870.25 examples/s]32232 examples [00:36, 853.59 examples/s]32321 examples [00:37, 862.73 examples/s]32409 examples [00:37, 867.01 examples/s]32499 examples [00:37, 874.34 examples/s]32587 examples [00:37, 869.00 examples/s]32679 examples [00:37, 882.73 examples/s]32773 examples [00:37, 896.91 examples/s]32863 examples [00:37, 893.81 examples/s]32955 examples [00:37, 898.95 examples/s]33049 examples [00:37, 910.57 examples/s]33143 examples [00:37, 917.96 examples/s]33239 examples [00:38, 928.73 examples/s]33332 examples [00:38, 917.96 examples/s]33425 examples [00:38, 920.03 examples/s]33520 examples [00:38, 928.71 examples/s]33613 examples [00:38, 907.65 examples/s]33704 examples [00:38, 896.99 examples/s]33794 examples [00:38, 895.77 examples/s]33889 examples [00:38, 908.87 examples/s]33980 examples [00:38, 908.24 examples/s]34071 examples [00:39, 898.49 examples/s]34163 examples [00:39, 902.68 examples/s]34254 examples [00:39, 898.60 examples/s]34344 examples [00:39, 895.35 examples/s]34435 examples [00:39, 897.44 examples/s]34525 examples [00:39, 881.91 examples/s]34616 examples [00:39, 887.75 examples/s]34705 examples [00:39, 877.77 examples/s]34793 examples [00:39, 869.35 examples/s]34884 examples [00:39, 880.64 examples/s]34973 examples [00:40, 881.83 examples/s]35062 examples [00:40, 881.62 examples/s]35151 examples [00:40, 876.23 examples/s]35239 examples [00:40, 877.17 examples/s]35327 examples [00:40, 874.69 examples/s]35416 examples [00:40, 878.64 examples/s]35505 examples [00:40, 880.69 examples/s]35594 examples [00:40, 878.14 examples/s]35683 examples [00:40, 880.99 examples/s]35773 examples [00:40, 885.13 examples/s]35862 examples [00:41, 885.58 examples/s]35951 examples [00:41, 865.25 examples/s]36038 examples [00:41, 856.33 examples/s]36124 examples [00:41, 844.03 examples/s]36214 examples [00:41, 858.99 examples/s]36301 examples [00:41, 850.58 examples/s]36393 examples [00:41, 868.94 examples/s]36481 examples [00:41, 864.21 examples/s]36578 examples [00:41, 891.87 examples/s]36669 examples [00:41, 896.07 examples/s]36759 examples [00:42, 863.59 examples/s]36846 examples [00:42, 860.32 examples/s]36933 examples [00:42, 849.76 examples/s]37021 examples [00:42, 857.91 examples/s]37108 examples [00:42, 860.40 examples/s]37198 examples [00:42, 869.64 examples/s]37287 examples [00:42, 874.75 examples/s]37376 examples [00:42, 878.13 examples/s]37473 examples [00:42, 903.51 examples/s]37564 examples [00:42, 903.18 examples/s]37655 examples [00:43, 901.51 examples/s]37747 examples [00:43, 906.06 examples/s]37838 examples [00:43, 896.79 examples/s]37930 examples [00:43, 900.63 examples/s]38021 examples [00:43, 899.29 examples/s]38111 examples [00:43, 888.19 examples/s]38200 examples [00:43, 883.50 examples/s]38289 examples [00:43, 884.74 examples/s]38378 examples [00:43, 871.17 examples/s]38466 examples [00:44, 851.32 examples/s]38553 examples [00:44, 856.00 examples/s]38643 examples [00:44, 866.31 examples/s]38737 examples [00:44, 886.52 examples/s]38831 examples [00:44, 899.66 examples/s]38925 examples [00:44, 911.16 examples/s]39017 examples [00:44, 908.22 examples/s]39108 examples [00:44, 884.27 examples/s]39197 examples [00:44, 849.22 examples/s]39283 examples [00:44, 852.15 examples/s]39374 examples [00:45, 867.20 examples/s]39463 examples [00:45, 873.84 examples/s]39553 examples [00:45, 878.86 examples/s]39642 examples [00:45, 877.81 examples/s]39733 examples [00:45, 885.01 examples/s]39822 examples [00:45, 859.00 examples/s]39909 examples [00:45, 847.71 examples/s]39995 examples [00:45, 849.71 examples/s]40081 examples [00:45, 807.98 examples/s]40167 examples [00:45, 822.26 examples/s]40259 examples [00:46, 848.64 examples/s]40352 examples [00:46, 869.11 examples/s]40441 examples [00:46, 873.30 examples/s]40529 examples [00:46, 869.59 examples/s]40617 examples [00:46, 870.35 examples/s]40708 examples [00:46, 879.15 examples/s]40799 examples [00:46, 886.02 examples/s]40891 examples [00:46, 895.41 examples/s]40981 examples [00:46, 885.98 examples/s]41070 examples [00:47, 886.43 examples/s]41165 examples [00:47, 903.12 examples/s]41256 examples [00:47, 876.05 examples/s]41344 examples [00:47, 875.19 examples/s]41432 examples [00:47, 866.62 examples/s]41521 examples [00:47, 872.58 examples/s]41611 examples [00:47, 878.75 examples/s]41700 examples [00:47, 880.52 examples/s]41793 examples [00:47, 893.67 examples/s]41885 examples [00:47, 899.07 examples/s]41975 examples [00:48, 898.58 examples/s]42069 examples [00:48, 908.19 examples/s]42160 examples [00:48, 900.04 examples/s]42251 examples [00:48, 897.38 examples/s]42342 examples [00:48, 901.12 examples/s]42433 examples [00:48, 894.29 examples/s]42523 examples [00:48, 893.18 examples/s]42615 examples [00:48, 900.74 examples/s]42706 examples [00:48, 878.08 examples/s]42794 examples [00:48, 844.68 examples/s]42887 examples [00:49, 868.50 examples/s]42980 examples [00:49, 885.31 examples/s]43069 examples [00:49, 874.70 examples/s]43157 examples [00:49, 872.98 examples/s]43249 examples [00:49, 884.47 examples/s]43343 examples [00:49, 898.27 examples/s]43434 examples [00:49, 897.36 examples/s]43524 examples [00:49, 897.63 examples/s]43618 examples [00:49, 908.59 examples/s]43709 examples [00:49, 897.20 examples/s]43799 examples [00:50, 888.31 examples/s]43891 examples [00:50, 895.00 examples/s]43981 examples [00:50, 891.25 examples/s]44076 examples [00:50, 906.07 examples/s]44167 examples [00:50, 897.02 examples/s]44257 examples [00:50, 891.94 examples/s]44347 examples [00:50, 860.11 examples/s]44438 examples [00:50, 872.45 examples/s]44526 examples [00:50, 872.91 examples/s]44614 examples [00:51, 854.97 examples/s]44704 examples [00:51, 866.28 examples/s]44794 examples [00:51, 875.73 examples/s]44886 examples [00:51, 885.96 examples/s]44975 examples [00:51, 877.19 examples/s]45063 examples [00:51, 873.91 examples/s]45153 examples [00:51, 879.15 examples/s]45243 examples [00:51, 883.59 examples/s]45332 examples [00:51, 875.73 examples/s]45424 examples [00:51, 884.86 examples/s]45515 examples [00:52, 889.89 examples/s]45605 examples [00:52, 884.27 examples/s]45694 examples [00:52, 875.78 examples/s]45782 examples [00:52, 866.22 examples/s]45869 examples [00:52, 844.47 examples/s]45957 examples [00:52, 854.82 examples/s]46043 examples [00:52, 837.61 examples/s]46127 examples [00:52, 833.41 examples/s]46212 examples [00:52, 835.89 examples/s]46302 examples [00:52, 852.62 examples/s]46394 examples [00:53, 869.54 examples/s]46484 examples [00:53, 876.86 examples/s]46574 examples [00:53, 882.01 examples/s]46663 examples [00:53, 879.50 examples/s]46752 examples [00:53, 867.08 examples/s]46844 examples [00:53, 880.46 examples/s]46934 examples [00:53, 883.72 examples/s]47023 examples [00:53, 878.47 examples/s]47111 examples [00:53, 855.82 examples/s]47197 examples [00:53, 852.12 examples/s]47293 examples [00:54, 880.39 examples/s]47389 examples [00:54, 900.56 examples/s]47483 examples [00:54, 910.07 examples/s]47575 examples [00:54, 908.45 examples/s]47667 examples [00:54, 889.76 examples/s]47757 examples [00:54, 881.42 examples/s]47846 examples [00:54, 876.72 examples/s]47936 examples [00:54, 882.17 examples/s]48025 examples [00:54, 882.43 examples/s]48114 examples [00:55, 874.37 examples/s]48203 examples [00:55, 877.22 examples/s]48291 examples [00:55, 856.05 examples/s]48381 examples [00:55, 868.43 examples/s]48472 examples [00:55, 878.80 examples/s]48561 examples [00:55, 875.47 examples/s]48649 examples [00:55, 874.42 examples/s]48738 examples [00:55, 876.23 examples/s]48826 examples [00:55, 876.95 examples/s]48915 examples [00:55, 878.02 examples/s]49003 examples [00:56, 878.28 examples/s]49093 examples [00:56, 883.12 examples/s]49182 examples [00:56, 874.51 examples/s]49274 examples [00:56, 887.63 examples/s]49363 examples [00:56, 886.96 examples/s]49452 examples [00:56, 880.67 examples/s]49541 examples [00:56, 876.70 examples/s]49632 examples [00:56, 886.07 examples/s]49722 examples [00:56, 887.17 examples/s]49811 examples [00:56, 882.56 examples/s]49900 examples [00:57, 875.78 examples/s]49993 examples [00:57, 890.98 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6028/50000 [00:00<00:00, 60279.48 examples/s] 35%|███▌      | 17651/50000 [00:00<00:00, 70453.68 examples/s] 59%|█████▊    | 29344/50000 [00:00<00:00, 79991.51 examples/s] 81%|████████  | 40270/50000 [00:00<00:00, 86979.68 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.98 examples/s]160 examples [00:00, 748.90 examples/s]249 examples [00:00, 785.48 examples/s]338 examples [00:00, 812.80 examples/s]430 examples [00:00, 842.15 examples/s]524 examples [00:00, 866.79 examples/s]614 examples [00:00, 874.87 examples/s]697 examples [00:00, 682.49 examples/s]782 examples [00:00, 723.64 examples/s]869 examples [00:01, 761.88 examples/s]956 examples [00:01, 789.51 examples/s]1044 examples [00:01, 814.29 examples/s]1136 examples [00:01, 842.38 examples/s]1233 examples [00:01, 876.55 examples/s]1322 examples [00:01, 878.45 examples/s]1411 examples [00:01, 880.68 examples/s]1500 examples [00:01, 856.18 examples/s]1587 examples [00:01, 853.84 examples/s]1683 examples [00:02, 881.34 examples/s]1777 examples [00:02, 896.74 examples/s]1868 examples [00:02, 896.27 examples/s]1958 examples [00:02, 862.32 examples/s]2045 examples [00:02, 857.27 examples/s]2132 examples [00:02, 857.27 examples/s]2220 examples [00:02, 863.09 examples/s]2314 examples [00:02, 883.74 examples/s]2403 examples [00:02, 879.51 examples/s]2492 examples [00:02, 873.36 examples/s]2583 examples [00:03, 881.91 examples/s]2675 examples [00:03, 892.10 examples/s]2765 examples [00:03, 892.13 examples/s]2855 examples [00:03, 892.67 examples/s]2945 examples [00:03, 887.58 examples/s]3034 examples [00:03, 884.93 examples/s]3124 examples [00:03, 886.83 examples/s]3213 examples [00:03, 883.08 examples/s]3302 examples [00:03, 877.64 examples/s]3396 examples [00:03, 894.13 examples/s]3486 examples [00:04, 881.08 examples/s]3579 examples [00:04, 894.83 examples/s]3669 examples [00:04, 880.43 examples/s]3761 examples [00:04, 885.95 examples/s]3852 examples [00:04, 890.10 examples/s]3942 examples [00:04, 862.41 examples/s]4033 examples [00:04, 874.23 examples/s]4129 examples [00:04, 895.67 examples/s]4223 examples [00:04, 906.64 examples/s]4314 examples [00:04, 899.00 examples/s]4405 examples [00:05, 870.44 examples/s]4493 examples [00:05, 861.55 examples/s]4581 examples [00:05, 865.20 examples/s]4668 examples [00:05, 859.66 examples/s]4761 examples [00:05, 879.54 examples/s]4858 examples [00:05, 903.87 examples/s]4952 examples [00:05, 914.14 examples/s]5044 examples [00:05, 908.25 examples/s]5136 examples [00:05, 910.86 examples/s]5228 examples [00:06, 910.50 examples/s]5320 examples [00:06, 905.22 examples/s]5411 examples [00:06, 884.40 examples/s]5500 examples [00:06, 878.54 examples/s]5595 examples [00:06, 896.10 examples/s]5685 examples [00:06, 879.87 examples/s]5774 examples [00:06, 870.41 examples/s]5862 examples [00:06, 866.92 examples/s]5949 examples [00:06, 867.12 examples/s]6041 examples [00:06, 882.18 examples/s]6131 examples [00:07, 886.56 examples/s]6228 examples [00:07, 909.68 examples/s]6329 examples [00:07, 936.40 examples/s]6423 examples [00:07, 929.01 examples/s]6517 examples [00:07, 901.63 examples/s]6608 examples [00:07, 889.54 examples/s]6698 examples [00:07, 884.89 examples/s]6787 examples [00:07, 874.02 examples/s]6875 examples [00:07, 842.77 examples/s]6962 examples [00:07, 849.20 examples/s]7051 examples [00:08, 858.92 examples/s]7139 examples [00:08, 863.21 examples/s]7228 examples [00:08, 869.37 examples/s]7317 examples [00:08, 873.47 examples/s]7405 examples [00:08, 860.02 examples/s]7494 examples [00:08, 866.17 examples/s]7589 examples [00:08, 888.20 examples/s]7688 examples [00:08, 916.26 examples/s]7782 examples [00:08, 923.23 examples/s]7875 examples [00:08, 921.27 examples/s]7968 examples [00:09, 913.45 examples/s]8060 examples [00:09, 890.71 examples/s]8150 examples [00:09, 881.68 examples/s]8239 examples [00:09, 852.11 examples/s]8325 examples [00:09, 824.90 examples/s]8408 examples [00:09, 819.68 examples/s]8496 examples [00:09, 834.95 examples/s]8582 examples [00:09, 841.92 examples/s]8668 examples [00:09, 844.87 examples/s]8754 examples [00:10, 849.06 examples/s]8842 examples [00:10, 856.73 examples/s]8928 examples [00:10, 850.17 examples/s]9014 examples [00:10, 824.21 examples/s]9102 examples [00:10, 837.39 examples/s]9187 examples [00:10, 840.72 examples/s]9272 examples [00:10, 829.74 examples/s]9358 examples [00:10, 836.30 examples/s]9446 examples [00:10, 847.58 examples/s]9534 examples [00:10, 854.61 examples/s]9620 examples [00:11, 853.45 examples/s]9706 examples [00:11, 840.74 examples/s]9791 examples [00:11, 830.61 examples/s]9875 examples [00:11, 824.26 examples/s]9959 examples [00:11, 825.17 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVD8L3J/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVD8L3J/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f94d204a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f94d204a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f94d204a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f953dee74e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94b361cf98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f94d204a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f94d204a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f94d204a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94b4414320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94b44141d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f94a80446a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f94a80446a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f94a80446a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f94a80447b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f94a80447b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f94a80447b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4eecc85aa0a3c7ddfed7f1460a58feb2e371b166275e2398f46b3a5e4e1c50f6
  Stored in directory: /tmp/pip-ephem-wheel-cache-jaeixx2o/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:36:31, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:45:43, 14.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:47:15, 20.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:15:32, 29.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<5:45:58, 41.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.51M/862M [00:01<4:01:11, 59.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:47:54, 84.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:01<1:56:52, 120kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:01<1:21:38, 172kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:01<56:51, 245kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:02<39:41, 349kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<27:40, 496kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:02<19:19, 706kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:02<13:32, 1.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<10:14, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<09:03, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<09:59, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<07:48, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<05:41, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<07:40, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<07:01, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:15, 2.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<06:22, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:18, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:49, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:09<04:14, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<23:45, 556kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<17:58, 735kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<12:53, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<12:05, 1.09MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<11:09, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<08:29, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<08:02, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<06:59, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<05:13, 2.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:40, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<06:00, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<04:31, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:13, 2.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:40, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<04:15, 3.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:02, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<05:32, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<04:12, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<04:09, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<04:08, 3.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:52, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:26, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:04, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<03:42, 3.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<8:45:19, 24.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<6:08:08, 34.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<4:16:57, 48.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<3:08:09, 66.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<2:14:21, 93.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<1:34:31, 132kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<1:06:14, 189kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<50:18, 248kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<36:33, 341kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<25:51, 481kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<20:53, 594kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<15:53, 780kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<11:25, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:53, 1.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:40, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:19, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:35, 2.67MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<52:41, 233kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<38:07, 322kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<26:54, 454kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<21:38, 563kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<17:34, 694kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:55, 943kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:58, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:57, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:34, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:23, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:40, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:54, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<04:15, 2.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:09, 791kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:51, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:47, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:40, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:40, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:34, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:48, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:22, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:48, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:39, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:16, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:49, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:26:03, 136kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:01:26, 191kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<43:10, 271kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<32:49, 355kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<24:09, 482kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<17:07, 678kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<12:12, 950kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<18:00, 643kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<16:46, 690kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<12:44, 908kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<09:08, 1.26MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:31, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:41, 1.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:33, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:33, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:31, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:58, 2.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:38, 3.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:54, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:28, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:42, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:56, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:05, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:44, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:36, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:23, 2.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:59, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:28, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:14, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:04, 3.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<2:14:05, 82.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:35:43, 116kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<1:07:23, 164kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<48:52, 226kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<36:06, 305kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<25:42, 428kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<19:48, 553kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<15:46, 694kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<11:29, 951kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:53, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:52, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:36, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:43, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<22:07, 489kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<17:20, 624kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<12:30, 863kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<08:54, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:50, 992kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<09:24, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<05:26, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:46:58, 22.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<5:27:25, 32.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<3:48:43, 46.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:42:37, 65.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<1:57:32, 90.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:23:00, 128kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<58:16, 182kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<42:46, 247kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<31:47, 332kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<22:40, 464kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<17:36, 595kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<14:05, 743kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:18, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:58, 1.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:04, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:01, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:57, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:36, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:59, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:18, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:08, 2.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:39, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:59, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:55, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:28, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:51, 2.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:49, 2.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:24, 2.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<04:47, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<03:43, 2.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<02:42, 3.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<20:29, 488kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<16:02, 623kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:34, 862kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<08:11, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<17:29, 567kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<13:59, 709kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:12, 971kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:48, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:50, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:54, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:47, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:43, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:24, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:45, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:58, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:53, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:22, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:45, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:44, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:15, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:39, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:11, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:16, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:19, 2.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:20, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:20, 2.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:26, 3.85MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<14:40, 639kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<11:51, 791kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:41, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:39, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:00, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:47, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:52, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:24, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:52, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:58, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:03, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:55, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:18, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:34, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:35, 2.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:03, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:27, 2.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:29, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:59, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:20, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:25, 2.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<02:34, 3.46MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:17, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:32, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:29, 2.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:35, 3.42MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:19, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:14, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:02, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:19, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:33, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:29, 2.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:38, 3.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:17, 2.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:11, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<05:05, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:58, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:56, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:49, 2.25MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:08, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:24, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:44, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<6:30:41, 21.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<4:33:57, 30.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<3:11:13, 44.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<2:15:52, 61.9kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:38:02, 85.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<1:09:14, 121kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<48:28, 173kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<36:04, 231kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<26:39, 313kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<18:59, 439kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<14:38, 565kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<11:38, 710kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<08:29, 972kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:19, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:32, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:55, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:49, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:49, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:42, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:58, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:09, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:14, 2.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:38, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:54, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:04, 2.60MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:30, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:48, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:59, 2.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:26, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:47, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<02:55, 2.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:08, 3.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:28, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:51, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<04:24, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:24, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:41, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:52, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:59, 2.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:09, 3.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<13:03, 581kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<10:25, 728kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:36, 995kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<06:35, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:55, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:27, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:23, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:20, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:36, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:46, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<02:57, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:18, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:33, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:47, 2.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:11, 2.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:30, 2.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:42, 2.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<01:59, 3.62MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:18, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:55, 1.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:44, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:50, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:53, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:58, 2.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:09, 3.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<07:13, 973kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<06:15, 1.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:40, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<04:26, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<04:06, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:18, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:45, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:14, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:54, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:58, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:58, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:28, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:28, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:32, 2.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:54, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:23, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:25, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:29, 2.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:14, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:36, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:37, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:37, 2.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:27, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:04, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:34, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:16, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:48, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:49, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:41, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:40, 2.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:22, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:34, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:30, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:31, 2.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:41, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:43, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:39, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:38, 2.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<07:20, 864kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<06:33, 968kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<04:52, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:30, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:31, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:33, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:28, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:31, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:55, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:16, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:23, 2.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:36, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:57, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:08, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:16, 2.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:02, 1.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<05:35, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:14, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:01, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:44, 775kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<06:41, 896kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:59, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:33, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<09:23, 632kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<07:57, 745kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:51, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:13, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:19, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:17, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:18, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:23, 2.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<03:18, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<4:10:51, 23.1kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<2:56:01, 32.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<2:02:50, 46.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<1:26:44, 66.0kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<1:01:54, 92.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<43:30, 131kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<30:25, 187kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<22:51, 247kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<17:12, 329kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<12:19, 458kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<08:36, 649kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<12:31, 446kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<09:57, 561kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<07:12, 773kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<05:06, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:26, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:59, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:43, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:41, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:45, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:58, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:03, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:13, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:05, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:33, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:02, 2.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:30, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:42, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:53, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:04, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:24, 971kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:54, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:42, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:37, 1.97MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<07:12, 717kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<06:15, 826kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:40, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<03:19, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<06:06, 835kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<05:21, 951kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:58, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:50, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:06, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:56, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:59, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:14, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:38, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:44, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:03, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:14, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:50, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:35, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:56, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:48, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:32, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:53, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:07, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:47, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:31, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:47, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:01, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:30, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<07:02, 667kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<06:27, 726kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:49, 968kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:27, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:44, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<04:01, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:06, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:39, 2.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<08:54, 511kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<07:38, 596kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<05:38, 805kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:03, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<02:52, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<08:48, 508kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<07:37, 587kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:37, 794kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:59, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:52, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<09:44, 453kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<08:10, 539kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:59, 735kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:15, 1.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:03, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<11:22, 382kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<09:13, 470kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<06:44, 642kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:46, 900kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:38, 921kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:33, 936kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:27, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:29, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:48, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<10:43, 392kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<08:43, 482kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:24, 654kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<04:31, 919kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<04:24, 939kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<04:21, 949kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:18, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:22, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:43, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<10:01, 406kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<08:17, 491kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<06:05, 665kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<04:18, 932kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:12, 951kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:09, 959kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:10, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:15, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:39, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<10:46, 365kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<08:45, 448kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<06:21, 616kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<04:29, 866kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<03:11, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<10:54, 353kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<08:45, 440kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<06:24, 600kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:31, 844kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:18, 880kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<04:10, 907kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:12, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:17, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:45, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<03:00, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:20, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:41, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:14, 2.93MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<09:24, 388kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<07:42, 473kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<05:39, 643kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:59, 905kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:51, 926kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<03:48, 941kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:55, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:06, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:31, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:50, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:12, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:36, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:08, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:29, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:57, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:25, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:02, 3.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<11:16, 299kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<08:55, 378kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<06:26, 522kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<04:30, 737kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<03:12, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<14:36, 226kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<11:10, 295kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<08:00, 411kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:36, 582kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<03:57, 818kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<14:07, 229kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<10:49, 299kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<07:45, 415kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<05:25, 588kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<03:50, 826kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<13:08, 241kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<10:10, 311kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<07:17, 432kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<05:07, 611kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<03:36, 859kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<12:46, 242kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<09:49, 315kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<07:04, 436kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<04:57, 616kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:24, 686kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:58, 762kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:59, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:07, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:24, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:29, 1.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<01:54, 1.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:54, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:13, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:45, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:46, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:00, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:09, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:41, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:56, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:30, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:05, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:59, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<1:57:24, 22.8kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<1:22:25, 32.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<57:22, 46.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<39:59, 65.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<28:40, 91.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<20:08, 129kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<13:58, 184kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<10:29, 242kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<08:02, 316kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<05:44, 440kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<04:00, 623kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:35, 688kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:19, 744kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:28, 992kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:45, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:49, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:53, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:28, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:03, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:36, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:40, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:10, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:01, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:30, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:05, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:20, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:28, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:09, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:50, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:36, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:16, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:54, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:44, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:38, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:14, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:52, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<05:36, 355kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:22, 454kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<03:08, 630kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<02:11, 886kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:06, 907kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:55, 991kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:27, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:02, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:33, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:37, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:15, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:53, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:15, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:22, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:46, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:13, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:19, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:02, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:44, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:13, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:17, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:00, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:42, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:11, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:14, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:56, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:40, 2.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:03, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:07, 1.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:52, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:37, 2.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:09, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:08, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:37, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:53, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:57, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:44, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:31, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:46, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:33, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:47, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:26, 2.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:45, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:25, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:39, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:30, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:21, 2.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:49, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:50, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:38, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:37, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:20, 2.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:33, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:18, 2.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:21, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:15, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:20, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 2.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:20, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:25, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:13, 2.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:19, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 2.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:16, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 3.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:26, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:25, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:17, 1.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:16, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 2.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:11, 972kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:10, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:09, 728kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:08, 836kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.69MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<59:39:12,  1.86it/s]  0%|          | 751/400000 [00:00<41:41:00,  2.66it/s]  0%|          | 1523/400000 [00:00<29:07:34,  3.80it/s]  1%|          | 2220/400000 [00:00<20:21:27,  5.43it/s]  1%|          | 2933/400000 [00:00<14:13:45,  7.75it/s]  1%|          | 3665/400000 [00:01<9:56:48, 11.07it/s]   1%|          | 4405/400000 [00:01<6:57:14, 15.80it/s]  1%|▏         | 5130/400000 [00:01<4:51:48, 22.55it/s]  1%|▏         | 5849/400000 [00:01<3:24:10, 32.17it/s]  2%|▏         | 6602/400000 [00:01<2:22:54, 45.88it/s]  2%|▏         | 7307/400000 [00:01<1:40:09, 65.35it/s]  2%|▏         | 8023/400000 [00:01<1:10:15, 92.99it/s]  2%|▏         | 8756/400000 [00:01<49:21, 132.13it/s]   2%|▏         | 9479/400000 [00:01<34:45, 187.29it/s]  3%|▎         | 10227/400000 [00:01<24:32, 264.71it/s]  3%|▎         | 10980/400000 [00:02<17:24, 372.54it/s]  3%|▎         | 11720/400000 [00:02<12:25, 520.96it/s]  3%|▎         | 12462/400000 [00:02<08:56, 722.48it/s]  3%|▎         | 13199/400000 [00:02<06:31, 989.14it/s]  3%|▎         | 13929/400000 [00:02<04:49, 1331.35it/s]  4%|▎         | 14647/400000 [00:02<03:38, 1761.49it/s]  4%|▍         | 15384/400000 [00:02<02:48, 2282.36it/s]  4%|▍         | 16107/400000 [00:02<02:13, 2866.56it/s]  4%|▍         | 16834/400000 [00:02<01:49, 3503.10it/s]  4%|▍         | 17581/400000 [00:02<01:31, 4166.06it/s]  5%|▍         | 18332/400000 [00:03<01:19, 4808.13it/s]  5%|▍         | 19068/400000 [00:03<01:11, 5344.34it/s]  5%|▍         | 19800/400000 [00:03<01:05, 5806.91it/s]  5%|▌         | 20531/400000 [00:03<01:02, 6107.77it/s]  5%|▌         | 21295/400000 [00:03<00:58, 6498.76it/s]  6%|▌         | 22059/400000 [00:03<00:55, 6802.68it/s]  6%|▌         | 22816/400000 [00:03<00:53, 7015.17it/s]  6%|▌         | 23563/400000 [00:03<00:52, 7104.69it/s]  6%|▌         | 24306/400000 [00:03<00:52, 7106.53it/s]  6%|▋         | 25039/400000 [00:03<00:52, 7102.43it/s]  6%|▋         | 25801/400000 [00:04<00:51, 7248.10it/s]  7%|▋         | 26557/400000 [00:04<00:50, 7336.34it/s]  7%|▋         | 27300/400000 [00:04<00:50, 7347.98it/s]  7%|▋         | 28041/400000 [00:04<00:50, 7311.52it/s]  7%|▋         | 28777/400000 [00:04<00:51, 7235.34it/s]  7%|▋         | 29508/400000 [00:04<00:51, 7256.42it/s]  8%|▊         | 30245/400000 [00:04<00:50, 7289.29it/s]  8%|▊         | 30986/400000 [00:04<00:50, 7321.69it/s]  8%|▊         | 31720/400000 [00:04<00:51, 7126.65it/s]  8%|▊         | 32435/400000 [00:04<00:52, 7064.38it/s]  8%|▊         | 33175/400000 [00:05<00:51, 7161.28it/s]  8%|▊         | 33918/400000 [00:05<00:50, 7238.81it/s]  9%|▊         | 34668/400000 [00:05<00:49, 7313.17it/s]  9%|▉         | 35405/400000 [00:05<00:49, 7329.20it/s]  9%|▉         | 36139/400000 [00:05<00:49, 7331.24it/s]  9%|▉         | 36883/400000 [00:05<00:49, 7360.86it/s]  9%|▉         | 37620/400000 [00:05<00:49, 7260.01it/s] 10%|▉         | 38376/400000 [00:05<00:49, 7345.10it/s] 10%|▉         | 39112/400000 [00:05<00:49, 7318.93it/s] 10%|▉         | 39874/400000 [00:06<00:48, 7404.72it/s] 10%|█         | 40634/400000 [00:06<00:48, 7459.98it/s] 10%|█         | 41413/400000 [00:06<00:47, 7553.99it/s] 11%|█         | 42170/400000 [00:06<00:47, 7503.48it/s] 11%|█         | 42921/400000 [00:06<00:47, 7454.27it/s] 11%|█         | 43682/400000 [00:06<00:47, 7497.72it/s] 11%|█         | 44435/400000 [00:06<00:47, 7505.60it/s] 11%|█▏        | 45186/400000 [00:06<00:47, 7501.39it/s] 11%|█▏        | 45937/400000 [00:06<00:47, 7483.15it/s] 12%|█▏        | 46686/400000 [00:06<00:47, 7399.45it/s] 12%|█▏        | 47459/400000 [00:07<00:47, 7494.82it/s] 12%|█▏        | 48209/400000 [00:07<00:48, 7273.23it/s] 12%|█▏        | 48939/400000 [00:07<00:48, 7236.85it/s] 12%|█▏        | 49673/400000 [00:07<00:48, 7264.72it/s] 13%|█▎        | 50407/400000 [00:07<00:47, 7284.58it/s] 13%|█▎        | 51137/400000 [00:07<00:48, 7213.00it/s] 13%|█▎        | 51875/400000 [00:07<00:47, 7260.49it/s] 13%|█▎        | 52607/400000 [00:07<00:47, 7277.54it/s] 13%|█▎        | 53359/400000 [00:07<00:47, 7347.89it/s] 14%|█▎        | 54095/400000 [00:07<00:48, 7129.42it/s] 14%|█▎        | 54819/400000 [00:08<00:48, 7159.93it/s] 14%|█▍        | 55545/400000 [00:08<00:47, 7187.43it/s] 14%|█▍        | 56265/400000 [00:08<00:47, 7170.10it/s] 14%|█▍        | 56983/400000 [00:08<00:47, 7165.14it/s] 14%|█▍        | 57700/400000 [00:08<00:47, 7161.77it/s] 15%|█▍        | 58417/400000 [00:08<00:47, 7162.93it/s] 15%|█▍        | 59149/400000 [00:08<00:47, 7208.01it/s] 15%|█▍        | 59887/400000 [00:08<00:46, 7258.25it/s] 15%|█▌        | 60643/400000 [00:08<00:46, 7343.95it/s] 15%|█▌        | 61378/400000 [00:08<00:46, 7326.65it/s] 16%|█▌        | 62129/400000 [00:09<00:45, 7379.28it/s] 16%|█▌        | 62884/400000 [00:09<00:45, 7429.49it/s] 16%|█▌        | 63628/400000 [00:09<00:45, 7380.26it/s] 16%|█▌        | 64379/400000 [00:09<00:45, 7417.83it/s] 16%|█▋        | 65122/400000 [00:09<00:45, 7361.11it/s] 16%|█▋        | 65886/400000 [00:09<00:44, 7440.22it/s] 17%|█▋        | 66638/400000 [00:09<00:44, 7462.49it/s] 17%|█▋        | 67385/400000 [00:09<00:44, 7454.70it/s] 17%|█▋        | 68131/400000 [00:09<00:45, 7351.30it/s] 17%|█▋        | 68867/400000 [00:09<00:45, 7262.19it/s] 17%|█▋        | 69594/400000 [00:10<00:46, 7099.79it/s] 18%|█▊        | 70339/400000 [00:10<00:45, 7201.00it/s] 18%|█▊        | 71061/400000 [00:10<00:45, 7179.92it/s] 18%|█▊        | 71808/400000 [00:10<00:45, 7263.82it/s] 18%|█▊        | 72536/400000 [00:10<00:45, 7195.90it/s] 18%|█▊        | 73314/400000 [00:10<00:44, 7360.39it/s] 19%|█▊        | 74064/400000 [00:10<00:44, 7399.87it/s] 19%|█▊        | 74816/400000 [00:10<00:43, 7433.24it/s] 19%|█▉        | 75561/400000 [00:10<00:44, 7366.77it/s] 19%|█▉        | 76299/400000 [00:10<00:44, 7304.82it/s] 19%|█▉        | 77031/400000 [00:11<00:44, 7249.79it/s] 19%|█▉        | 77797/400000 [00:11<00:43, 7366.99it/s] 20%|█▉        | 78537/400000 [00:11<00:43, 7375.37it/s] 20%|█▉        | 79295/400000 [00:11<00:43, 7435.36it/s] 20%|██        | 80040/400000 [00:11<00:43, 7385.54it/s] 20%|██        | 80779/400000 [00:11<00:43, 7360.33it/s] 20%|██        | 81516/400000 [00:11<00:43, 7345.54it/s] 21%|██        | 82265/400000 [00:11<00:43, 7387.90it/s] 21%|██        | 83010/400000 [00:11<00:42, 7405.51it/s] 21%|██        | 83751/400000 [00:11<00:42, 7381.33it/s] 21%|██        | 84498/400000 [00:12<00:42, 7406.64it/s] 21%|██▏       | 85239/400000 [00:12<00:42, 7363.07it/s] 21%|██▏       | 85995/400000 [00:12<00:42, 7420.59it/s] 22%|██▏       | 86738/400000 [00:12<00:42, 7423.07it/s] 22%|██▏       | 87481/400000 [00:12<00:42, 7384.16it/s] 22%|██▏       | 88235/400000 [00:12<00:41, 7429.07it/s] 22%|██▏       | 88979/400000 [00:12<00:41, 7416.71it/s] 22%|██▏       | 89721/400000 [00:12<00:42, 7265.29it/s] 23%|██▎       | 90462/400000 [00:12<00:42, 7305.17it/s] 23%|██▎       | 91194/400000 [00:12<00:42, 7260.17it/s] 23%|██▎       | 91921/400000 [00:13<00:42, 7262.09it/s] 23%|██▎       | 92666/400000 [00:13<00:42, 7316.89it/s] 23%|██▎       | 93405/400000 [00:13<00:41, 7338.00it/s] 24%|██▎       | 94140/400000 [00:13<00:42, 7273.62it/s] 24%|██▎       | 94868/400000 [00:13<00:42, 7219.24it/s] 24%|██▍       | 95617/400000 [00:13<00:41, 7297.29it/s] 24%|██▍       | 96365/400000 [00:13<00:41, 7349.85it/s] 24%|██▍       | 97112/400000 [00:13<00:41, 7382.90it/s] 24%|██▍       | 97872/400000 [00:13<00:40, 7445.96it/s] 25%|██▍       | 98617/400000 [00:14<00:41, 7260.79it/s] 25%|██▍       | 99377/400000 [00:14<00:40, 7358.94it/s] 25%|██▌       | 100118/400000 [00:14<00:40, 7373.99it/s] 25%|██▌       | 100864/400000 [00:14<00:40, 7397.54it/s] 25%|██▌       | 101613/400000 [00:14<00:40, 7423.40it/s] 26%|██▌       | 102356/400000 [00:14<00:40, 7403.80it/s] 26%|██▌       | 103097/400000 [00:14<00:40, 7398.97it/s] 26%|██▌       | 103853/400000 [00:14<00:39, 7445.73it/s] 26%|██▌       | 104611/400000 [00:14<00:39, 7483.89it/s] 26%|██▋       | 105360/400000 [00:14<00:39, 7461.79it/s] 27%|██▋       | 106107/400000 [00:15<00:39, 7385.21it/s] 27%|██▋       | 106846/400000 [00:15<00:40, 7321.05it/s] 27%|██▋       | 107579/400000 [00:15<00:39, 7318.57it/s] 27%|██▋       | 108333/400000 [00:15<00:39, 7382.49it/s] 27%|██▋       | 109072/400000 [00:15<00:39, 7355.43it/s] 27%|██▋       | 109808/400000 [00:15<00:39, 7341.51it/s] 28%|██▊       | 110563/400000 [00:15<00:39, 7402.06it/s] 28%|██▊       | 111304/400000 [00:15<00:39, 7329.71it/s] 28%|██▊       | 112063/400000 [00:15<00:38, 7405.50it/s] 28%|██▊       | 112804/400000 [00:15<00:38, 7375.97it/s] 28%|██▊       | 113542/400000 [00:16<00:38, 7353.23it/s] 29%|██▊       | 114285/400000 [00:16<00:38, 7374.71it/s] 29%|██▉       | 115033/400000 [00:16<00:38, 7403.39it/s] 29%|██▉       | 115774/400000 [00:16<00:38, 7376.91it/s] 29%|██▉       | 116512/400000 [00:16<00:38, 7376.72it/s] 29%|██▉       | 117250/400000 [00:16<00:38, 7261.74it/s] 29%|██▉       | 117977/400000 [00:16<00:39, 7191.87it/s] 30%|██▉       | 118706/400000 [00:16<00:38, 7220.39it/s] 30%|██▉       | 119466/400000 [00:16<00:38, 7329.32it/s] 30%|███       | 120201/400000 [00:16<00:38, 7333.47it/s] 30%|███       | 120935/400000 [00:17<00:38, 7335.09it/s] 30%|███       | 121669/400000 [00:17<00:39, 7088.22it/s] 31%|███       | 122405/400000 [00:17<00:38, 7167.00it/s] 31%|███       | 123138/400000 [00:17<00:38, 7211.67it/s] 31%|███       | 123861/400000 [00:17<00:38, 7212.43it/s] 31%|███       | 124584/400000 [00:17<00:38, 7158.77it/s] 31%|███▏      | 125304/400000 [00:17<00:38, 7167.72it/s] 32%|███▏      | 126022/400000 [00:17<00:38, 7140.54it/s] 32%|███▏      | 126746/400000 [00:17<00:38, 7167.77it/s] 32%|███▏      | 127502/400000 [00:17<00:37, 7280.45it/s] 32%|███▏      | 128252/400000 [00:18<00:37, 7341.78it/s] 32%|███▏      | 128987/400000 [00:18<00:37, 7259.38it/s] 32%|███▏      | 129733/400000 [00:18<00:36, 7316.77it/s] 33%|███▎      | 130470/400000 [00:18<00:36, 7332.10it/s] 33%|███▎      | 131214/400000 [00:18<00:36, 7364.01it/s] 33%|███▎      | 131951/400000 [00:18<00:36, 7337.39it/s] 33%|███▎      | 132685/400000 [00:18<00:37, 7190.20it/s] 33%|███▎      | 133439/400000 [00:18<00:36, 7289.41it/s] 34%|███▎      | 134183/400000 [00:18<00:36, 7333.42it/s] 34%|███▎      | 134929/400000 [00:18<00:35, 7369.35it/s] 34%|███▍      | 135667/400000 [00:19<00:36, 7265.52it/s] 34%|███▍      | 136404/400000 [00:19<00:36, 7295.13it/s] 34%|███▍      | 137135/400000 [00:19<00:36, 7265.07it/s] 34%|███▍      | 137897/400000 [00:19<00:35, 7365.92it/s] 35%|███▍      | 138635/400000 [00:19<00:36, 7128.94it/s] 35%|███▍      | 139376/400000 [00:19<00:36, 7208.94it/s] 35%|███▌      | 140136/400000 [00:19<00:35, 7320.06it/s] 35%|███▌      | 140883/400000 [00:19<00:35, 7362.64it/s] 35%|███▌      | 141621/400000 [00:19<00:35, 7274.90it/s] 36%|███▌      | 142350/400000 [00:19<00:35, 7212.58it/s] 36%|███▌      | 143073/400000 [00:20<00:36, 7075.21it/s] 36%|███▌      | 143829/400000 [00:20<00:35, 7212.12it/s] 36%|███▌      | 144585/400000 [00:20<00:34, 7312.65it/s] 36%|███▋      | 145335/400000 [00:20<00:34, 7367.56it/s] 37%|███▋      | 146073/400000 [00:20<00:34, 7322.61it/s] 37%|███▋      | 146807/400000 [00:20<00:34, 7308.59it/s] 37%|███▋      | 147569/400000 [00:20<00:34, 7397.07it/s] 37%|███▋      | 148310/400000 [00:20<00:34, 7349.87it/s] 37%|███▋      | 149046/400000 [00:20<00:34, 7350.27it/s] 37%|███▋      | 149793/400000 [00:21<00:33, 7383.64it/s] 38%|███▊      | 150546/400000 [00:21<00:33, 7424.57it/s] 38%|███▊      | 151314/400000 [00:21<00:33, 7499.35it/s] 38%|███▊      | 152070/400000 [00:21<00:32, 7516.95it/s] 38%|███▊      | 152830/400000 [00:21<00:32, 7539.17it/s] 38%|███▊      | 153589/400000 [00:21<00:32, 7553.44it/s] 39%|███▊      | 154345/400000 [00:21<00:33, 7377.80it/s] 39%|███▉      | 155090/400000 [00:21<00:33, 7398.30it/s] 39%|███▉      | 155852/400000 [00:21<00:32, 7460.85it/s] 39%|███▉      | 156602/400000 [00:21<00:32, 7471.75it/s] 39%|███▉      | 157356/400000 [00:22<00:32, 7490.84it/s] 40%|███▉      | 158106/400000 [00:22<00:32, 7454.42it/s] 40%|███▉      | 158852/400000 [00:22<00:32, 7369.29it/s] 40%|███▉      | 159621/400000 [00:22<00:32, 7462.12it/s] 40%|████      | 160386/400000 [00:22<00:31, 7514.71it/s] 40%|████      | 161138/400000 [00:22<00:31, 7485.71it/s] 40%|████      | 161887/400000 [00:22<00:31, 7450.29it/s] 41%|████      | 162633/400000 [00:22<00:32, 7408.87it/s] 41%|████      | 163375/400000 [00:22<00:32, 7391.52it/s] 41%|████      | 164118/400000 [00:22<00:31, 7401.26it/s] 41%|████      | 164859/400000 [00:23<00:31, 7387.36it/s] 41%|████▏     | 165605/400000 [00:23<00:31, 7408.28it/s] 42%|████▏     | 166346/400000 [00:23<00:32, 7175.71it/s] 42%|████▏     | 167066/400000 [00:23<00:33, 7042.59it/s] 42%|████▏     | 167782/400000 [00:23<00:32, 7076.12it/s] 42%|████▏     | 168529/400000 [00:23<00:32, 7188.65it/s] 42%|████▏     | 169258/400000 [00:23<00:31, 7216.89it/s] 42%|████▏     | 169981/400000 [00:23<00:32, 7064.84it/s] 43%|████▎     | 170718/400000 [00:23<00:32, 7152.99it/s] 43%|████▎     | 171441/400000 [00:23<00:31, 7173.58it/s] 43%|████▎     | 172205/400000 [00:24<00:31, 7304.59it/s] 43%|████▎     | 172971/400000 [00:24<00:30, 7405.57it/s] 43%|████▎     | 173713/400000 [00:24<00:30, 7394.62it/s] 44%|████▎     | 174463/400000 [00:24<00:30, 7424.14it/s] 44%|████▍     | 175211/400000 [00:24<00:30, 7439.26it/s] 44%|████▍     | 175956/400000 [00:24<00:30, 7383.91it/s] 44%|████▍     | 176709/400000 [00:24<00:30, 7425.81it/s] 44%|████▍     | 177452/400000 [00:24<00:30, 7284.73it/s] 45%|████▍     | 178182/400000 [00:24<00:31, 7016.69it/s] 45%|████▍     | 178920/400000 [00:24<00:31, 7119.72it/s] 45%|████▍     | 179648/400000 [00:25<00:30, 7164.61it/s] 45%|████▌     | 180372/400000 [00:25<00:30, 7184.78it/s] 45%|████▌     | 181094/400000 [00:25<00:30, 7193.38it/s] 45%|████▌     | 181833/400000 [00:25<00:30, 7250.46it/s] 46%|████▌     | 182573/400000 [00:25<00:29, 7292.60it/s] 46%|████▌     | 183313/400000 [00:25<00:29, 7322.32it/s] 46%|████▌     | 184046/400000 [00:25<00:29, 7312.56it/s] 46%|████▌     | 184778/400000 [00:25<00:29, 7301.43it/s] 46%|████▋     | 185529/400000 [00:25<00:29, 7362.59it/s] 47%|████▋     | 186266/400000 [00:25<00:29, 7331.92it/s] 47%|████▋     | 187012/400000 [00:26<00:28, 7368.96it/s] 47%|████▋     | 187757/400000 [00:26<00:28, 7390.31it/s] 47%|████▋     | 188497/400000 [00:26<00:28, 7301.58it/s] 47%|████▋     | 189228/400000 [00:26<00:29, 7254.55it/s] 47%|████▋     | 189966/400000 [00:26<00:28, 7291.28it/s] 48%|████▊     | 190703/400000 [00:26<00:28, 7314.43it/s] 48%|████▊     | 191440/400000 [00:26<00:28, 7330.49it/s] 48%|████▊     | 192174/400000 [00:26<00:28, 7210.04it/s] 48%|████▊     | 192914/400000 [00:26<00:28, 7265.74it/s] 48%|████▊     | 193672/400000 [00:26<00:28, 7355.88it/s] 49%|████▊     | 194430/400000 [00:27<00:27, 7421.20it/s] 49%|████▉     | 195173/400000 [00:27<00:27, 7350.32it/s] 49%|████▉     | 195927/400000 [00:27<00:27, 7405.01it/s] 49%|████▉     | 196678/400000 [00:27<00:27, 7434.25it/s] 49%|████▉     | 197440/400000 [00:27<00:27, 7487.34it/s] 50%|████▉     | 198220/400000 [00:27<00:26, 7575.74it/s] 50%|████▉     | 199027/400000 [00:27<00:26, 7715.87it/s] 50%|████▉     | 199800/400000 [00:27<00:26, 7680.41it/s] 50%|█████     | 200569/400000 [00:27<00:26, 7609.67it/s] 50%|█████     | 201346/400000 [00:28<00:25, 7655.82it/s] 51%|█████     | 202148/400000 [00:28<00:25, 7761.17it/s] 51%|█████     | 202936/400000 [00:28<00:25, 7794.75it/s] 51%|█████     | 203717/400000 [00:28<00:25, 7664.81it/s] 51%|█████     | 204485/400000 [00:28<00:26, 7469.65it/s] 51%|█████▏    | 205274/400000 [00:28<00:25, 7589.84it/s] 52%|█████▏    | 206035/400000 [00:28<00:25, 7515.95it/s] 52%|█████▏    | 206799/400000 [00:28<00:25, 7551.73it/s] 52%|█████▏    | 207570/400000 [00:28<00:25, 7596.14it/s] 52%|█████▏    | 208354/400000 [00:28<00:25, 7665.24it/s] 52%|█████▏    | 209158/400000 [00:29<00:24, 7770.63it/s] 52%|█████▏    | 209936/400000 [00:29<00:24, 7706.43it/s] 53%|█████▎    | 210708/400000 [00:29<00:24, 7678.88it/s] 53%|█████▎    | 211477/400000 [00:29<00:24, 7565.93it/s] 53%|█████▎    | 212248/400000 [00:29<00:24, 7608.48it/s] 53%|█████▎    | 213047/400000 [00:29<00:24, 7718.99it/s] 53%|█████▎    | 213820/400000 [00:29<00:24, 7713.82it/s] 54%|█████▎    | 214592/400000 [00:29<00:24, 7531.36it/s] 54%|█████▍    | 215347/400000 [00:29<00:25, 7260.02it/s] 54%|█████▍    | 216077/400000 [00:29<00:25, 7089.28it/s] 54%|█████▍    | 216839/400000 [00:30<00:25, 7238.48it/s] 54%|█████▍    | 217623/400000 [00:30<00:24, 7408.67it/s] 55%|█████▍    | 218401/400000 [00:30<00:24, 7515.25it/s] 55%|█████▍    | 219155/400000 [00:30<00:24, 7514.00it/s] 55%|█████▍    | 219939/400000 [00:30<00:23, 7607.88it/s] 55%|█████▌    | 220711/400000 [00:30<00:23, 7640.46it/s] 55%|█████▌    | 221477/400000 [00:30<00:23, 7584.70it/s] 56%|█████▌    | 222265/400000 [00:30<00:23, 7670.31it/s] 56%|█████▌    | 223033/400000 [00:30<00:23, 7609.81it/s] 56%|█████▌    | 223852/400000 [00:30<00:22, 7773.87it/s] 56%|█████▌    | 224644/400000 [00:31<00:22, 7814.43it/s] 56%|█████▋    | 225427/400000 [00:31<00:22, 7751.30it/s] 57%|█████▋    | 226212/400000 [00:31<00:22, 7777.89it/s] 57%|█████▋    | 226991/400000 [00:31<00:22, 7571.74it/s] 57%|█████▋    | 227750/400000 [00:31<00:22, 7509.32it/s] 57%|█████▋    | 228503/400000 [00:31<00:22, 7493.58it/s] 57%|█████▋    | 229254/400000 [00:31<00:22, 7433.77it/s] 57%|█████▋    | 229999/400000 [00:31<00:22, 7418.64it/s] 58%|█████▊    | 230751/400000 [00:31<00:22, 7447.20it/s] 58%|█████▊    | 231497/400000 [00:31<00:22, 7390.15it/s] 58%|█████▊    | 232274/400000 [00:32<00:22, 7499.80it/s] 58%|█████▊    | 233025/400000 [00:32<00:22, 7470.12it/s] 58%|█████▊    | 233773/400000 [00:32<00:22, 7471.10it/s] 59%|█████▊    | 234551/400000 [00:32<00:21, 7560.04it/s] 59%|█████▉    | 235322/400000 [00:32<00:21, 7604.14it/s] 59%|█████▉    | 236083/400000 [00:32<00:21, 7518.42it/s] 59%|█████▉    | 236848/400000 [00:32<00:21, 7557.09it/s] 59%|█████▉    | 237605/400000 [00:32<00:21, 7422.17it/s] 60%|█████▉    | 238376/400000 [00:32<00:21, 7504.51it/s] 60%|█████▉    | 239194/400000 [00:32<00:20, 7693.06it/s] 60%|█████▉    | 239966/400000 [00:33<00:21, 7446.18it/s] 60%|██████    | 240731/400000 [00:33<00:21, 7504.62it/s] 60%|██████    | 241484/400000 [00:33<00:21, 7440.64it/s] 61%|██████    | 242230/400000 [00:33<00:21, 7425.17it/s] 61%|██████    | 242982/400000 [00:33<00:21, 7452.32it/s] 61%|██████    | 243729/400000 [00:33<00:21, 7421.33it/s] 61%|██████    | 244475/400000 [00:33<00:20, 7431.48it/s] 61%|██████▏   | 245219/400000 [00:33<00:20, 7393.74it/s] 61%|██████▏   | 245979/400000 [00:33<00:20, 7452.48it/s] 62%|██████▏   | 246755/400000 [00:34<00:20, 7540.19it/s] 62%|██████▏   | 247510/400000 [00:34<00:20, 7516.77it/s] 62%|██████▏   | 248263/400000 [00:34<00:20, 7501.71it/s] 62%|██████▏   | 249014/400000 [00:34<00:20, 7498.76it/s] 62%|██████▏   | 249790/400000 [00:34<00:19, 7572.68it/s] 63%|██████▎   | 250551/400000 [00:34<00:19, 7580.96it/s] 63%|██████▎   | 251310/400000 [00:34<00:19, 7490.82it/s] 63%|██████▎   | 252070/400000 [00:34<00:19, 7519.73it/s] 63%|██████▎   | 252823/400000 [00:34<00:19, 7372.33it/s] 63%|██████▎   | 253593/400000 [00:34<00:19, 7466.22it/s] 64%|██████▎   | 254341/400000 [00:35<00:20, 7256.20it/s] 64%|██████▍   | 255079/400000 [00:35<00:19, 7290.61it/s] 64%|██████▍   | 255842/400000 [00:35<00:19, 7387.84it/s] 64%|██████▍   | 256624/400000 [00:35<00:19, 7512.06it/s] 64%|██████▍   | 257391/400000 [00:35<00:18, 7558.11it/s] 65%|██████▍   | 258190/400000 [00:35<00:18, 7681.18it/s] 65%|██████▍   | 259002/400000 [00:35<00:18, 7807.55it/s] 65%|██████▍   | 259789/400000 [00:35<00:17, 7824.43it/s] 65%|██████▌   | 260573/400000 [00:35<00:17, 7815.82it/s] 65%|██████▌   | 261356/400000 [00:35<00:18, 7514.14it/s] 66%|██████▌   | 262111/400000 [00:36<00:18, 7522.32it/s] 66%|██████▌   | 262905/400000 [00:36<00:17, 7640.88it/s] 66%|██████▌   | 263710/400000 [00:36<00:17, 7756.28it/s] 66%|██████▌   | 264488/400000 [00:36<00:17, 7721.20it/s] 66%|██████▋   | 265291/400000 [00:36<00:17, 7810.66it/s] 67%|██████▋   | 266074/400000 [00:36<00:17, 7779.81it/s] 67%|██████▋   | 266853/400000 [00:36<00:17, 7590.23it/s] 67%|██████▋   | 267631/400000 [00:36<00:17, 7645.37it/s] 67%|██████▋   | 268397/400000 [00:36<00:17, 7440.09it/s] 67%|██████▋   | 269188/400000 [00:36<00:17, 7574.38it/s] 67%|██████▋   | 269973/400000 [00:37<00:16, 7654.18it/s] 68%|██████▊   | 270797/400000 [00:37<00:16, 7819.77it/s] 68%|██████▊   | 271613/400000 [00:37<00:16, 7918.21it/s] 68%|██████▊   | 272407/400000 [00:37<00:16, 7770.64it/s] 68%|██████▊   | 273186/400000 [00:37<00:16, 7604.86it/s] 68%|██████▊   | 273953/400000 [00:37<00:16, 7621.32it/s] 69%|██████▊   | 274717/400000 [00:37<00:16, 7513.87it/s] 69%|██████▉   | 275500/400000 [00:37<00:16, 7605.67it/s] 69%|██████▉   | 276262/400000 [00:37<00:16, 7476.56it/s] 69%|██████▉   | 277041/400000 [00:38<00:16, 7566.25it/s] 69%|██████▉   | 277817/400000 [00:38<00:16, 7622.91it/s] 70%|██████▉   | 278581/400000 [00:38<00:16, 7567.05it/s] 70%|██████▉   | 279363/400000 [00:38<00:15, 7639.06it/s] 70%|███████   | 280128/400000 [00:38<00:15, 7515.10it/s] 70%|███████   | 280907/400000 [00:38<00:15, 7593.15it/s] 70%|███████   | 281668/400000 [00:38<00:15, 7553.18it/s] 71%|███████   | 282424/400000 [00:38<00:15, 7395.37it/s] 71%|███████   | 283165/400000 [00:38<00:15, 7309.07it/s] 71%|███████   | 283922/400000 [00:38<00:15, 7385.25it/s] 71%|███████   | 284686/400000 [00:39<00:15, 7459.70it/s] 71%|███████▏  | 285473/400000 [00:39<00:15, 7576.62it/s] 72%|███████▏  | 286268/400000 [00:39<00:14, 7684.37it/s] 72%|███████▏  | 287038/400000 [00:39<00:14, 7646.12it/s] 72%|███████▏  | 287804/400000 [00:39<00:14, 7545.09it/s] 72%|███████▏  | 288560/400000 [00:39<00:14, 7438.21it/s] 72%|███████▏  | 289309/400000 [00:39<00:14, 7450.96it/s] 73%|███████▎  | 290055/400000 [00:39<00:14, 7445.05it/s] 73%|███████▎  | 290800/400000 [00:39<00:14, 7432.83it/s] 73%|███████▎  | 291544/400000 [00:39<00:15, 7103.65it/s] 73%|███████▎  | 292296/400000 [00:40<00:14, 7221.01it/s] 73%|███████▎  | 293062/400000 [00:40<00:14, 7345.41it/s] 73%|███████▎  | 293817/400000 [00:40<00:14, 7404.07it/s] 74%|███████▎  | 294598/400000 [00:40<00:14, 7520.60it/s] 74%|███████▍  | 295352/400000 [00:40<00:13, 7491.43it/s] 74%|███████▍  | 296145/400000 [00:40<00:13, 7616.94it/s] 74%|███████▍  | 296917/400000 [00:40<00:13, 7645.01it/s] 74%|███████▍  | 297683/400000 [00:40<00:13, 7558.10it/s] 75%|███████▍  | 298448/400000 [00:40<00:13, 7585.27it/s] 75%|███████▍  | 299208/400000 [00:40<00:13, 7421.99it/s] 75%|███████▍  | 299952/400000 [00:41<00:13, 7382.96it/s] 75%|███████▌  | 300705/400000 [00:41<00:13, 7425.35it/s] 75%|███████▌  | 301482/400000 [00:41<00:13, 7524.98it/s] 76%|███████▌  | 302259/400000 [00:41<00:12, 7595.92it/s] 76%|███████▌  | 303029/400000 [00:41<00:12, 7625.26it/s] 76%|███████▌  | 303808/400000 [00:41<00:12, 7673.86it/s] 76%|███████▌  | 304576/400000 [00:41<00:12, 7555.06it/s] 76%|███████▋  | 305364/400000 [00:41<00:12, 7649.42it/s] 77%|███████▋  | 306153/400000 [00:41<00:12, 7719.87it/s] 77%|███████▋  | 306926/400000 [00:41<00:12, 7680.60it/s] 77%|███████▋  | 307713/400000 [00:42<00:11, 7735.69it/s] 77%|███████▋  | 308488/400000 [00:42<00:12, 7480.53it/s] 77%|███████▋  | 309259/400000 [00:42<00:12, 7545.36it/s] 78%|███████▊  | 310037/400000 [00:42<00:11, 7611.89it/s] 78%|███████▊  | 310800/400000 [00:42<00:11, 7514.74it/s] 78%|███████▊  | 311575/400000 [00:42<00:11, 7582.39it/s] 78%|███████▊  | 312345/400000 [00:42<00:11, 7616.11it/s] 78%|███████▊  | 313108/400000 [00:42<00:11, 7600.38it/s] 78%|███████▊  | 313869/400000 [00:42<00:11, 7562.00it/s] 79%|███████▊  | 314633/400000 [00:42<00:11, 7583.49it/s] 79%|███████▉  | 315425/400000 [00:43<00:11, 7671.74it/s] 79%|███████▉  | 316226/400000 [00:43<00:10, 7768.72it/s] 79%|███████▉  | 317004/400000 [00:43<00:10, 7753.40it/s] 79%|███████▉  | 317821/400000 [00:43<00:10, 7872.49it/s] 80%|███████▉  | 318609/400000 [00:43<00:10, 7723.33it/s] 80%|███████▉  | 319385/400000 [00:43<00:10, 7733.30it/s] 80%|████████  | 320168/400000 [00:43<00:10, 7761.35it/s] 80%|████████  | 320945/400000 [00:43<00:10, 7713.03it/s] 80%|████████  | 321717/400000 [00:43<00:10, 7670.64it/s] 81%|████████  | 322485/400000 [00:44<00:10, 7625.29it/s] 81%|████████  | 323278/400000 [00:44<00:09, 7713.81it/s] 81%|████████  | 324050/400000 [00:44<00:09, 7688.44it/s] 81%|████████  | 324873/400000 [00:44<00:09, 7842.40it/s] 81%|████████▏ | 325659/400000 [00:44<00:09, 7774.90it/s] 82%|████████▏ | 326468/400000 [00:44<00:09, 7864.76it/s] 82%|████████▏ | 327256/400000 [00:44<00:09, 7761.51it/s] 82%|████████▏ | 328034/400000 [00:44<00:09, 7669.94it/s] 82%|████████▏ | 328803/400000 [00:44<00:09, 7674.69it/s] 82%|████████▏ | 329572/400000 [00:44<00:09, 7426.28it/s] 83%|████████▎ | 330320/400000 [00:45<00:09, 7441.60it/s] 83%|████████▎ | 331115/400000 [00:45<00:09, 7584.63it/s] 83%|████████▎ | 331884/400000 [00:45<00:08, 7614.47it/s] 83%|████████▎ | 332657/400000 [00:45<00:08, 7648.66it/s] 83%|████████▎ | 333423/400000 [00:45<00:08, 7575.80it/s] 84%|████████▎ | 334182/400000 [00:45<00:08, 7506.79it/s] 84%|████████▎ | 334958/400000 [00:45<00:08, 7579.23it/s] 84%|████████▍ | 335731/400000 [00:45<00:08, 7622.85it/s] 84%|████████▍ | 336506/400000 [00:45<00:08, 7660.04it/s] 84%|████████▍ | 337273/400000 [00:45<00:08, 7574.94it/s] 85%|████████▍ | 338031/400000 [00:46<00:08, 7533.02it/s] 85%|████████▍ | 338785/400000 [00:46<00:08, 7520.78it/s] 85%|████████▍ | 339538/400000 [00:46<00:08, 7427.15it/s] 85%|████████▌ | 340301/400000 [00:46<00:07, 7486.11it/s] 85%|████████▌ | 341066/400000 [00:46<00:07, 7534.03it/s] 85%|████████▌ | 341820/400000 [00:46<00:07, 7497.65it/s] 86%|████████▌ | 342571/400000 [00:46<00:08, 7153.57it/s] 86%|████████▌ | 343367/400000 [00:46<00:07, 7375.50it/s] 86%|████████▌ | 344151/400000 [00:46<00:07, 7508.87it/s] 86%|████████▌ | 344920/400000 [00:46<00:07, 7559.58it/s] 86%|████████▋ | 345722/400000 [00:47<00:07, 7689.38it/s] 87%|████████▋ | 346494/400000 [00:47<00:06, 7660.92it/s] 87%|████████▋ | 347265/400000 [00:47<00:06, 7675.44it/s] 87%|████████▋ | 348072/400000 [00:47<00:06, 7787.43it/s] 87%|████████▋ | 348852/400000 [00:47<00:06, 7666.69it/s] 87%|████████▋ | 349637/400000 [00:47<00:06, 7717.23it/s] 88%|████████▊ | 350438/400000 [00:47<00:06, 7799.92it/s] 88%|████████▊ | 351223/400000 [00:47<00:06, 7812.44it/s] 88%|████████▊ | 352005/400000 [00:47<00:06, 7775.42it/s] 88%|████████▊ | 352784/400000 [00:47<00:06, 7747.36it/s] 88%|████████▊ | 353600/400000 [00:48<00:05, 7863.79it/s] 89%|████████▊ | 354388/400000 [00:48<00:05, 7726.96it/s] 89%|████████▉ | 355162/400000 [00:48<00:05, 7639.36it/s] 89%|████████▉ | 355927/400000 [00:48<00:05, 7572.90it/s] 89%|████████▉ | 356686/400000 [00:48<00:05, 7550.12it/s] 89%|████████▉ | 357442/400000 [00:48<00:05, 7410.15it/s] 90%|████████▉ | 358205/400000 [00:48<00:05, 7472.87it/s] 90%|████████▉ | 358994/400000 [00:48<00:05, 7590.37it/s] 90%|████████▉ | 359774/400000 [00:48<00:05, 7649.57it/s] 90%|█████████ | 360540/400000 [00:49<00:05, 7638.27it/s] 90%|█████████ | 361321/400000 [00:49<00:05, 7688.57it/s] 91%|█████████ | 362091/400000 [00:49<00:04, 7659.28it/s] 91%|█████████ | 362862/400000 [00:49<00:04, 7674.02it/s] 91%|█████████ | 363630/400000 [00:49<00:04, 7624.19it/s] 91%|█████████ | 364393/400000 [00:49<00:04, 7614.98it/s] 91%|█████████▏| 365155/400000 [00:49<00:04, 7585.04it/s] 91%|█████████▏| 365929/400000 [00:49<00:04, 7629.58it/s] 92%|█████████▏| 366693/400000 [00:49<00:04, 7594.57it/s] 92%|█████████▏| 367453/400000 [00:49<00:04, 7454.66it/s] 92%|█████████▏| 368200/400000 [00:50<00:04, 7272.11it/s] 92%|█████████▏| 368969/400000 [00:50<00:04, 7391.29it/s] 92%|█████████▏| 369735/400000 [00:50<00:04, 7469.74it/s] 93%|█████████▎| 370491/400000 [00:50<00:03, 7495.24it/s] 93%|█████████▎| 371282/400000 [00:50<00:03, 7614.67it/s] 93%|█████████▎| 372045/400000 [00:50<00:03, 7445.08it/s] 93%|█████████▎| 372819/400000 [00:50<00:03, 7529.07it/s] 93%|█████████▎| 373588/400000 [00:50<00:03, 7574.16it/s] 94%|█████████▎| 374347/400000 [00:50<00:03, 7533.87it/s] 94%|█████████▍| 375102/400000 [00:50<00:03, 7532.34it/s] 94%|█████████▍| 375856/400000 [00:51<00:03, 7494.07it/s] 94%|█████████▍| 376606/400000 [00:51<00:03, 7371.25it/s] 94%|█████████▍| 377365/400000 [00:51<00:03, 7434.46it/s] 95%|█████████▍| 378110/400000 [00:51<00:02, 7336.82it/s] 95%|█████████▍| 378864/400000 [00:51<00:02, 7394.48it/s] 95%|█████████▍| 379605/400000 [00:51<00:02, 7302.61it/s] 95%|█████████▌| 380336/400000 [00:51<00:02, 7115.63it/s] 95%|█████████▌| 381072/400000 [00:51<00:02, 7186.61it/s] 95%|█████████▌| 381826/400000 [00:51<00:02, 7289.08it/s] 96%|█████████▌| 382623/400000 [00:51<00:02, 7478.55it/s] 96%|█████████▌| 383390/400000 [00:52<00:02, 7533.23it/s] 96%|█████████▌| 384180/400000 [00:52<00:02, 7637.28it/s] 96%|█████████▌| 384946/400000 [00:52<00:01, 7574.03it/s] 96%|█████████▋| 385725/400000 [00:52<00:01, 7637.49it/s] 97%|█████████▋| 386490/400000 [00:52<00:01, 7551.90it/s] 97%|█████████▋| 387247/400000 [00:52<00:01, 7489.15it/s] 97%|█████████▋| 388011/400000 [00:52<00:01, 7532.57it/s] 97%|█████████▋| 388765/400000 [00:52<00:01, 7528.56it/s] 97%|█████████▋| 389519/400000 [00:52<00:01, 7460.13it/s] 98%|█████████▊| 390276/400000 [00:52<00:01, 7491.90it/s] 98%|█████████▊| 391026/400000 [00:53<00:01, 7334.96it/s] 98%|█████████▊| 391761/400000 [00:53<00:01, 7330.40it/s] 98%|█████████▊| 392495/400000 [00:53<00:01, 7331.38it/s] 98%|█████████▊| 393229/400000 [00:53<00:00, 7258.61it/s] 98%|█████████▊| 393998/400000 [00:53<00:00, 7380.42it/s] 99%|█████████▊| 394744/400000 [00:53<00:00, 7401.45it/s] 99%|█████████▉| 395503/400000 [00:53<00:00, 7456.67it/s] 99%|█████████▉| 396266/400000 [00:53<00:00, 7505.93it/s] 99%|█████████▉| 397027/400000 [00:53<00:00, 7536.21it/s] 99%|█████████▉| 397781/400000 [00:54<00:00, 7073.76it/s]100%|█████████▉| 398497/400000 [00:54<00:00, 7099.21it/s]100%|█████████▉| 399262/400000 [00:54<00:00, 7253.31it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7365.87it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f94a98570f0>, <torchtext.data.dataset.TabularDataset object at 0x7f94a9857240>, <torchtext.vocab.Vocab object at 0x7f94a9857160>), {}) 

