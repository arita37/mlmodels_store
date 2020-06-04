
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f93037c6378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f93037c6378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f93758170d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f93758170d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f930cfaa9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f930cfaa9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f93230fb8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f93230fb8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f93230fb8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 55%|█████▍    | 5439488/9912422 [00:00<00:00, 54385595.96it/s]9920512it [00:00, 30398606.64it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 307277.22it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 11177749.64it/s]         
0it [00:00, ?it/s]8192it [00:00, 177190.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f93046d49b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f93046d4cc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f93230fb510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f93230fb510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f93230fb510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:55,  2.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:55,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:38,  4.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:38,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:37,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:37,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:26,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:26,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:26,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:18,  7.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:13, 10.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:13, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:13, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:13, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:09, 13.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 17.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:07, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 17.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 21.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 31.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 36.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 36.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 40.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 40.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 43.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 46.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 46.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 51.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 53.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 53.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 54.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 54.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 54.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 54.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 55.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 55.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 55.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 55.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 56.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 56.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 56.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 56.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.91 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 55.91 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.45 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.69s/ url]
0 examples [00:00, ? examples/s]2020-06-04 00:10:06.013688: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-04 00:10:06.027802: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-04 00:10:06.027978: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ae89bf0830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-04 00:10:06.027997: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
7 examples [00:00, 69.88 examples/s]94 examples [00:00, 96.49 examples/s]190 examples [00:00, 132.09 examples/s]281 examples [00:00, 177.55 examples/s]369 examples [00:00, 233.32 examples/s]462 examples [00:00, 300.82 examples/s]554 examples [00:00, 376.70 examples/s]646 examples [00:00, 457.23 examples/s]734 examples [00:00, 533.44 examples/s]826 examples [00:01, 610.01 examples/s]920 examples [00:01, 681.63 examples/s]1018 examples [00:01, 748.92 examples/s]1110 examples [00:01, 792.69 examples/s]1202 examples [00:01, 814.28 examples/s]1295 examples [00:01, 845.06 examples/s]1390 examples [00:01, 873.14 examples/s]1483 examples [00:01, 886.79 examples/s]1576 examples [00:01, 882.71 examples/s]1668 examples [00:01, 887.58 examples/s]1759 examples [00:02, 891.22 examples/s]1850 examples [00:02, 893.07 examples/s]1942 examples [00:02, 900.38 examples/s]2036 examples [00:02, 910.87 examples/s]2128 examples [00:02, 912.63 examples/s]2220 examples [00:02, 914.74 examples/s]2312 examples [00:02, 899.46 examples/s]2406 examples [00:02, 910.56 examples/s]2501 examples [00:02, 919.54 examples/s]2594 examples [00:02, 901.68 examples/s]2685 examples [00:03, 896.03 examples/s]2777 examples [00:03, 900.64 examples/s]2870 examples [00:03, 909.22 examples/s]2962 examples [00:03, 907.09 examples/s]3053 examples [00:03, 885.91 examples/s]3148 examples [00:03, 903.81 examples/s]3241 examples [00:03, 907.67 examples/s]3333 examples [00:03, 910.91 examples/s]3425 examples [00:03, 901.60 examples/s]3516 examples [00:03, 886.93 examples/s]3605 examples [00:04, 885.57 examples/s]3695 examples [00:04, 888.15 examples/s]3786 examples [00:04, 891.86 examples/s]3876 examples [00:04, 890.84 examples/s]3968 examples [00:04, 898.09 examples/s]4058 examples [00:04, 894.32 examples/s]4148 examples [00:04, 888.54 examples/s]4238 examples [00:04, 889.88 examples/s]4328 examples [00:04, 861.76 examples/s]4421 examples [00:04, 879.19 examples/s]4513 examples [00:05, 889.67 examples/s]4603 examples [00:05, 873.87 examples/s]4691 examples [00:05, 843.28 examples/s]4776 examples [00:05, 844.41 examples/s]4861 examples [00:05, 837.45 examples/s]4956 examples [00:05, 866.24 examples/s]5048 examples [00:05, 878.28 examples/s]5137 examples [00:05, 876.19 examples/s]5225 examples [00:05, 869.87 examples/s]5313 examples [00:06, 870.59 examples/s]5405 examples [00:06, 882.70 examples/s]5494 examples [00:06, 877.17 examples/s]5586 examples [00:06, 888.19 examples/s]5675 examples [00:06, 883.87 examples/s]5765 examples [00:06, 887.24 examples/s]5861 examples [00:06, 907.03 examples/s]5953 examples [00:06, 907.72 examples/s]6045 examples [00:06, 911.05 examples/s]6137 examples [00:06, 911.63 examples/s]6229 examples [00:07, 879.27 examples/s]6321 examples [00:07, 890.11 examples/s]6412 examples [00:07, 894.04 examples/s]6505 examples [00:07, 902.89 examples/s]6596 examples [00:07, 896.01 examples/s]6686 examples [00:07, 896.44 examples/s]6776 examples [00:07, 896.24 examples/s]6866 examples [00:07, 892.11 examples/s]6956 examples [00:07, 884.99 examples/s]7045 examples [00:07, 866.34 examples/s]7134 examples [00:08, 871.35 examples/s]7222 examples [00:08, 852.61 examples/s]7308 examples [00:08, 852.55 examples/s]7397 examples [00:08, 860.57 examples/s]7485 examples [00:08, 865.66 examples/s]7575 examples [00:08, 872.84 examples/s]7663 examples [00:08, 872.93 examples/s]7752 examples [00:08, 876.37 examples/s]7842 examples [00:08, 881.18 examples/s]7931 examples [00:08, 878.26 examples/s]8020 examples [00:09, 881.36 examples/s]8110 examples [00:09, 886.64 examples/s]8199 examples [00:09, 880.91 examples/s]8288 examples [00:09, 851.97 examples/s]8374 examples [00:09, 842.77 examples/s]8465 examples [00:09, 860.19 examples/s]8558 examples [00:09, 878.26 examples/s]8651 examples [00:09, 890.94 examples/s]8743 examples [00:09, 898.28 examples/s]8833 examples [00:10, 862.62 examples/s]8926 examples [00:10, 881.42 examples/s]9015 examples [00:10, 873.31 examples/s]9103 examples [00:10, 863.02 examples/s]9190 examples [00:10, 852.41 examples/s]9277 examples [00:10, 855.11 examples/s]9371 examples [00:10, 877.91 examples/s]9463 examples [00:10, 886.62 examples/s]9554 examples [00:10, 892.83 examples/s]9644 examples [00:10, 883.43 examples/s]9733 examples [00:11, 866.47 examples/s]9826 examples [00:11, 884.06 examples/s]9915 examples [00:11, 863.90 examples/s]10002 examples [00:11, 829.58 examples/s]10086 examples [00:11, 825.62 examples/s]10171 examples [00:11, 832.27 examples/s]10258 examples [00:11, 841.41 examples/s]10345 examples [00:11, 849.41 examples/s]10431 examples [00:11, 848.16 examples/s]10517 examples [00:11, 850.41 examples/s]10608 examples [00:12, 866.52 examples/s]10695 examples [00:12, 858.00 examples/s]10787 examples [00:12, 872.64 examples/s]10879 examples [00:12, 884.41 examples/s]10968 examples [00:12, 872.37 examples/s]11057 examples [00:12, 877.24 examples/s]11147 examples [00:12, 882.22 examples/s]11236 examples [00:12, 875.79 examples/s]11327 examples [00:12, 883.86 examples/s]11416 examples [00:12, 878.44 examples/s]11505 examples [00:13, 879.42 examples/s]11593 examples [00:13, 868.59 examples/s]11680 examples [00:13, 866.51 examples/s]11772 examples [00:13, 880.51 examples/s]11861 examples [00:13, 875.29 examples/s]11954 examples [00:13, 890.63 examples/s]12045 examples [00:13, 895.17 examples/s]12135 examples [00:13, 893.35 examples/s]12225 examples [00:13, 879.54 examples/s]12315 examples [00:14, 878.07 examples/s]12403 examples [00:14, 859.39 examples/s]12500 examples [00:14, 887.56 examples/s]12590 examples [00:14, 881.30 examples/s]12679 examples [00:14, 861.14 examples/s]12766 examples [00:14, 861.27 examples/s]12860 examples [00:14, 881.49 examples/s]12957 examples [00:14, 904.10 examples/s]13054 examples [00:14, 922.49 examples/s]13147 examples [00:14, 916.38 examples/s]13239 examples [00:15, 911.49 examples/s]13331 examples [00:15, 908.68 examples/s]13423 examples [00:15, 909.70 examples/s]13515 examples [00:15, 901.70 examples/s]13606 examples [00:15, 899.50 examples/s]13697 examples [00:15, 889.81 examples/s]13787 examples [00:15, 884.65 examples/s]13876 examples [00:15, 870.72 examples/s]13966 examples [00:15, 877.00 examples/s]14060 examples [00:15, 894.08 examples/s]14150 examples [00:16, 892.05 examples/s]14240 examples [00:16, 857.26 examples/s]14330 examples [00:16, 861.07 examples/s]14423 examples [00:16, 879.07 examples/s]14512 examples [00:16, 875.51 examples/s]14606 examples [00:16, 892.11 examples/s]14699 examples [00:16, 901.38 examples/s]14795 examples [00:16, 917.97 examples/s]14889 examples [00:16, 923.81 examples/s]14982 examples [00:17, 908.27 examples/s]15073 examples [00:17, 900.63 examples/s]15164 examples [00:17, 902.97 examples/s]15255 examples [00:17, 886.97 examples/s]15344 examples [00:17, 846.14 examples/s]15435 examples [00:17, 862.61 examples/s]15525 examples [00:17, 871.05 examples/s]15619 examples [00:17, 888.80 examples/s]15709 examples [00:17, 892.09 examples/s]15799 examples [00:17, 886.89 examples/s]15895 examples [00:18, 905.29 examples/s]15986 examples [00:18, 887.94 examples/s]16076 examples [00:18, 850.75 examples/s]16162 examples [00:18, 840.07 examples/s]16252 examples [00:18, 856.30 examples/s]16343 examples [00:18, 870.70 examples/s]16431 examples [00:18, 862.25 examples/s]16521 examples [00:18, 872.25 examples/s]16609 examples [00:18, 866.82 examples/s]16698 examples [00:18, 872.17 examples/s]16789 examples [00:19, 881.93 examples/s]16878 examples [00:19, 873.30 examples/s]16968 examples [00:19, 880.16 examples/s]17058 examples [00:19, 884.41 examples/s]17152 examples [00:19, 897.99 examples/s]17242 examples [00:19, 879.05 examples/s]17331 examples [00:19, 880.27 examples/s]17427 examples [00:19, 900.91 examples/s]17521 examples [00:19, 910.26 examples/s]17617 examples [00:19, 923.09 examples/s]17710 examples [00:20, 908.86 examples/s]17802 examples [00:20, 901.98 examples/s]17896 examples [00:20, 912.70 examples/s]17991 examples [00:20, 922.79 examples/s]18084 examples [00:20, 903.37 examples/s]18175 examples [00:20, 860.16 examples/s]18267 examples [00:20, 876.08 examples/s]18364 examples [00:20, 901.61 examples/s]18457 examples [00:20, 908.72 examples/s]18553 examples [00:21, 921.59 examples/s]18646 examples [00:21, 921.33 examples/s]18739 examples [00:21, 914.74 examples/s]18833 examples [00:21, 920.56 examples/s]18926 examples [00:21, 920.37 examples/s]19019 examples [00:21, 902.62 examples/s]19112 examples [00:21, 909.36 examples/s]19204 examples [00:21, 905.66 examples/s]19295 examples [00:21, 894.39 examples/s]19388 examples [00:21, 904.41 examples/s]19480 examples [00:22, 905.71 examples/s]19571 examples [00:22, 864.47 examples/s]19659 examples [00:22, 867.02 examples/s]19747 examples [00:22, 833.55 examples/s]19836 examples [00:22, 846.26 examples/s]19922 examples [00:22, 842.77 examples/s]20007 examples [00:22, 805.99 examples/s]20096 examples [00:22, 829.10 examples/s]20183 examples [00:22, 839.51 examples/s]20268 examples [00:23, 792.36 examples/s]20349 examples [00:23, 756.36 examples/s]20428 examples [00:23, 764.29 examples/s]20516 examples [00:23, 793.76 examples/s]20606 examples [00:23, 822.67 examples/s]20697 examples [00:23, 843.41 examples/s]20783 examples [00:23, 814.43 examples/s]20868 examples [00:23, 822.79 examples/s]20955 examples [00:23, 836.20 examples/s]21045 examples [00:23, 852.83 examples/s]21132 examples [00:24, 855.89 examples/s]21218 examples [00:24, 844.04 examples/s]21306 examples [00:24, 851.40 examples/s]21392 examples [00:24, 846.87 examples/s]21477 examples [00:24, 799.61 examples/s]21563 examples [00:24, 816.65 examples/s]21646 examples [00:24, 815.07 examples/s]21728 examples [00:24, 789.21 examples/s]21810 examples [00:24, 795.81 examples/s]21895 examples [00:25, 809.17 examples/s]21978 examples [00:25, 813.84 examples/s]22067 examples [00:25, 834.13 examples/s]22151 examples [00:25, 835.47 examples/s]22244 examples [00:25, 861.62 examples/s]22335 examples [00:25, 874.27 examples/s]22425 examples [00:25, 880.46 examples/s]22516 examples [00:25, 888.44 examples/s]22606 examples [00:25, 890.46 examples/s]22700 examples [00:25, 902.41 examples/s]22794 examples [00:26, 912.75 examples/s]22886 examples [00:26, 913.60 examples/s]22978 examples [00:26, 906.68 examples/s]23069 examples [00:26, 888.56 examples/s]23158 examples [00:26, 885.51 examples/s]23250 examples [00:26, 893.83 examples/s]23343 examples [00:26, 904.16 examples/s]23434 examples [00:26, 886.25 examples/s]23523 examples [00:26, 886.12 examples/s]23618 examples [00:26, 902.12 examples/s]23710 examples [00:27, 906.64 examples/s]23801 examples [00:27, 894.09 examples/s]23891 examples [00:27, 874.33 examples/s]23985 examples [00:27, 890.61 examples/s]24076 examples [00:27, 894.77 examples/s]24170 examples [00:27, 906.97 examples/s]24261 examples [00:27, 901.85 examples/s]24352 examples [00:27, 900.40 examples/s]24443 examples [00:27, 896.29 examples/s]24533 examples [00:27, 896.63 examples/s]24623 examples [00:28, 895.33 examples/s]24713 examples [00:28, 880.72 examples/s]24802 examples [00:28, 872.40 examples/s]24890 examples [00:28, 871.36 examples/s]24979 examples [00:28, 876.48 examples/s]25067 examples [00:28, 855.16 examples/s]25156 examples [00:28, 864.24 examples/s]25243 examples [00:28, 863.95 examples/s]25330 examples [00:28, 828.85 examples/s]25416 examples [00:28, 837.65 examples/s]25508 examples [00:29, 859.23 examples/s]25600 examples [00:29, 876.20 examples/s]25693 examples [00:29, 891.00 examples/s]25783 examples [00:29, 891.19 examples/s]25873 examples [00:29, 878.99 examples/s]25962 examples [00:29, 879.69 examples/s]26051 examples [00:29, 875.70 examples/s]26139 examples [00:29, 857.78 examples/s]26225 examples [00:29, 857.65 examples/s]26319 examples [00:30, 880.50 examples/s]26414 examples [00:30, 898.88 examples/s]26505 examples [00:30, 891.50 examples/s]26595 examples [00:30, 877.10 examples/s]26685 examples [00:30, 881.77 examples/s]26774 examples [00:30, 869.73 examples/s]26863 examples [00:30, 872.31 examples/s]26951 examples [00:30, 846.55 examples/s]27036 examples [00:30, 846.47 examples/s]27124 examples [00:30, 853.56 examples/s]27212 examples [00:31, 859.24 examples/s]27300 examples [00:31, 863.38 examples/s]27388 examples [00:31, 867.24 examples/s]27477 examples [00:31, 869.09 examples/s]27568 examples [00:31, 879.40 examples/s]27656 examples [00:31, 834.61 examples/s]27741 examples [00:31, 837.97 examples/s]27831 examples [00:31, 853.99 examples/s]27918 examples [00:31, 857.17 examples/s]28007 examples [00:31, 864.13 examples/s]28096 examples [00:32, 870.73 examples/s]28187 examples [00:32, 880.63 examples/s]28276 examples [00:32, 861.10 examples/s]28364 examples [00:32, 864.12 examples/s]28451 examples [00:32, 847.18 examples/s]28536 examples [00:32, 842.40 examples/s]28626 examples [00:32, 858.46 examples/s]28718 examples [00:32, 874.40 examples/s]28809 examples [00:32, 882.39 examples/s]28898 examples [00:32, 883.90 examples/s]28987 examples [00:33, 885.34 examples/s]29077 examples [00:33, 888.50 examples/s]29169 examples [00:33, 896.31 examples/s]29259 examples [00:33, 896.32 examples/s]29349 examples [00:33, 892.68 examples/s]29441 examples [00:33, 900.34 examples/s]29532 examples [00:33, 895.09 examples/s]29622 examples [00:33, 879.00 examples/s]29710 examples [00:33, 876.06 examples/s]29798 examples [00:34, 871.55 examples/s]29887 examples [00:34, 874.89 examples/s]29975 examples [00:34, 869.81 examples/s]30063 examples [00:34, 828.30 examples/s]30149 examples [00:34, 835.55 examples/s]30239 examples [00:34, 851.04 examples/s]30332 examples [00:34, 870.70 examples/s]30420 examples [00:34, 834.14 examples/s]30504 examples [00:34, 812.99 examples/s]30594 examples [00:34, 836.99 examples/s]30685 examples [00:35, 855.74 examples/s]30776 examples [00:35, 869.61 examples/s]30868 examples [00:35, 883.95 examples/s]30957 examples [00:35, 885.54 examples/s]31046 examples [00:35, 870.56 examples/s]31134 examples [00:35, 869.29 examples/s]31223 examples [00:35, 874.78 examples/s]31311 examples [00:35, 855.35 examples/s]31401 examples [00:35, 867.84 examples/s]31491 examples [00:35, 876.12 examples/s]31580 examples [00:36, 877.11 examples/s]31671 examples [00:36, 884.25 examples/s]31760 examples [00:36, 881.85 examples/s]31850 examples [00:36, 884.48 examples/s]31939 examples [00:36, 861.11 examples/s]32026 examples [00:36, 843.50 examples/s]32114 examples [00:36, 852.39 examples/s]32202 examples [00:36, 858.29 examples/s]32288 examples [00:36, 846.55 examples/s]32376 examples [00:37, 854.26 examples/s]32462 examples [00:37, 850.94 examples/s]32553 examples [00:37, 866.10 examples/s]32647 examples [00:37, 885.43 examples/s]32738 examples [00:37, 891.56 examples/s]32828 examples [00:37, 888.13 examples/s]32917 examples [00:37, 885.17 examples/s]33009 examples [00:37, 895.21 examples/s]33099 examples [00:37, 892.86 examples/s]33189 examples [00:37, 851.23 examples/s]33278 examples [00:38, 861.81 examples/s]33369 examples [00:38, 874.77 examples/s]33459 examples [00:38, 877.97 examples/s]33549 examples [00:38, 881.81 examples/s]33638 examples [00:38, 879.64 examples/s]33727 examples [00:38, 872.89 examples/s]33817 examples [00:38, 879.24 examples/s]33905 examples [00:38, 878.21 examples/s]33993 examples [00:38, 833.39 examples/s]34079 examples [00:38, 840.62 examples/s]34168 examples [00:39, 853.37 examples/s]34260 examples [00:39, 870.74 examples/s]34350 examples [00:39, 877.28 examples/s]34441 examples [00:39, 885.41 examples/s]34530 examples [00:39, 845.82 examples/s]34619 examples [00:39, 858.03 examples/s]34712 examples [00:39, 876.64 examples/s]34803 examples [00:39, 885.51 examples/s]34894 examples [00:39, 892.46 examples/s]34984 examples [00:39, 887.17 examples/s]35073 examples [00:40, 885.81 examples/s]35162 examples [00:40, 875.36 examples/s]35252 examples [00:40, 882.01 examples/s]35343 examples [00:40, 888.20 examples/s]35432 examples [00:40, 872.95 examples/s]35521 examples [00:40, 877.23 examples/s]35613 examples [00:40, 887.35 examples/s]35707 examples [00:40, 900.34 examples/s]35801 examples [00:40, 910.84 examples/s]35893 examples [00:41, 893.69 examples/s]35985 examples [00:41, 901.24 examples/s]36078 examples [00:41, 908.92 examples/s]36173 examples [00:41, 917.71 examples/s]36267 examples [00:41, 922.20 examples/s]36360 examples [00:41, 918.36 examples/s]36452 examples [00:41, 894.07 examples/s]36545 examples [00:41, 902.76 examples/s]36638 examples [00:41, 906.75 examples/s]36732 examples [00:41, 915.30 examples/s]36824 examples [00:42, 911.85 examples/s]36916 examples [00:42, 882.46 examples/s]37007 examples [00:42, 889.92 examples/s]37100 examples [00:42, 901.11 examples/s]37192 examples [00:42, 904.22 examples/s]37283 examples [00:42, 884.44 examples/s]37372 examples [00:42, 875.66 examples/s]37460 examples [00:42, 865.51 examples/s]37553 examples [00:42, 882.30 examples/s]37645 examples [00:42, 892.06 examples/s]37736 examples [00:43, 895.89 examples/s]37826 examples [00:43, 895.85 examples/s]37917 examples [00:43, 899.22 examples/s]38012 examples [00:43, 911.83 examples/s]38104 examples [00:43, 910.80 examples/s]38196 examples [00:43, 897.61 examples/s]38288 examples [00:43, 903.48 examples/s]38379 examples [00:43, 886.03 examples/s]38470 examples [00:43, 887.89 examples/s]38559 examples [00:43, 878.94 examples/s]38647 examples [00:44, 843.63 examples/s]38737 examples [00:44, 857.88 examples/s]38832 examples [00:44, 882.08 examples/s]38921 examples [00:44, 873.01 examples/s]39010 examples [00:44, 875.67 examples/s]39098 examples [00:44, 876.76 examples/s]39188 examples [00:44, 883.59 examples/s]39277 examples [00:44, 885.17 examples/s]39368 examples [00:44, 891.51 examples/s]39458 examples [00:45, 884.11 examples/s]39547 examples [00:45, 882.53 examples/s]39642 examples [00:45, 900.39 examples/s]39737 examples [00:45, 913.21 examples/s]39833 examples [00:45, 924.69 examples/s]39926 examples [00:45, 907.03 examples/s]40017 examples [00:45, 852.78 examples/s]40109 examples [00:45, 869.88 examples/s]40197 examples [00:45, 871.65 examples/s]40287 examples [00:45, 878.97 examples/s]40381 examples [00:46, 894.44 examples/s]40471 examples [00:46, 885.05 examples/s]40564 examples [00:46, 897.41 examples/s]40657 examples [00:46, 905.65 examples/s]40750 examples [00:46, 911.43 examples/s]40842 examples [00:46, 880.07 examples/s]40931 examples [00:46, 880.35 examples/s]41020 examples [00:46, 882.88 examples/s]41110 examples [00:46, 886.60 examples/s]41200 examples [00:46, 890.29 examples/s]41290 examples [00:47, 854.63 examples/s]41377 examples [00:47, 857.79 examples/s]41464 examples [00:47, 859.92 examples/s]41555 examples [00:47, 872.09 examples/s]41649 examples [00:47, 891.09 examples/s]41743 examples [00:47, 903.55 examples/s]41834 examples [00:47, 899.94 examples/s]41926 examples [00:47, 904.80 examples/s]42017 examples [00:47, 888.37 examples/s]42106 examples [00:48, 849.74 examples/s]42197 examples [00:48, 864.60 examples/s]42285 examples [00:48, 866.84 examples/s]42372 examples [00:48, 856.51 examples/s]42458 examples [00:48, 850.06 examples/s]42548 examples [00:48, 863.66 examples/s]42639 examples [00:48, 876.64 examples/s]42728 examples [00:48, 879.04 examples/s]42817 examples [00:48, 863.04 examples/s]42912 examples [00:48, 885.74 examples/s]43002 examples [00:49, 888.74 examples/s]43092 examples [00:49, 883.83 examples/s]43181 examples [00:49, 878.94 examples/s]43275 examples [00:49, 895.50 examples/s]43365 examples [00:49, 890.39 examples/s]43455 examples [00:49, 883.14 examples/s]43544 examples [00:49, 874.20 examples/s]43632 examples [00:49, 867.29 examples/s]43726 examples [00:49, 887.84 examples/s]43815 examples [00:49, 880.95 examples/s]43908 examples [00:50, 893.05 examples/s]43998 examples [00:50, 873.41 examples/s]44095 examples [00:50, 899.43 examples/s]44186 examples [00:50, 898.40 examples/s]44281 examples [00:50, 912.97 examples/s]44373 examples [00:50, 898.22 examples/s]44464 examples [00:50, 890.26 examples/s]44559 examples [00:50, 906.56 examples/s]44654 examples [00:50, 917.94 examples/s]44746 examples [00:50, 918.13 examples/s]44838 examples [00:51, 902.48 examples/s]44929 examples [00:51, 895.14 examples/s]45022 examples [00:51, 905.24 examples/s]45116 examples [00:51, 913.22 examples/s]45210 examples [00:51, 919.23 examples/s]45302 examples [00:51, 891.51 examples/s]45392 examples [00:51, 878.85 examples/s]45482 examples [00:51, 885.04 examples/s]45573 examples [00:51, 890.16 examples/s]45663 examples [00:51, 891.19 examples/s]45753 examples [00:52, 886.94 examples/s]45846 examples [00:52, 897.14 examples/s]45936 examples [00:52, 886.16 examples/s]46029 examples [00:52, 897.78 examples/s]46119 examples [00:52, 871.18 examples/s]46211 examples [00:52, 882.96 examples/s]46300 examples [00:52, 879.80 examples/s]46389 examples [00:52, 879.61 examples/s]46484 examples [00:52, 898.13 examples/s]46577 examples [00:53, 905.42 examples/s]46670 examples [00:53, 911.28 examples/s]46762 examples [00:53, 868.74 examples/s]46853 examples [00:53, 878.67 examples/s]46943 examples [00:53, 883.11 examples/s]47036 examples [00:53, 894.69 examples/s]47126 examples [00:53, 892.52 examples/s]47216 examples [00:53, 881.45 examples/s]47305 examples [00:53, 883.77 examples/s]47394 examples [00:53, 881.79 examples/s]47483 examples [00:54, 867.14 examples/s]47570 examples [00:54, 864.06 examples/s]47663 examples [00:54, 882.11 examples/s]47755 examples [00:54, 891.98 examples/s]47846 examples [00:54, 897.06 examples/s]47936 examples [00:54, 887.51 examples/s]48025 examples [00:54, 863.22 examples/s]48116 examples [00:54, 876.42 examples/s]48212 examples [00:54, 897.23 examples/s]48302 examples [00:54, 875.24 examples/s]48390 examples [00:55, 819.81 examples/s]48479 examples [00:55, 838.24 examples/s]48572 examples [00:55, 862.94 examples/s]48666 examples [00:55, 882.90 examples/s]48762 examples [00:55, 893.57 examples/s]48853 examples [00:55, 898.31 examples/s]48944 examples [00:55, 876.66 examples/s]49038 examples [00:55, 893.66 examples/s]49128 examples [00:55, 858.93 examples/s]49219 examples [00:56, 872.20 examples/s]49312 examples [00:56, 886.59 examples/s]49402 examples [00:56, 863.52 examples/s]49495 examples [00:56, 880.58 examples/s]49591 examples [00:56, 902.72 examples/s]49682 examples [00:56, 899.15 examples/s]49775 examples [00:56, 908.09 examples/s]49867 examples [00:56, 898.39 examples/s]49959 examples [00:56, 903.34 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6310/50000 [00:00<00:00, 63097.20 examples/s] 34%|███▍      | 17235/50000 [00:00<00:00, 72253.75 examples/s] 56%|█████▌    | 27910/50000 [00:00<00:00, 80009.30 examples/s] 78%|███████▊  | 38944/50000 [00:00<00:00, 87199.57 examples/s]                                                               0 examples [00:00, ? examples/s]67 examples [00:00, 665.77 examples/s]158 examples [00:00, 722.97 examples/s]249 examples [00:00, 769.26 examples/s]341 examples [00:00, 808.18 examples/s]434 examples [00:00, 840.37 examples/s]527 examples [00:00, 864.90 examples/s]619 examples [00:00, 877.73 examples/s]711 examples [00:00, 887.84 examples/s]804 examples [00:00, 899.95 examples/s]895 examples [00:01, 902.12 examples/s]989 examples [00:01, 911.18 examples/s]1081 examples [00:01, 912.53 examples/s]1172 examples [00:01, 908.39 examples/s]1264 examples [00:01, 909.24 examples/s]1355 examples [00:01, 900.43 examples/s]1445 examples [00:01, 736.98 examples/s]1537 examples [00:01, 782.30 examples/s]1625 examples [00:01, 808.32 examples/s]1722 examples [00:01, 850.63 examples/s]1810 examples [00:02, 856.16 examples/s]1905 examples [00:02, 880.56 examples/s]1995 examples [00:02, 877.66 examples/s]2087 examples [00:02, 888.63 examples/s]2181 examples [00:02, 902.00 examples/s]2272 examples [00:02, 893.18 examples/s]2362 examples [00:02, 885.15 examples/s]2451 examples [00:02, 880.92 examples/s]2540 examples [00:02, 872.81 examples/s]2629 examples [00:03, 875.34 examples/s]2719 examples [00:03, 879.06 examples/s]2809 examples [00:03, 884.70 examples/s]2902 examples [00:03, 895.51 examples/s]2995 examples [00:03, 905.10 examples/s]3090 examples [00:03, 917.05 examples/s]3183 examples [00:03, 918.16 examples/s]3275 examples [00:03, 893.21 examples/s]3366 examples [00:03, 895.15 examples/s]3456 examples [00:03, 891.55 examples/s]3550 examples [00:04, 904.75 examples/s]3646 examples [00:04, 918.79 examples/s]3739 examples [00:04, 915.15 examples/s]3831 examples [00:04, 914.90 examples/s]3923 examples [00:04, 906.26 examples/s]4014 examples [00:04, 901.44 examples/s]4108 examples [00:04, 910.08 examples/s]4200 examples [00:04, 903.57 examples/s]4292 examples [00:04, 907.00 examples/s]4383 examples [00:04, 875.51 examples/s]4476 examples [00:05, 890.54 examples/s]4566 examples [00:05, 884.81 examples/s]4655 examples [00:05, 875.79 examples/s]4747 examples [00:05, 887.13 examples/s]4839 examples [00:05, 895.37 examples/s]4933 examples [00:05, 907.19 examples/s]5028 examples [00:05, 915.36 examples/s]5120 examples [00:05, 910.88 examples/s]5212 examples [00:05, 893.34 examples/s]5302 examples [00:05, 893.42 examples/s]5392 examples [00:06, 886.05 examples/s]5482 examples [00:06, 889.04 examples/s]5574 examples [00:06, 897.70 examples/s]5665 examples [00:06, 898.77 examples/s]5755 examples [00:06, 890.88 examples/s]5846 examples [00:06, 894.34 examples/s]5936 examples [00:06, 893.52 examples/s]6031 examples [00:06, 909.57 examples/s]6123 examples [00:06, 881.25 examples/s]6217 examples [00:06, 896.23 examples/s]6307 examples [00:07, 877.84 examples/s]6401 examples [00:07, 895.59 examples/s]6492 examples [00:07, 899.07 examples/s]6584 examples [00:07, 902.63 examples/s]6677 examples [00:07, 910.12 examples/s]6769 examples [00:07, 912.21 examples/s]6861 examples [00:07, 894.49 examples/s]6951 examples [00:07, 889.69 examples/s]7041 examples [00:07, 858.48 examples/s]7128 examples [00:08, 849.51 examples/s]7214 examples [00:08, 847.66 examples/s]7304 examples [00:08, 861.50 examples/s]7398 examples [00:08, 882.82 examples/s]7488 examples [00:08, 885.56 examples/s]7583 examples [00:08, 903.78 examples/s]7678 examples [00:08, 915.48 examples/s]7773 examples [00:08, 923.77 examples/s]7866 examples [00:08, 925.32 examples/s]7959 examples [00:08, 912.19 examples/s]8051 examples [00:09, 909.54 examples/s]8144 examples [00:09, 900.43 examples/s]8235 examples [00:09, 888.16 examples/s]8324 examples [00:09, 887.54 examples/s]8413 examples [00:09, 887.14 examples/s]8504 examples [00:09, 892.49 examples/s]8598 examples [00:09, 905.53 examples/s]8689 examples [00:09, 887.79 examples/s]8782 examples [00:09, 898.26 examples/s]8873 examples [00:09, 900.99 examples/s]8968 examples [00:10, 915.03 examples/s]9060 examples [00:10, 890.44 examples/s]9150 examples [00:10, 886.02 examples/s]9240 examples [00:10, 887.32 examples/s]9329 examples [00:10, 887.53 examples/s]9420 examples [00:10, 893.15 examples/s]9513 examples [00:10, 902.12 examples/s]9607 examples [00:10, 910.58 examples/s]9700 examples [00:10, 915.25 examples/s]9792 examples [00:10, 898.05 examples/s]9882 examples [00:11, 884.99 examples/s]9975 examples [00:11, 896.42 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4AH0TZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4AH0TZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f93230fb8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f93230fb8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f93230fb8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f93001b5fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f932307ecf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f93230fb8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f93230fb8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f93230fb8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f938f599dd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9305209fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f92f80c9730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f92f80c9730> 

  function with postional parmater data_info <function split_train_valid at 0x7f92f80c9730> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f92f80c9840> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f92f80c9840> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f92f80c9840> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=91f3a8396a747d92e5a324d66103926b2bd352bc21ef4bb839aff31fe43fbf6e
  Stored in directory: /tmp/pip-ephem-wheel-cache-r9s80gyv/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:01:25, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:20:56, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:29:52, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:03:23, 29.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:37:29, 42.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.10M/862M [00:01<3:54:49, 60.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:43:55, 86.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:54:05, 123kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:19:24, 176kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<55:18, 251kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:02<38:32, 358kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:02<26:53, 509kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:02<18:52, 723kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:02<13:13, 1.03MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<09:44, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<08:42, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<07:58, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<06:02, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:45, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:31, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<04:57, 2.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:02, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:39, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<04:31, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<12:01, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<7:16:10, 30.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<5:05:12, 43.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<3:35:28, 61.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<2:33:51, 85.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<1:48:16, 121kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:12<1:15:43, 173kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<1:00:29, 217kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<43:26, 301kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<30:40, 426kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<24:31, 531kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<19:48, 658kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<14:27, 900kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<12:12, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<09:51, 1.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<07:12, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<08:03, 1.60MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<07:01, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<05:15, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<06:31, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<07:24, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:46, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:22<04:10, 3.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<10:50, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<08:57, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<06:36, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:25, 1.71MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:00, 1.59MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:09, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:30, 2.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:43, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:46, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:01, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:18, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:04, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:37, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:05, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<21:58, 568kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<16:43, 745kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:59, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:06, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:31, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:56, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:13, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:29, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:15, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:05, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:59, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:20, 2.81MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<19:17, 632kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<14:37, 834kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<10:33, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:02, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:35, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:18, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:15, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<18:27, 653kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<14:12, 848kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<10:14, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:46, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:22, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:12, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<05:10, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<20:36, 578kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:42, 758kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<11:15, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<10:29, 1.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:37, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:19, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:00, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:30, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:49, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:13, 2.77MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<17:36, 665kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:34, 862kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:46, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:23, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:49, 1.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:45, 2.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:34, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:11, 1.61MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:36, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<04:04, 2.83MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<19:16, 597kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<14:44, 780kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<10:33, 1.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:53, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:28, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:14, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<05:12, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<21:30, 529kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<16:04, 706kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<11:32, 982kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<10:33, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:36, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:19, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:55, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:13, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:37, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:04, 2.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<16:28, 677kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<12:43, 876kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<09:09, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:51, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:38, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:33, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:28, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:45, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:19, 2.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:25, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:47, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:44, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:53, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:43, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:36, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<03:21, 3.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<18:47, 572kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<14:20, 749kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<10:15, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:30, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<09:01, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:49, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:52, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<11:22, 932kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<09:06, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:36, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:57, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:26, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:07, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:31, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<18:48, 552kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<14:15, 728kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<10:14, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:27, 1.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:49, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:39, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:45, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<13:23, 765kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<10:29, 977kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<07:36, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:34, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:33, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:45, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:38<04:08, 2.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:11, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:47, 1.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:58, 2.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:46, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:07, 1.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:50, 2.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:24, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<6:12:13, 26.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<4:20:16, 38.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<3:03:17, 54.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<2:10:21, 76.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<1:31:38, 108kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<1:04:10, 154kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<47:37, 207kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<34:21, 286kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<24:14, 405kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<19:09, 510kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<16:34, 590kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:13, 799kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<08:39, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<12:50, 756kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<10:18, 941kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:29, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:14, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:38, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:04, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<13:15, 722kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<11:27, 835kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<08:39, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<06:10, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<14:27, 657kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:04, 857kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:20, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<05:55, 1.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<15:38, 603kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<12:10, 775kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<08:42, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<06:14, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<16:05, 582kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<12:24, 754kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<08:53, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<06:20, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<31:18, 297kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<22:54, 405kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<16:14, 570kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<13:22, 690kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<11:19, 814kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:23, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<05:57, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<1:09:31, 132kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<49:37, 184kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<34:52, 262kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<26:18, 345kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<20:20, 447kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<14:42, 617kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<10:21, 871kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<21:04, 428kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<15:42, 573kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<11:11, 803kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<09:45, 916kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<08:49, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:36, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<04:41, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<11:14, 790kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:50, 1.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:23, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:24, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:21, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:51, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:30, 2.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:10, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:16, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:54, 2.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:38, 1.87MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:04, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:01, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:54, 2.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<15:13, 565kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<11:34, 742kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<08:18, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<07:41, 1.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<07:13, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:27, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<15:28, 547kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<11:44, 721kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:24, 1.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:45, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:20, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:37, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:04, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:24, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:09, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:00, 2.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:59, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:03, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:44, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:29, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:53, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:49, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:47, 2.92MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:19, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:36, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:26, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:10, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:37, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:39, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:38, 3.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<18:23, 434kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<13:43, 581kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:45, 814kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<08:32, 926kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<07:38, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:42, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<04:03, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<10:25, 752kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:00, 978kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:52, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:11, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<10:29, 741kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<08:08, 953kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:52, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:53, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:49, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:27, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:46, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:10, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:07, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:50, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:28, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<02:35, 2.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:32, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:09, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:22, 3.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<01:46, 4.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:19, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:55, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:18, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:31, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<4:48:17, 25.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<3:21:24, 36.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<2:21:30, 51.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:40:48, 72.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:10:54, 103kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<49:26, 146kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<39:10, 184kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<28:13, 256kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<19:52, 362kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<15:22, 466kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<12:20, 580kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<09:00, 793kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<06:20, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<15:08, 468kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<11:11, 632kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<07:59, 882kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:11, 976kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:31, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:53, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<03:28, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<09:37, 722kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:27, 930kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:23, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:20, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:17, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:02, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:54, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<11:17, 603kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<08:39, 787kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<06:13, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:49, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:30, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:10, 1.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:59, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:29, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:33, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:20, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:47, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:13, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:25, 2.71MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:12, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:55, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:12, 2.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:01, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:34, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:47, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:05, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:07, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:52, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:10, 2.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:55, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:42, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:01, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:52, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:42, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<01:57, 3.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<11:06, 557kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<08:26, 732kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<06:01, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:32, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:14, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:57, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:33, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:50, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:50, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:18, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:38, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:51, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:03, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<10:05, 585kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<07:42, 766kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:30, 1.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:06, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:02, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:49, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:48, 2.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:18, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:01, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:17, 2.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:50, 2.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:01, 2.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:38, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:25, 2.31MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<01:45, 3.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<09:45, 570kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:24, 749kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<05:18, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<04:56, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:37, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:29, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:29, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:24, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:21, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:09, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:24, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:35, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:45, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:59, 2.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:51, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:16, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:25, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:50, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:09, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:27, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:47, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:50, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:33, 2.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:54, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:26, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:47, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:13, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:36, 3.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<05:03, 991kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:04, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:57, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:09, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:12, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:29, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:47, 2.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<06:56, 702kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:22, 905kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:51, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:45, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:37, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:44, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:58, 2.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:43, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:06, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:16, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:54, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<3:02:31, 25.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<2:07:18, 36.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<1:29:06, 51.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<1:03:24, 72.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<44:32, 103kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<30:55, 147kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<25:08, 180kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<18:04, 251kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<12:41, 355kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<09:47, 456kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<07:19, 608kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:13, 849kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:35, 956kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:08, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:07, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:13, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<14:35, 297kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<10:39, 405kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<07:32, 570kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:09, 691kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:45, 894kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<03:25, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:20, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:13, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:28, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:45, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:59, 590kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<05:19, 772kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:49, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:32, 1.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:20, 1.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:31, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:47, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<06:42, 593kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:07, 775kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:40, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:24, 1.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:15, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:27, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:44, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:46, 806kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:44, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:42, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:43, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:42, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:05, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:29, 2.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<06:43, 551kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<05:05, 726kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:37, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:20, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:06, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:21, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:40, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<12:27, 286kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<09:05, 392kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<06:23, 553kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<05:13, 670kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:23, 796kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:13, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:17, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:56, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:25, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:46, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:58, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:05, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:10, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:53, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:40, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:14, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:34, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:26, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:04, 2.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:28, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:41, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:19, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<00:58, 3.19MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:39, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:29, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:06, 2.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:27, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:20, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:00, 2.98MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:20, 2.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:14, 2.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:56, 3.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:18, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:33, 1.85MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:13, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:53, 3.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:28, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:04, 2.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:46, 3.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:27, 1.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:59, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:27, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:37, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:02, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:21, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:31, 1.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:51, 2.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<04:15, 594kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<03:14, 781kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:18, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:09, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:10, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:39, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:11, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:26, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:37, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:16, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:54, 2.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:58, 2.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<1:29:58, 25.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<1:02:50, 36.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<43:15, 52.7kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<31:08, 72.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<23:01, 98.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<16:21, 138kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<11:24, 196kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<08:15, 266kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<06:19, 347kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:31, 481kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<03:08, 680kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:24, 623kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:55, 724kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:09, 977kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<01:31, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:42, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:40, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:16, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:56, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:47, 1.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:21, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:57, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:57, 982kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:47, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:21, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:56, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:51, 991kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:43, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:17, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:54, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:44, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:35, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:11, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:50, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:30, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:29, 1.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:07, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:48, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:06, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:07, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:50, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:36, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:05, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:50, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:36, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:53, 1.69MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:56, 1.59MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:43, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:30, 2.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:31, 942kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:25, 999kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:04, 1.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:45, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:14, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:13, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:55, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:39, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:51, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:51, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:39, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:27, 2.68MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:46, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:48, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:26, 2.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:25, 813kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:14, 931kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:54, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:38, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:47, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:46, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:24, 2.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:32, 664kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:16, 794kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:55, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:38, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:20, 710kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:08, 834kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:49, 1.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:33, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:58, 899kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:53, 978kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:40, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:28, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:32, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:17, 2.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:22, 540kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:06, 666kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:47, 911kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:32, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:38, 1.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:35, 1.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:17, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<01:06, 551kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:54, 660kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:39, 899kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:26, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:30, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:28, 1.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:14, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:25, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:24, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:20, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:19, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:15, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:09, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:16, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:15, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:07, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:14, 1.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:14, 1.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:06, 2.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.68MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<21:12:14,  5.24it/s]  0%|          | 746/400000 [00:00<14:49:10,  7.48it/s]  0%|          | 1496/400000 [00:00<10:21:31, 10.69it/s]  1%|          | 2187/400000 [00:00<7:14:35, 15.26it/s]   1%|          | 2904/400000 [00:00<5:03:56, 21.77it/s]  1%|          | 3606/400000 [00:00<3:32:40, 31.07it/s]  1%|          | 4294/400000 [00:00<2:28:53, 44.29it/s]  1%|          | 4992/400000 [00:00<1:44:19, 63.10it/s]  1%|▏         | 5683/400000 [00:00<1:13:11, 89.80it/s]  2%|▏         | 6402/400000 [00:01<51:24, 127.60it/s]   2%|▏         | 7079/400000 [00:01<36:14, 180.67it/s]  2%|▏         | 7764/400000 [00:01<25:36, 255.22it/s]  2%|▏         | 8430/400000 [00:01<18:11, 358.64it/s]  2%|▏         | 9132/400000 [00:01<12:59, 501.36it/s]  2%|▏         | 9858/400000 [00:01<09:20, 695.62it/s]  3%|▎         | 10555/400000 [00:01<06:48, 952.96it/s]  3%|▎         | 11259/400000 [00:01<05:02, 1286.68it/s]  3%|▎         | 11998/400000 [00:01<03:46, 1710.35it/s]  3%|▎         | 12734/400000 [00:02<02:54, 2222.00it/s]  3%|▎         | 13451/400000 [00:02<02:19, 2770.77it/s]  4%|▎         | 14149/400000 [00:02<01:54, 3381.09it/s]  4%|▎         | 14881/400000 [00:02<01:35, 4031.55it/s]  4%|▍         | 15589/400000 [00:02<01:23, 4586.16it/s]  4%|▍         | 16287/400000 [00:02<01:15, 5108.40it/s]  4%|▍         | 16985/400000 [00:02<01:09, 5547.04it/s]  4%|▍         | 17691/400000 [00:02<01:04, 5927.02it/s]  5%|▍         | 18438/400000 [00:02<01:00, 6317.19it/s]  5%|▍         | 19152/400000 [00:02<00:58, 6527.28it/s]  5%|▍         | 19864/400000 [00:03<00:57, 6610.60it/s]  5%|▌         | 20603/400000 [00:03<00:55, 6825.01it/s]  5%|▌         | 21317/400000 [00:03<00:55, 6833.87it/s]  6%|▌         | 22063/400000 [00:03<00:53, 7009.45it/s]  6%|▌         | 22804/400000 [00:03<00:52, 7122.80it/s]  6%|▌         | 23566/400000 [00:03<00:51, 7263.07it/s]  6%|▌         | 24302/400000 [00:03<00:52, 7110.40it/s]  6%|▋         | 25020/400000 [00:03<00:52, 7130.70it/s]  6%|▋         | 25747/400000 [00:03<00:52, 7171.71it/s]  7%|▋         | 26491/400000 [00:03<00:51, 7247.61it/s]  7%|▋         | 27250/400000 [00:04<00:50, 7346.81it/s]  7%|▋         | 27987/400000 [00:04<00:51, 7214.45it/s]  7%|▋         | 28711/400000 [00:04<00:51, 7153.60it/s]  7%|▋         | 29452/400000 [00:04<00:51, 7227.60it/s]  8%|▊         | 30187/400000 [00:04<00:50, 7262.91it/s]  8%|▊         | 30958/400000 [00:04<00:49, 7390.19it/s]  8%|▊         | 31705/400000 [00:04<00:49, 7410.51it/s]  8%|▊         | 32447/400000 [00:04<00:49, 7374.38it/s]  8%|▊         | 33186/400000 [00:04<00:49, 7337.58it/s]  8%|▊         | 33922/400000 [00:04<00:49, 7342.11it/s]  9%|▊         | 34664/400000 [00:05<00:49, 7364.63it/s]  9%|▉         | 35401/400000 [00:05<00:49, 7360.49it/s]  9%|▉         | 36138/400000 [00:05<00:49, 7304.07it/s]  9%|▉         | 36881/400000 [00:05<00:49, 7340.08it/s]  9%|▉         | 37616/400000 [00:05<00:50, 7208.74it/s] 10%|▉         | 38354/400000 [00:05<00:49, 7259.04it/s] 10%|▉         | 39107/400000 [00:05<00:49, 7335.27it/s] 10%|▉         | 39845/400000 [00:05<00:49, 7347.06it/s] 10%|█         | 40585/400000 [00:05<00:48, 7361.02it/s] 10%|█         | 41342/400000 [00:05<00:48, 7421.17it/s] 11%|█         | 42085/400000 [00:06<00:48, 7349.57it/s] 11%|█         | 42821/400000 [00:06<00:49, 7253.19it/s] 11%|█         | 43547/400000 [00:06<00:49, 7158.06it/s] 11%|█         | 44273/400000 [00:06<00:49, 7187.97it/s] 11%|█▏        | 45030/400000 [00:06<00:48, 7297.68it/s] 11%|█▏        | 45762/400000 [00:06<00:48, 7304.04it/s] 12%|█▏        | 46493/400000 [00:06<00:48, 7298.06it/s] 12%|█▏        | 47224/400000 [00:06<00:48, 7221.86it/s] 12%|█▏        | 47947/400000 [00:06<00:50, 6980.46it/s] 12%|█▏        | 48682/400000 [00:06<00:49, 7086.35it/s] 12%|█▏        | 49416/400000 [00:07<00:48, 7157.78it/s] 13%|█▎        | 50134/400000 [00:07<00:49, 7046.87it/s] 13%|█▎        | 50841/400000 [00:07<00:49, 7020.33it/s] 13%|█▎        | 51545/400000 [00:07<00:49, 6993.80it/s] 13%|█▎        | 52291/400000 [00:07<00:48, 7126.69it/s] 13%|█▎        | 53039/400000 [00:07<00:48, 7227.65it/s] 13%|█▎        | 53763/400000 [00:07<00:49, 7020.56it/s] 14%|█▎        | 54468/400000 [00:07<00:51, 6766.64it/s] 14%|█▍        | 55200/400000 [00:07<00:49, 6921.53it/s] 14%|█▍        | 55926/400000 [00:08<00:49, 7018.28it/s] 14%|█▍        | 56681/400000 [00:08<00:47, 7169.66it/s] 14%|█▍        | 57401/400000 [00:08<00:48, 7092.40it/s] 15%|█▍        | 58123/400000 [00:08<00:47, 7128.20it/s] 15%|█▍        | 58864/400000 [00:08<00:47, 7210.31it/s] 15%|█▍        | 59619/400000 [00:08<00:46, 7308.20it/s] 15%|█▌        | 60387/400000 [00:08<00:45, 7414.58it/s] 15%|█▌        | 61157/400000 [00:08<00:45, 7496.31it/s] 15%|█▌        | 61914/400000 [00:08<00:44, 7516.41it/s] 16%|█▌        | 62685/400000 [00:08<00:44, 7573.42it/s] 16%|█▌        | 63443/400000 [00:09<00:44, 7532.06it/s] 16%|█▌        | 64197/400000 [00:09<00:44, 7511.72it/s] 16%|█▌        | 64949/400000 [00:09<00:44, 7506.88it/s] 16%|█▋        | 65700/400000 [00:09<00:44, 7495.35it/s] 17%|█▋        | 66450/400000 [00:09<00:44, 7470.73it/s] 17%|█▋        | 67198/400000 [00:09<00:46, 7161.08it/s] 17%|█▋        | 67917/400000 [00:09<00:47, 6999.50it/s] 17%|█▋        | 68661/400000 [00:09<00:46, 7124.69it/s] 17%|█▋        | 69384/400000 [00:09<00:46, 7154.12it/s] 18%|█▊        | 70102/400000 [00:09<00:46, 7092.18it/s] 18%|█▊        | 70843/400000 [00:10<00:45, 7184.22it/s] 18%|█▊        | 71577/400000 [00:10<00:45, 7228.19it/s] 18%|█▊        | 72301/400000 [00:10<00:45, 7143.74it/s] 18%|█▊        | 73047/400000 [00:10<00:45, 7233.73it/s] 18%|█▊        | 73830/400000 [00:10<00:44, 7402.75it/s] 19%|█▊        | 74572/400000 [00:10<00:44, 7364.76it/s] 19%|█▉        | 75337/400000 [00:10<00:43, 7447.99it/s] 19%|█▉        | 76113/400000 [00:10<00:42, 7537.77it/s] 19%|█▉        | 76868/400000 [00:10<00:43, 7469.28it/s] 19%|█▉        | 77665/400000 [00:10<00:42, 7609.80it/s] 20%|█▉        | 78430/400000 [00:11<00:42, 7620.22it/s] 20%|█▉        | 79193/400000 [00:11<00:42, 7602.91it/s] 20%|█▉        | 79954/400000 [00:11<00:42, 7552.97it/s] 20%|██        | 80710/400000 [00:11<00:42, 7505.45it/s] 20%|██        | 81461/400000 [00:11<00:42, 7490.06it/s] 21%|██        | 82211/400000 [00:11<00:44, 7102.73it/s] 21%|██        | 82980/400000 [00:11<00:43, 7267.54it/s] 21%|██        | 83761/400000 [00:11<00:42, 7419.65it/s] 21%|██        | 84507/400000 [00:11<00:42, 7425.38it/s] 21%|██▏       | 85253/400000 [00:11<00:42, 7433.95it/s] 22%|██▏       | 86004/400000 [00:12<00:42, 7455.62it/s] 22%|██▏       | 86751/400000 [00:12<00:42, 7443.39it/s] 22%|██▏       | 87527/400000 [00:12<00:41, 7532.73it/s] 22%|██▏       | 88298/400000 [00:12<00:41, 7580.65it/s] 22%|██▏       | 89065/400000 [00:12<00:40, 7606.29it/s] 22%|██▏       | 89827/400000 [00:12<00:40, 7605.07it/s] 23%|██▎       | 90588/400000 [00:12<00:41, 7545.33it/s] 23%|██▎       | 91395/400000 [00:12<00:40, 7694.36it/s] 23%|██▎       | 92166/400000 [00:12<00:40, 7578.06it/s] 23%|██▎       | 92925/400000 [00:13<00:40, 7571.16it/s] 23%|██▎       | 93683/400000 [00:13<00:40, 7522.43it/s] 24%|██▎       | 94450/400000 [00:13<00:40, 7565.46it/s] 24%|██▍       | 95208/400000 [00:13<00:41, 7368.42it/s] 24%|██▍       | 95947/400000 [00:13<00:42, 7169.92it/s] 24%|██▍       | 96728/400000 [00:13<00:41, 7350.60it/s] 24%|██▍       | 97522/400000 [00:13<00:40, 7517.34it/s] 25%|██▍       | 98277/400000 [00:13<00:40, 7496.20it/s] 25%|██▍       | 99045/400000 [00:13<00:39, 7547.97it/s] 25%|██▍       | 99817/400000 [00:13<00:39, 7596.53it/s] 25%|██▌       | 100600/400000 [00:14<00:39, 7665.00it/s] 25%|██▌       | 101379/400000 [00:14<00:38, 7701.60it/s] 26%|██▌       | 102153/400000 [00:14<00:38, 7711.62it/s] 26%|██▌       | 102925/400000 [00:14<00:39, 7613.71it/s] 26%|██▌       | 103687/400000 [00:14<00:39, 7544.69it/s] 26%|██▌       | 104447/400000 [00:14<00:39, 7560.01it/s] 26%|██▋       | 105238/400000 [00:14<00:38, 7661.37it/s] 27%|██▋       | 106012/400000 [00:14<00:38, 7683.03it/s] 27%|██▋       | 106781/400000 [00:14<00:38, 7641.37it/s] 27%|██▋       | 107546/400000 [00:14<00:38, 7637.00it/s] 27%|██▋       | 108310/400000 [00:15<00:38, 7587.13it/s] 27%|██▋       | 109077/400000 [00:15<00:38, 7610.12it/s] 27%|██▋       | 109852/400000 [00:15<00:37, 7649.88it/s] 28%|██▊       | 110618/400000 [00:15<00:39, 7395.98it/s] 28%|██▊       | 111360/400000 [00:15<00:39, 7262.06it/s] 28%|██▊       | 112089/400000 [00:15<00:39, 7260.68it/s] 28%|██▊       | 112882/400000 [00:15<00:38, 7449.06it/s] 28%|██▊       | 113654/400000 [00:15<00:38, 7527.85it/s] 29%|██▊       | 114444/400000 [00:15<00:37, 7634.57it/s] 29%|██▉       | 115210/400000 [00:15<00:38, 7485.81it/s] 29%|██▉       | 115961/400000 [00:16<00:38, 7445.50it/s] 29%|██▉       | 116707/400000 [00:16<00:38, 7406.01it/s] 29%|██▉       | 117459/400000 [00:16<00:37, 7438.88it/s] 30%|██▉       | 118223/400000 [00:16<00:37, 7495.70it/s] 30%|██▉       | 118982/400000 [00:16<00:37, 7522.92it/s] 30%|██▉       | 119747/400000 [00:16<00:37, 7556.83it/s] 30%|███       | 120504/400000 [00:16<00:37, 7516.40it/s] 30%|███       | 121269/400000 [00:16<00:36, 7554.68it/s] 31%|███       | 122047/400000 [00:16<00:36, 7619.32it/s] 31%|███       | 122813/400000 [00:16<00:36, 7628.65it/s] 31%|███       | 123582/400000 [00:17<00:36, 7645.96it/s] 31%|███       | 124347/400000 [00:17<00:36, 7546.63it/s] 31%|███▏      | 125127/400000 [00:17<00:36, 7619.05it/s] 31%|███▏      | 125926/400000 [00:17<00:35, 7724.63it/s] 32%|███▏      | 126700/400000 [00:17<00:36, 7568.78it/s] 32%|███▏      | 127461/400000 [00:17<00:35, 7576.54it/s] 32%|███▏      | 128220/400000 [00:17<00:36, 7497.63it/s] 32%|███▏      | 128983/400000 [00:17<00:35, 7536.06it/s] 32%|███▏      | 129761/400000 [00:17<00:35, 7607.34it/s] 33%|███▎      | 130523/400000 [00:17<00:35, 7546.03it/s] 33%|███▎      | 131309/400000 [00:18<00:35, 7636.08it/s] 33%|███▎      | 132099/400000 [00:18<00:34, 7712.48it/s] 33%|███▎      | 132871/400000 [00:18<00:35, 7503.43it/s] 33%|███▎      | 133663/400000 [00:18<00:34, 7622.67it/s] 34%|███▎      | 134427/400000 [00:18<00:35, 7449.19it/s] 34%|███▍      | 135229/400000 [00:18<00:34, 7610.46it/s] 34%|███▍      | 136008/400000 [00:18<00:34, 7662.05it/s] 34%|███▍      | 136776/400000 [00:18<00:34, 7522.96it/s] 34%|███▍      | 137550/400000 [00:18<00:34, 7586.51it/s] 35%|███▍      | 138310/400000 [00:19<00:35, 7447.25it/s] 35%|███▍      | 139057/400000 [00:19<00:35, 7290.76it/s] 35%|███▍      | 139788/400000 [00:19<00:35, 7267.80it/s] 35%|███▌      | 140522/400000 [00:19<00:35, 7286.63it/s] 35%|███▌      | 141252/400000 [00:19<00:35, 7226.13it/s] 35%|███▌      | 141976/400000 [00:19<00:36, 7019.69it/s] 36%|███▌      | 142753/400000 [00:19<00:35, 7228.90it/s] 36%|███▌      | 143513/400000 [00:19<00:34, 7335.66it/s] 36%|███▌      | 144295/400000 [00:19<00:34, 7472.14it/s] 36%|███▋      | 145071/400000 [00:19<00:33, 7555.20it/s] 36%|███▋      | 145829/400000 [00:20<00:34, 7305.34it/s] 37%|███▋      | 146563/400000 [00:20<00:35, 7067.06it/s] 37%|███▋      | 147300/400000 [00:20<00:35, 7153.51it/s] 37%|███▋      | 148056/400000 [00:20<00:34, 7269.29it/s] 37%|███▋      | 148825/400000 [00:20<00:33, 7390.46it/s] 37%|███▋      | 149567/400000 [00:20<00:34, 7331.08it/s] 38%|███▊      | 150302/400000 [00:20<00:34, 7323.29it/s] 38%|███▊      | 151036/400000 [00:20<00:34, 7290.09it/s] 38%|███▊      | 151789/400000 [00:20<00:33, 7359.41it/s] 38%|███▊      | 152565/400000 [00:20<00:33, 7474.98it/s] 38%|███▊      | 153320/400000 [00:21<00:32, 7496.78it/s] 39%|███▊      | 154071/400000 [00:21<00:33, 7350.56it/s] 39%|███▊      | 154808/400000 [00:21<00:33, 7355.44it/s] 39%|███▉      | 155578/400000 [00:21<00:32, 7452.95it/s] 39%|███▉      | 156329/400000 [00:21<00:32, 7467.94it/s] 39%|███▉      | 157077/400000 [00:21<00:33, 7333.41it/s] 39%|███▉      | 157845/400000 [00:21<00:32, 7433.16it/s] 40%|███▉      | 158590/400000 [00:21<00:33, 7280.72it/s] 40%|███▉      | 159353/400000 [00:21<00:32, 7379.33it/s] 40%|████      | 160127/400000 [00:21<00:32, 7483.50it/s] 40%|████      | 160877/400000 [00:22<00:32, 7329.59it/s] 40%|████      | 161612/400000 [00:22<00:33, 7129.89it/s] 41%|████      | 162350/400000 [00:22<00:32, 7202.34it/s] 41%|████      | 163073/400000 [00:22<00:34, 6938.27it/s] 41%|████      | 163771/400000 [00:22<00:34, 6947.56it/s] 41%|████      | 164498/400000 [00:22<00:33, 7039.69it/s] 41%|████▏     | 165204/400000 [00:22<00:33, 7006.40it/s] 41%|████▏     | 165949/400000 [00:22<00:32, 7132.97it/s] 42%|████▏     | 166688/400000 [00:22<00:32, 7206.55it/s] 42%|████▏     | 167412/400000 [00:23<00:32, 7213.69it/s] 42%|████▏     | 168135/400000 [00:23<00:32, 7054.60it/s] 42%|████▏     | 168871/400000 [00:23<00:32, 7141.53it/s] 42%|████▏     | 169617/400000 [00:23<00:31, 7232.48it/s] 43%|████▎     | 170388/400000 [00:23<00:31, 7368.73it/s] 43%|████▎     | 171127/400000 [00:23<00:31, 7345.87it/s] 43%|████▎     | 171863/400000 [00:23<00:31, 7346.32it/s] 43%|████▎     | 172654/400000 [00:23<00:30, 7505.96it/s] 43%|████▎     | 173406/400000 [00:23<00:30, 7484.33it/s] 44%|████▎     | 174156/400000 [00:23<00:30, 7439.50it/s] 44%|████▎     | 174901/400000 [00:24<00:30, 7411.81it/s] 44%|████▍     | 175643/400000 [00:24<00:31, 7224.79it/s] 44%|████▍     | 176408/400000 [00:24<00:30, 7347.01it/s] 44%|████▍     | 177185/400000 [00:24<00:29, 7467.06it/s] 44%|████▍     | 177946/400000 [00:24<00:29, 7507.73it/s] 45%|████▍     | 178711/400000 [00:24<00:29, 7547.99it/s] 45%|████▍     | 179467/400000 [00:24<00:29, 7478.89it/s] 45%|████▌     | 180234/400000 [00:24<00:29, 7533.32it/s] 45%|████▌     | 181017/400000 [00:24<00:28, 7617.03it/s] 45%|████▌     | 181790/400000 [00:24<00:28, 7650.41it/s] 46%|████▌     | 182556/400000 [00:25<00:28, 7572.19it/s] 46%|████▌     | 183314/400000 [00:25<00:29, 7379.85it/s] 46%|████▌     | 184054/400000 [00:25<00:29, 7385.29it/s] 46%|████▌     | 184829/400000 [00:25<00:28, 7490.01it/s] 46%|████▋     | 185623/400000 [00:25<00:28, 7617.88it/s] 47%|████▋     | 186387/400000 [00:25<00:28, 7558.89it/s] 47%|████▋     | 187144/400000 [00:25<00:28, 7532.12it/s] 47%|████▋     | 187924/400000 [00:25<00:27, 7609.62it/s] 47%|████▋     | 188686/400000 [00:25<00:28, 7512.23it/s] 47%|████▋     | 189438/400000 [00:25<00:28, 7461.45it/s] 48%|████▊     | 190185/400000 [00:26<00:28, 7262.95it/s] 48%|████▊     | 190913/400000 [00:26<00:29, 7203.37it/s] 48%|████▊     | 191637/400000 [00:26<00:28, 7212.18it/s] 48%|████▊     | 192371/400000 [00:26<00:28, 7247.26it/s] 48%|████▊     | 193097/400000 [00:26<00:28, 7225.72it/s] 48%|████▊     | 193835/400000 [00:26<00:28, 7268.56it/s] 49%|████▊     | 194567/400000 [00:26<00:28, 7282.37it/s] 49%|████▉     | 195332/400000 [00:26<00:27, 7386.52it/s] 49%|████▉     | 196088/400000 [00:26<00:27, 7434.62it/s] 49%|████▉     | 196832/400000 [00:26<00:27, 7389.93it/s] 49%|████▉     | 197572/400000 [00:27<00:27, 7387.72it/s] 50%|████▉     | 198312/400000 [00:27<00:27, 7385.86it/s] 50%|████▉     | 199083/400000 [00:27<00:26, 7479.39it/s] 50%|████▉     | 199832/400000 [00:27<00:26, 7469.89it/s] 50%|█████     | 200580/400000 [00:27<00:26, 7393.69it/s] 50%|█████     | 201324/400000 [00:27<00:26, 7404.54it/s] 51%|█████     | 202065/400000 [00:27<00:27, 7319.14it/s] 51%|█████     | 202825/400000 [00:27<00:26, 7399.21it/s] 51%|█████     | 203591/400000 [00:27<00:26, 7472.15it/s] 51%|█████     | 204339/400000 [00:28<00:26, 7411.35it/s] 51%|█████▏    | 205106/400000 [00:28<00:26, 7484.06it/s] 51%|█████▏    | 205855/400000 [00:28<00:26, 7293.51it/s] 52%|█████▏    | 206586/400000 [00:28<00:26, 7235.37it/s] 52%|█████▏    | 207342/400000 [00:28<00:26, 7329.16it/s] 52%|█████▏    | 208076/400000 [00:28<00:26, 7314.86it/s] 52%|█████▏    | 208809/400000 [00:28<00:26, 7266.51it/s] 52%|█████▏    | 209552/400000 [00:28<00:26, 7312.84it/s] 53%|█████▎    | 210320/400000 [00:28<00:25, 7416.14it/s] 53%|█████▎    | 211118/400000 [00:28<00:24, 7574.04it/s] 53%|█████▎    | 211879/400000 [00:29<00:24, 7583.50it/s] 53%|█████▎    | 212639/400000 [00:29<00:25, 7424.79it/s] 53%|█████▎    | 213383/400000 [00:29<00:25, 7381.28it/s] 54%|█████▎    | 214123/400000 [00:29<00:25, 7365.65it/s] 54%|█████▎    | 214865/400000 [00:29<00:25, 7381.84it/s] 54%|█████▍    | 215617/400000 [00:29<00:24, 7421.96it/s] 54%|█████▍    | 216368/400000 [00:29<00:24, 7442.32it/s] 54%|█████▍    | 217113/400000 [00:29<00:24, 7444.31it/s] 54%|█████▍    | 217858/400000 [00:29<00:24, 7408.14it/s] 55%|█████▍    | 218603/400000 [00:29<00:24, 7420.15it/s] 55%|█████▍    | 219379/400000 [00:30<00:24, 7516.62it/s] 55%|█████▌    | 220154/400000 [00:30<00:23, 7584.85it/s] 55%|█████▌    | 220913/400000 [00:30<00:24, 7414.08it/s] 55%|█████▌    | 221656/400000 [00:30<00:24, 7399.55it/s] 56%|█████▌    | 222427/400000 [00:30<00:23, 7488.45it/s] 56%|█████▌    | 223177/400000 [00:30<00:24, 7338.93it/s] 56%|█████▌    | 223936/400000 [00:30<00:23, 7411.26it/s] 56%|█████▌    | 224679/400000 [00:30<00:23, 7345.82it/s] 56%|█████▋    | 225426/400000 [00:30<00:23, 7379.45it/s] 57%|█████▋    | 226165/400000 [00:30<00:23, 7292.39it/s] 57%|█████▋    | 226895/400000 [00:31<00:23, 7262.52it/s] 57%|█████▋    | 227622/400000 [00:31<00:23, 7221.37it/s] 57%|█████▋    | 228354/400000 [00:31<00:23, 7248.36it/s] 57%|█████▋    | 229080/400000 [00:31<00:23, 7204.79it/s] 57%|█████▋    | 229844/400000 [00:31<00:23, 7329.95it/s] 58%|█████▊    | 230595/400000 [00:31<00:22, 7379.58it/s] 58%|█████▊    | 231341/400000 [00:31<00:22, 7399.65it/s] 58%|█████▊    | 232111/400000 [00:31<00:22, 7486.74it/s] 58%|█████▊    | 232893/400000 [00:31<00:22, 7583.39it/s] 58%|█████▊    | 233661/400000 [00:31<00:21, 7612.02it/s] 59%|█████▊    | 234423/400000 [00:32<00:21, 7585.08it/s] 59%|█████▉    | 235182/400000 [00:32<00:21, 7524.40it/s] 59%|█████▉    | 235935/400000 [00:32<00:22, 7426.36it/s] 59%|█████▉    | 236703/400000 [00:32<00:21, 7500.67it/s] 59%|█████▉    | 237454/400000 [00:32<00:21, 7487.85it/s] 60%|█████▉    | 238227/400000 [00:32<00:21, 7556.34it/s] 60%|█████▉    | 238998/400000 [00:32<00:21, 7599.88it/s] 60%|█████▉    | 239759/400000 [00:32<00:22, 7164.21it/s] 60%|██████    | 240523/400000 [00:32<00:21, 7298.19it/s] 60%|██████    | 241326/400000 [00:32<00:21, 7501.98it/s] 61%|██████    | 242096/400000 [00:33<00:20, 7558.38it/s] 61%|██████    | 242860/400000 [00:33<00:20, 7581.80it/s] 61%|██████    | 243621/400000 [00:33<00:20, 7470.31it/s] 61%|██████    | 244371/400000 [00:33<00:20, 7456.52it/s] 61%|██████▏   | 245132/400000 [00:33<00:20, 7499.82it/s] 61%|██████▏   | 245896/400000 [00:33<00:20, 7538.99it/s] 62%|██████▏   | 246666/400000 [00:33<00:20, 7585.87it/s] 62%|██████▏   | 247426/400000 [00:33<00:20, 7521.49it/s] 62%|██████▏   | 248179/400000 [00:33<00:20, 7444.79it/s] 62%|██████▏   | 248925/400000 [00:34<00:20, 7260.77it/s] 62%|██████▏   | 249688/400000 [00:34<00:20, 7367.48it/s] 63%|██████▎   | 250454/400000 [00:34<00:20, 7451.51it/s] 63%|██████▎   | 251218/400000 [00:34<00:19, 7506.66it/s] 63%|██████▎   | 251970/400000 [00:34<00:20, 7222.95it/s] 63%|██████▎   | 252696/400000 [00:34<00:20, 7229.54it/s] 63%|██████▎   | 253422/400000 [00:34<00:20, 7189.28it/s] 64%|██████▎   | 254158/400000 [00:34<00:20, 7238.01it/s] 64%|██████▎   | 254904/400000 [00:34<00:19, 7302.52it/s] 64%|██████▍   | 255653/400000 [00:34<00:19, 7355.45it/s] 64%|██████▍   | 256417/400000 [00:35<00:19, 7436.49it/s] 64%|██████▍   | 257162/400000 [00:35<00:19, 7353.08it/s] 64%|██████▍   | 257899/400000 [00:35<00:19, 7337.42it/s] 65%|██████▍   | 258634/400000 [00:35<00:19, 7335.73it/s] 65%|██████▍   | 259379/400000 [00:35<00:19, 7369.53it/s] 65%|██████▌   | 260117/400000 [00:35<00:19, 7280.43it/s] 65%|██████▌   | 260846/400000 [00:35<00:19, 7273.27it/s] 65%|██████▌   | 261587/400000 [00:35<00:18, 7311.25it/s] 66%|██████▌   | 262319/400000 [00:35<00:19, 7220.25it/s] 66%|██████▌   | 263042/400000 [00:35<00:19, 7099.80it/s] 66%|██████▌   | 263753/400000 [00:36<00:19, 6876.43it/s] 66%|██████▌   | 264506/400000 [00:36<00:19, 7058.42it/s] 66%|██████▋   | 265221/400000 [00:36<00:19, 7083.97it/s] 66%|██████▋   | 265932/400000 [00:36<00:19, 6965.59it/s] 67%|██████▋   | 266675/400000 [00:36<00:18, 7097.88it/s] 67%|██████▋   | 267390/400000 [00:36<00:18, 7113.09it/s] 67%|██████▋   | 268103/400000 [00:36<00:18, 7039.37it/s] 67%|██████▋   | 268849/400000 [00:36<00:18, 7159.31it/s] 67%|██████▋   | 269571/400000 [00:36<00:18, 7175.50it/s] 68%|██████▊   | 270303/400000 [00:36<00:17, 7216.12it/s] 68%|██████▊   | 271043/400000 [00:37<00:17, 7267.71it/s] 68%|██████▊   | 271786/400000 [00:37<00:17, 7315.46it/s] 68%|██████▊   | 272519/400000 [00:37<00:17, 7283.88it/s] 68%|██████▊   | 273248/400000 [00:37<00:17, 7198.54it/s] 68%|██████▊   | 273969/400000 [00:37<00:17, 7003.78it/s] 69%|██████▊   | 274693/400000 [00:37<00:17, 7071.39it/s] 69%|██████▉   | 275422/400000 [00:37<00:17, 7132.37it/s] 69%|██████▉   | 276137/400000 [00:37<00:17, 7086.42it/s] 69%|██████▉   | 276852/400000 [00:37<00:17, 7103.43it/s] 69%|██████▉   | 277563/400000 [00:37<00:17, 7061.57it/s] 70%|██████▉   | 278292/400000 [00:38<00:17, 7128.44it/s] 70%|██████▉   | 279027/400000 [00:38<00:16, 7192.06it/s] 70%|██████▉   | 279747/400000 [00:38<00:16, 7141.52it/s] 70%|███████   | 280491/400000 [00:38<00:16, 7228.47it/s] 70%|███████   | 281274/400000 [00:38<00:16, 7397.56it/s] 71%|███████   | 282035/400000 [00:38<00:15, 7452.98it/s] 71%|███████   | 282803/400000 [00:38<00:15, 7519.65it/s] 71%|███████   | 283588/400000 [00:38<00:15, 7613.33it/s] 71%|███████   | 284366/400000 [00:38<00:15, 7661.79it/s] 71%|███████▏  | 285138/400000 [00:38<00:14, 7676.88it/s] 71%|███████▏  | 285910/400000 [00:39<00:14, 7686.77it/s] 72%|███████▏  | 286691/400000 [00:39<00:14, 7723.24it/s] 72%|███████▏  | 287464/400000 [00:39<00:14, 7659.25it/s] 72%|███████▏  | 288231/400000 [00:39<00:14, 7525.80it/s] 72%|███████▏  | 288985/400000 [00:39<00:14, 7525.75it/s] 72%|███████▏  | 289766/400000 [00:39<00:14, 7608.33it/s] 73%|███████▎  | 290529/400000 [00:39<00:14, 7612.40it/s] 73%|███████▎  | 291291/400000 [00:39<00:14, 7583.50it/s] 73%|███████▎  | 292050/400000 [00:39<00:14, 7578.64it/s] 73%|███████▎  | 292809/400000 [00:40<00:14, 7478.76it/s] 73%|███████▎  | 293558/400000 [00:40<00:14, 7387.22it/s] 74%|███████▎  | 294309/400000 [00:40<00:14, 7422.13it/s] 74%|███████▍  | 295052/400000 [00:40<00:14, 7418.74it/s] 74%|███████▍  | 295795/400000 [00:40<00:14, 7026.33it/s] 74%|███████▍  | 296542/400000 [00:40<00:14, 7152.58it/s] 74%|███████▍  | 297261/400000 [00:40<00:14, 7074.70it/s] 74%|███████▍  | 297995/400000 [00:40<00:14, 7151.93it/s] 75%|███████▍  | 298713/400000 [00:40<00:14, 7094.08it/s] 75%|███████▍  | 299425/400000 [00:40<00:14, 7036.79it/s] 75%|███████▌  | 300165/400000 [00:41<00:13, 7140.77it/s] 75%|███████▌  | 300888/400000 [00:41<00:13, 7164.82it/s] 75%|███████▌  | 301648/400000 [00:41<00:13, 7289.41it/s] 76%|███████▌  | 302380/400000 [00:41<00:13, 7298.05it/s] 76%|███████▌  | 303115/400000 [00:41<00:13, 7312.32it/s] 76%|███████▌  | 303847/400000 [00:41<00:13, 7244.40it/s] 76%|███████▌  | 304572/400000 [00:41<00:13, 7212.99it/s] 76%|███████▋  | 305294/400000 [00:41<00:13, 7116.53it/s] 77%|███████▋  | 306013/400000 [00:41<00:13, 7137.26it/s] 77%|███████▋  | 306728/400000 [00:41<00:13, 6949.17it/s] 77%|███████▋  | 307425/400000 [00:42<00:13, 6931.22it/s] 77%|███████▋  | 308120/400000 [00:42<00:13, 6838.53it/s] 77%|███████▋  | 308825/400000 [00:42<00:13, 6900.14it/s] 77%|███████▋  | 309540/400000 [00:42<00:12, 6972.97it/s] 78%|███████▊  | 310255/400000 [00:42<00:12, 7024.56it/s] 78%|███████▊  | 311003/400000 [00:42<00:12, 7155.01it/s] 78%|███████▊  | 311801/400000 [00:42<00:11, 7383.15it/s] 78%|███████▊  | 312542/400000 [00:42<00:12, 7271.70it/s] 78%|███████▊  | 313310/400000 [00:42<00:11, 7387.65it/s] 79%|███████▊  | 314076/400000 [00:42<00:11, 7465.51it/s] 79%|███████▊  | 314847/400000 [00:43<00:11, 7534.69it/s] 79%|███████▉  | 315623/400000 [00:43<00:11, 7600.39it/s] 79%|███████▉  | 316385/400000 [00:43<00:11, 7531.47it/s] 79%|███████▉  | 317139/400000 [00:43<00:11, 7486.68it/s] 79%|███████▉  | 317889/400000 [00:43<00:11, 7445.90it/s] 80%|███████▉  | 318660/400000 [00:43<00:10, 7523.20it/s] 80%|███████▉  | 319413/400000 [00:43<00:11, 7091.08it/s] 80%|████████  | 320175/400000 [00:43<00:11, 7239.82it/s] 80%|████████  | 320925/400000 [00:43<00:10, 7314.27it/s] 80%|████████  | 321678/400000 [00:44<00:10, 7375.17it/s] 81%|████████  | 322443/400000 [00:44<00:10, 7454.40it/s] 81%|████████  | 323191/400000 [00:44<00:10, 7417.18it/s] 81%|████████  | 323962/400000 [00:44<00:10, 7501.67it/s] 81%|████████  | 324732/400000 [00:44<00:09, 7557.33it/s] 81%|████████▏ | 325489/400000 [00:44<00:10, 7405.62it/s] 82%|████████▏ | 326231/400000 [00:44<00:10, 7311.41it/s] 82%|████████▏ | 326964/400000 [00:44<00:10, 7183.94it/s] 82%|████████▏ | 327684/400000 [00:44<00:10, 7139.04it/s] 82%|████████▏ | 328425/400000 [00:44<00:09, 7216.41it/s] 82%|████████▏ | 329160/400000 [00:45<00:09, 7254.18it/s] 82%|████████▏ | 329887/400000 [00:45<00:09, 7244.88it/s] 83%|████████▎ | 330646/400000 [00:45<00:09, 7344.80it/s] 83%|████████▎ | 331394/400000 [00:45<00:09, 7384.62it/s] 83%|████████▎ | 332144/400000 [00:45<00:09, 7418.23it/s] 83%|████████▎ | 332887/400000 [00:45<00:09, 7353.65it/s] 83%|████████▎ | 333623/400000 [00:45<00:09, 7253.34it/s] 84%|████████▎ | 334401/400000 [00:45<00:08, 7403.67it/s] 84%|████████▍ | 335182/400000 [00:45<00:08, 7519.07it/s] 84%|████████▍ | 335938/400000 [00:45<00:08, 7531.01it/s] 84%|████████▍ | 336692/400000 [00:46<00:08, 7506.14it/s] 84%|████████▍ | 337444/400000 [00:46<00:08, 7465.53it/s] 85%|████████▍ | 338192/400000 [00:46<00:08, 7241.13it/s] 85%|████████▍ | 338918/400000 [00:46<00:08, 7170.61it/s] 85%|████████▍ | 339681/400000 [00:46<00:08, 7302.04it/s] 85%|████████▌ | 340428/400000 [00:46<00:08, 7351.45it/s] 85%|████████▌ | 341178/400000 [00:46<00:07, 7395.13it/s] 85%|████████▌ | 341919/400000 [00:46<00:07, 7292.22it/s] 86%|████████▌ | 342650/400000 [00:46<00:07, 7276.28it/s] 86%|████████▌ | 343419/400000 [00:46<00:07, 7394.39it/s] 86%|████████▌ | 344162/400000 [00:47<00:07, 7403.07it/s] 86%|████████▌ | 344938/400000 [00:47<00:07, 7504.14it/s] 86%|████████▋ | 345690/400000 [00:47<00:07, 7376.36it/s] 87%|████████▋ | 346429/400000 [00:47<00:07, 7370.08it/s] 87%|████████▋ | 347167/400000 [00:47<00:07, 7099.08it/s] 87%|████████▋ | 347923/400000 [00:47<00:07, 7229.68it/s] 87%|████████▋ | 348703/400000 [00:47<00:06, 7390.90it/s] 87%|████████▋ | 349445/400000 [00:47<00:06, 7392.54it/s] 88%|████████▊ | 350187/400000 [00:47<00:06, 7376.50it/s] 88%|████████▊ | 350926/400000 [00:47<00:06, 7355.47it/s] 88%|████████▊ | 351663/400000 [00:48<00:06, 7355.33it/s] 88%|████████▊ | 352451/400000 [00:48<00:06, 7503.32it/s] 88%|████████▊ | 353203/400000 [00:48<00:06, 7421.02it/s] 88%|████████▊ | 353982/400000 [00:48<00:06, 7525.68it/s] 89%|████████▊ | 354736/400000 [00:48<00:06, 7517.92it/s] 89%|████████▉ | 355489/400000 [00:48<00:05, 7474.88it/s] 89%|████████▉ | 356274/400000 [00:48<00:05, 7581.96it/s] 89%|████████▉ | 357033/400000 [00:48<00:05, 7501.28it/s] 89%|████████▉ | 357784/400000 [00:48<00:05, 7449.67it/s] 90%|████████▉ | 358577/400000 [00:48<00:05, 7587.42it/s] 90%|████████▉ | 359337/400000 [00:49<00:05, 7497.77it/s] 90%|█████████ | 360111/400000 [00:49<00:05, 7566.66it/s] 90%|█████████ | 360878/400000 [00:49<00:05, 7596.53it/s] 90%|█████████ | 361653/400000 [00:49<00:05, 7639.96it/s] 91%|█████████ | 362429/400000 [00:49<00:04, 7673.84it/s] 91%|█████████ | 363197/400000 [00:49<00:04, 7488.40it/s] 91%|█████████ | 363953/400000 [00:49<00:04, 7509.22it/s] 91%|█████████ | 364705/400000 [00:49<00:04, 7461.54it/s] 91%|█████████▏| 365452/400000 [00:49<00:04, 7410.87it/s] 92%|█████████▏| 366225/400000 [00:50<00:04, 7502.38it/s] 92%|█████████▏| 366976/400000 [00:50<00:04, 7411.73it/s] 92%|█████████▏| 367724/400000 [00:50<00:04, 7430.83it/s] 92%|█████████▏| 368479/400000 [00:50<00:04, 7464.17it/s] 92%|█████████▏| 369246/400000 [00:50<00:04, 7522.02it/s] 92%|█████████▏| 369999/400000 [00:50<00:04, 7366.92it/s] 93%|█████████▎| 370737/400000 [00:50<00:04, 7199.75it/s] 93%|█████████▎| 371531/400000 [00:50<00:03, 7405.58it/s] 93%|█████████▎| 372293/400000 [00:50<00:03, 7467.42it/s] 93%|█████████▎| 373064/400000 [00:50<00:03, 7536.93it/s] 93%|█████████▎| 373840/400000 [00:51<00:03, 7602.37it/s] 94%|█████████▎| 374602/400000 [00:51<00:03, 7450.75it/s] 94%|█████████▍| 375349/400000 [00:51<00:03, 7230.54it/s] 94%|█████████▍| 376075/400000 [00:51<00:03, 7060.21it/s] 94%|█████████▍| 376784/400000 [00:51<00:03, 7037.86it/s] 94%|█████████▍| 377532/400000 [00:51<00:03, 7162.81it/s] 95%|█████████▍| 378251/400000 [00:51<00:03, 7149.66it/s] 95%|█████████▍| 379034/400000 [00:51<00:02, 7340.55it/s] 95%|█████████▍| 379811/400000 [00:51<00:02, 7462.56it/s] 95%|█████████▌| 380560/400000 [00:51<00:02, 7416.30it/s] 95%|█████████▌| 381350/400000 [00:52<00:02, 7554.61it/s] 96%|█████████▌| 382108/400000 [00:52<00:02, 7532.56it/s] 96%|█████████▌| 382875/400000 [00:52<00:02, 7571.41it/s] 96%|█████████▌| 383634/400000 [00:52<00:02, 7500.59it/s] 96%|█████████▌| 384394/400000 [00:52<00:02, 7529.73it/s] 96%|█████████▋| 385148/400000 [00:52<00:01, 7445.87it/s] 96%|█████████▋| 385894/400000 [00:52<00:01, 7409.68it/s] 97%|█████████▋| 386636/400000 [00:52<00:01, 7356.56it/s] 97%|█████████▋| 387396/400000 [00:52<00:01, 7425.16it/s] 97%|█████████▋| 388139/400000 [00:52<00:01, 7234.43it/s] 97%|█████████▋| 388867/400000 [00:53<00:01, 7247.59it/s] 97%|█████████▋| 389593/400000 [00:53<00:01, 7085.95it/s] 98%|█████████▊| 390318/400000 [00:53<00:01, 7133.23it/s] 98%|█████████▊| 391050/400000 [00:53<00:01, 7187.09it/s] 98%|█████████▊| 391789/400000 [00:53<00:01, 7244.06it/s] 98%|█████████▊| 392529/400000 [00:53<00:01, 7287.79it/s] 98%|█████████▊| 393259/400000 [00:53<00:00, 7286.26it/s] 99%|█████████▊| 394035/400000 [00:53<00:00, 7422.10it/s] 99%|█████████▊| 394797/400000 [00:53<00:00, 7478.28it/s] 99%|█████████▉| 395563/400000 [00:53<00:00, 7530.93it/s] 99%|█████████▉| 396317/400000 [00:54<00:00, 7487.68it/s] 99%|█████████▉| 397067/400000 [00:54<00:00, 7454.59it/s] 99%|█████████▉| 397813/400000 [00:54<00:00, 7396.03it/s]100%|█████████▉| 398566/400000 [00:54<00:00, 7434.85it/s]100%|█████████▉| 399317/400000 [00:54<00:00, 7454.44it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7327.93it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f92f80f30f0>, <torchtext.data.dataset.TabularDataset object at 0x7f92f80f3240>, <torchtext.vocab.Vocab object at 0x7f92f80f3160>), {}) 

