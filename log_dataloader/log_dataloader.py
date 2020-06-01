
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fd52adf62f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fd52adf62f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fd59ce460d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fd59ce460d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fd5345dc9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fd5345dc9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd54a72a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd54a72a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd54a72a8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2859008/9912422 [00:00<00:00, 28589016.68it/s]9920512it [00:00, 32920506.04it/s]                             
0it [00:00, ?it/s] 57%|█████▋    | 16384/28881 [00:00<00:00, 156572.06it/s]32768it [00:00, 311629.64it/s]                           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466309.96it/s]1654784it [00:00, 11818548.49it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 71760.25it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd52d105a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd52d105da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd54a72a510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd54a72a510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fd54a72a510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:42,  3.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:42,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:42,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:42,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:41,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:29,  5.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:29,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:29,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:29,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:28,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:28,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:20,  7.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:20,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:20,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:20,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:19,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:19,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:14,  9.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:14,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:09, 13.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:06, 17.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:06, 17.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 22.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 35.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 48.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 53.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 57.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 63.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 63.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 71.24 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.20 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ url]
0 examples [00:00, ? examples/s]2020-06-01 18:10:01.613916: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-01 18:10:01.626613: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-01 18:10:01.626794: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562e62eba290 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-01 18:10:01.626813: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
21 examples [00:00, 208.49 examples/s]122 examples [00:00, 273.49 examples/s]213 examples [00:00, 346.11 examples/s]297 examples [00:00, 419.70 examples/s]386 examples [00:00, 498.54 examples/s]478 examples [00:00, 577.53 examples/s]571 examples [00:00, 651.02 examples/s]654 examples [00:00, 694.83 examples/s]745 examples [00:00, 745.81 examples/s]829 examples [00:01, 763.82 examples/s]920 examples [00:01, 800.60 examples/s]1017 examples [00:01, 843.43 examples/s]1110 examples [00:01, 866.29 examples/s]1200 examples [00:01, 875.90 examples/s]1290 examples [00:01, 879.78 examples/s]1386 examples [00:01, 900.08 examples/s]1482 examples [00:01, 916.06 examples/s]1575 examples [00:01, 910.12 examples/s]1667 examples [00:01, 906.87 examples/s]1760 examples [00:02, 912.02 examples/s]1853 examples [00:02, 915.67 examples/s]1953 examples [00:02, 938.83 examples/s]2054 examples [00:02, 956.61 examples/s]2150 examples [00:02, 920.99 examples/s]2243 examples [00:02, 872.14 examples/s]2332 examples [00:02, 826.15 examples/s]2416 examples [00:02, 826.20 examples/s]2500 examples [00:02, 828.58 examples/s]2599 examples [00:02, 869.80 examples/s]2697 examples [00:03, 898.23 examples/s]2795 examples [00:03, 921.01 examples/s]2888 examples [00:03, 923.43 examples/s]2981 examples [00:03, 924.05 examples/s]3074 examples [00:03, 899.03 examples/s]3165 examples [00:03, 899.78 examples/s]3256 examples [00:03, 902.51 examples/s]3348 examples [00:03, 906.20 examples/s]3445 examples [00:03, 923.67 examples/s]3539 examples [00:03, 927.55 examples/s]3633 examples [00:04, 929.95 examples/s]3728 examples [00:04, 933.15 examples/s]3822 examples [00:04, 920.65 examples/s]3915 examples [00:04, 919.57 examples/s]4011 examples [00:04, 928.92 examples/s]4108 examples [00:04, 939.70 examples/s]4203 examples [00:04, 941.16 examples/s]4300 examples [00:04, 947.22 examples/s]4395 examples [00:04, 938.51 examples/s]4489 examples [00:05, 933.85 examples/s]4583 examples [00:05, 923.56 examples/s]4676 examples [00:05, 915.53 examples/s]4773 examples [00:05, 929.95 examples/s]4867 examples [00:05, 917.10 examples/s]4959 examples [00:05, 905.33 examples/s]5052 examples [00:05, 911.86 examples/s]5147 examples [00:05, 919.78 examples/s]5240 examples [00:05, 892.84 examples/s]5330 examples [00:05, 875.36 examples/s]5420 examples [00:06, 881.91 examples/s]5519 examples [00:06, 909.72 examples/s]5617 examples [00:06, 928.46 examples/s]5715 examples [00:06, 942.53 examples/s]5811 examples [00:06, 945.50 examples/s]5906 examples [00:06, 941.16 examples/s]6004 examples [00:06, 950.58 examples/s]6103 examples [00:06, 960.97 examples/s]6200 examples [00:06, 958.27 examples/s]6296 examples [00:06, 952.33 examples/s]6394 examples [00:07, 958.47 examples/s]6493 examples [00:07, 966.41 examples/s]6590 examples [00:07, 960.64 examples/s]6687 examples [00:07, 958.99 examples/s]6784 examples [00:07, 960.54 examples/s]6881 examples [00:07, 939.52 examples/s]6976 examples [00:07, 934.99 examples/s]7071 examples [00:07, 937.70 examples/s]7170 examples [00:07, 950.12 examples/s]7266 examples [00:07, 952.85 examples/s]7362 examples [00:08, 933.48 examples/s]7456 examples [00:08, 929.88 examples/s]7556 examples [00:08, 948.66 examples/s]7656 examples [00:08, 960.72 examples/s]7754 examples [00:08, 965.04 examples/s]7852 examples [00:08, 967.18 examples/s]7949 examples [00:08, 952.39 examples/s]8045 examples [00:08, 933.10 examples/s]8139 examples [00:08, 905.78 examples/s]8230 examples [00:09, 896.30 examples/s]8320 examples [00:09, 894.91 examples/s]8411 examples [00:09, 891.47 examples/s]8506 examples [00:09, 906.03 examples/s]8606 examples [00:09, 930.17 examples/s]8703 examples [00:09, 940.33 examples/s]8799 examples [00:09, 945.11 examples/s]8895 examples [00:09, 949.41 examples/s]8992 examples [00:09, 953.22 examples/s]9092 examples [00:09, 966.58 examples/s]9189 examples [00:10, 954.59 examples/s]9285 examples [00:10, 942.94 examples/s]9380 examples [00:10, 930.77 examples/s]9474 examples [00:10, 917.03 examples/s]9568 examples [00:10, 922.07 examples/s]9664 examples [00:10, 932.58 examples/s]9762 examples [00:10, 944.68 examples/s]9860 examples [00:10, 953.34 examples/s]9956 examples [00:10, 952.91 examples/s]10052 examples [00:10, 890.10 examples/s]10142 examples [00:11, 861.28 examples/s]10229 examples [00:11, 862.86 examples/s]10320 examples [00:11, 875.75 examples/s]10413 examples [00:11, 889.45 examples/s]10508 examples [00:11, 906.25 examples/s]10600 examples [00:11, 907.67 examples/s]10696 examples [00:11, 921.06 examples/s]10797 examples [00:11, 944.93 examples/s]10892 examples [00:11, 941.00 examples/s]10987 examples [00:11, 942.76 examples/s]11082 examples [00:12, 918.05 examples/s]11176 examples [00:12, 924.14 examples/s]11271 examples [00:12, 925.60 examples/s]11364 examples [00:12, 924.36 examples/s]11461 examples [00:12, 934.96 examples/s]11555 examples [00:12, 903.18 examples/s]11646 examples [00:12, 894.20 examples/s]11737 examples [00:12, 897.09 examples/s]11835 examples [00:12, 918.17 examples/s]11932 examples [00:13, 932.19 examples/s]12027 examples [00:13, 935.76 examples/s]12123 examples [00:13, 940.99 examples/s]12218 examples [00:13, 925.91 examples/s]12311 examples [00:13, 911.55 examples/s]12405 examples [00:13, 918.98 examples/s]12500 examples [00:13, 926.08 examples/s]12598 examples [00:13, 940.66 examples/s]12693 examples [00:13, 929.82 examples/s]12788 examples [00:13, 934.52 examples/s]12888 examples [00:14, 952.33 examples/s]12986 examples [00:14, 960.32 examples/s]13085 examples [00:14, 964.73 examples/s]13186 examples [00:14, 975.08 examples/s]13287 examples [00:14, 982.53 examples/s]13387 examples [00:14, 985.78 examples/s]13486 examples [00:14, 979.61 examples/s]13585 examples [00:14, 980.11 examples/s]13684 examples [00:14, 980.61 examples/s]13783 examples [00:14, 957.98 examples/s]13879 examples [00:15, 945.85 examples/s]13980 examples [00:15, 961.97 examples/s]14081 examples [00:15, 974.11 examples/s]14181 examples [00:15, 980.54 examples/s]14280 examples [00:15, 976.43 examples/s]14378 examples [00:15, 972.57 examples/s]14477 examples [00:15, 977.40 examples/s]14577 examples [00:15, 980.07 examples/s]14676 examples [00:15, 977.75 examples/s]14777 examples [00:15, 984.85 examples/s]14876 examples [00:16, 979.06 examples/s]14977 examples [00:16, 987.77 examples/s]15078 examples [00:16, 991.64 examples/s]15178 examples [00:16, 987.06 examples/s]15277 examples [00:16, 981.05 examples/s]15376 examples [00:16, 975.20 examples/s]15474 examples [00:16, 961.85 examples/s]15572 examples [00:16, 964.44 examples/s]15669 examples [00:16, 957.46 examples/s]15765 examples [00:17, 928.28 examples/s]15859 examples [00:17, 931.45 examples/s]15955 examples [00:17, 938.62 examples/s]16052 examples [00:17, 946.40 examples/s]16147 examples [00:17, 921.08 examples/s]16242 examples [00:17, 928.86 examples/s]16336 examples [00:17, 929.03 examples/s]16430 examples [00:17, 918.24 examples/s]16529 examples [00:17, 937.30 examples/s]16630 examples [00:17, 955.19 examples/s]16726 examples [00:18, 889.66 examples/s]16817 examples [00:18, 874.79 examples/s]16907 examples [00:18, 880.95 examples/s]16997 examples [00:18, 884.34 examples/s]17086 examples [00:18, 884.23 examples/s]17175 examples [00:18, 877.90 examples/s]17263 examples [00:18, 864.12 examples/s]17354 examples [00:18, 876.33 examples/s]17448 examples [00:18, 892.08 examples/s]17539 examples [00:18, 896.40 examples/s]17629 examples [00:19, 897.42 examples/s]17719 examples [00:19, 895.49 examples/s]17813 examples [00:19, 906.33 examples/s]17911 examples [00:19, 926.17 examples/s]18007 examples [00:19, 933.39 examples/s]18101 examples [00:19, 933.57 examples/s]18195 examples [00:19, 932.71 examples/s]18291 examples [00:19, 937.54 examples/s]18385 examples [00:19, 930.35 examples/s]18479 examples [00:19, 926.11 examples/s]18578 examples [00:20, 942.84 examples/s]18673 examples [00:20, 937.25 examples/s]18770 examples [00:20, 945.81 examples/s]18865 examples [00:20, 918.43 examples/s]18958 examples [00:20, 898.13 examples/s]19049 examples [00:20, 874.65 examples/s]19138 examples [00:20, 877.21 examples/s]19237 examples [00:20, 906.30 examples/s]19333 examples [00:20, 920.42 examples/s]19427 examples [00:21, 924.37 examples/s]19520 examples [00:21, 901.34 examples/s]19611 examples [00:21, 888.87 examples/s]19708 examples [00:21, 909.99 examples/s]19803 examples [00:21, 920.22 examples/s]19896 examples [00:21, 920.54 examples/s]19993 examples [00:21, 933.39 examples/s]20087 examples [00:21, 886.68 examples/s]20177 examples [00:21, 887.08 examples/s]20267 examples [00:21, 870.69 examples/s]20355 examples [00:22, 866.67 examples/s]20443 examples [00:22, 867.16 examples/s]20530 examples [00:22, 867.45 examples/s]20622 examples [00:22, 879.69 examples/s]20715 examples [00:22, 891.84 examples/s]20807 examples [00:22, 897.61 examples/s]20897 examples [00:22, 882.29 examples/s]20990 examples [00:22, 894.29 examples/s]21086 examples [00:22, 912.84 examples/s]21183 examples [00:22, 927.04 examples/s]21279 examples [00:23, 934.85 examples/s]21373 examples [00:23, 911.39 examples/s]21470 examples [00:23, 927.01 examples/s]21567 examples [00:23, 938.94 examples/s]21665 examples [00:23, 948.42 examples/s]21761 examples [00:23, 925.58 examples/s]21854 examples [00:23, 908.76 examples/s]21948 examples [00:23, 916.91 examples/s]22046 examples [00:23, 934.52 examples/s]22145 examples [00:24, 950.34 examples/s]22241 examples [00:24, 934.15 examples/s]22335 examples [00:24, 920.03 examples/s]22428 examples [00:24, 902.53 examples/s]22522 examples [00:24, 911.83 examples/s]22614 examples [00:24, 908.12 examples/s]22705 examples [00:24, 903.98 examples/s]22800 examples [00:24, 917.01 examples/s]22899 examples [00:24, 936.14 examples/s]22997 examples [00:24, 947.30 examples/s]23093 examples [00:25, 949.60 examples/s]23189 examples [00:25, 933.06 examples/s]23283 examples [00:25, 926.55 examples/s]23379 examples [00:25, 935.10 examples/s]23478 examples [00:25, 948.36 examples/s]23576 examples [00:25, 956.44 examples/s]23675 examples [00:25, 963.41 examples/s]23772 examples [00:25, 959.71 examples/s]23869 examples [00:25, 935.23 examples/s]23963 examples [00:25, 919.52 examples/s]24056 examples [00:26, 902.16 examples/s]24147 examples [00:26, 896.23 examples/s]24237 examples [00:26, 892.16 examples/s]24335 examples [00:26, 915.44 examples/s]24431 examples [00:26, 927.97 examples/s]24527 examples [00:26, 937.02 examples/s]24623 examples [00:26, 942.99 examples/s]24718 examples [00:26, 933.43 examples/s]24812 examples [00:26, 922.69 examples/s]24905 examples [00:26, 894.30 examples/s]24999 examples [00:27, 904.93 examples/s]25096 examples [00:27, 921.70 examples/s]25189 examples [00:27, 902.02 examples/s]25287 examples [00:27, 923.74 examples/s]25380 examples [00:27, 911.44 examples/s]25474 examples [00:27, 918.27 examples/s]25571 examples [00:27, 932.06 examples/s]25667 examples [00:27, 938.37 examples/s]25766 examples [00:27, 949.11 examples/s]25863 examples [00:28, 954.80 examples/s]25959 examples [00:28, 938.73 examples/s]26054 examples [00:28, 919.09 examples/s]26152 examples [00:28, 934.88 examples/s]26246 examples [00:28, 909.40 examples/s]26344 examples [00:28, 929.35 examples/s]26443 examples [00:28, 946.26 examples/s]26542 examples [00:28, 956.54 examples/s]26638 examples [00:28, 956.38 examples/s]26735 examples [00:28, 958.88 examples/s]26832 examples [00:29, 960.01 examples/s]26929 examples [00:29, 959.38 examples/s]27025 examples [00:29, 942.00 examples/s]27120 examples [00:29, 915.75 examples/s]27212 examples [00:29, 914.27 examples/s]27305 examples [00:29, 917.12 examples/s]27401 examples [00:29, 928.31 examples/s]27494 examples [00:29, 916.80 examples/s]27586 examples [00:29, 912.65 examples/s]27678 examples [00:29, 882.76 examples/s]27767 examples [00:30, 863.51 examples/s]27858 examples [00:30, 876.32 examples/s]27946 examples [00:30, 848.97 examples/s]28039 examples [00:30, 870.46 examples/s]28140 examples [00:30, 906.69 examples/s]28240 examples [00:30, 932.02 examples/s]28339 examples [00:30, 948.28 examples/s]28435 examples [00:30, 940.30 examples/s]28530 examples [00:30, 931.31 examples/s]28624 examples [00:31, 906.29 examples/s]28721 examples [00:31, 924.16 examples/s]28819 examples [00:31, 937.98 examples/s]28914 examples [00:31, 931.22 examples/s]29008 examples [00:31, 919.01 examples/s]29101 examples [00:31, 905.09 examples/s]29195 examples [00:31, 914.23 examples/s]29287 examples [00:31, 910.55 examples/s]29379 examples [00:31, 900.96 examples/s]29470 examples [00:31, 901.67 examples/s]29565 examples [00:32, 914.64 examples/s]29661 examples [00:32, 925.36 examples/s]29754 examples [00:32, 926.54 examples/s]29848 examples [00:32, 927.96 examples/s]29941 examples [00:32, 917.69 examples/s]30033 examples [00:32, 885.93 examples/s]30132 examples [00:32, 913.69 examples/s]30230 examples [00:32, 932.34 examples/s]30326 examples [00:32, 938.81 examples/s]30425 examples [00:32, 951.49 examples/s]30522 examples [00:33, 956.38 examples/s]30619 examples [00:33, 960.37 examples/s]30716 examples [00:33, 951.49 examples/s]30812 examples [00:33, 887.63 examples/s]30902 examples [00:33, 881.96 examples/s]30991 examples [00:33, 882.35 examples/s]31083 examples [00:33, 892.15 examples/s]31183 examples [00:33, 921.08 examples/s]31279 examples [00:33, 932.12 examples/s]31380 examples [00:34, 951.94 examples/s]31480 examples [00:34, 964.35 examples/s]31580 examples [00:34, 973.97 examples/s]31678 examples [00:34, 965.17 examples/s]31775 examples [00:34, 945.56 examples/s]31870 examples [00:34, 927.61 examples/s]31963 examples [00:34, 909.83 examples/s]32056 examples [00:34, 915.50 examples/s]32149 examples [00:34, 917.53 examples/s]32241 examples [00:34, 899.42 examples/s]32332 examples [00:35, 888.64 examples/s]32429 examples [00:35, 911.12 examples/s]32521 examples [00:35, 910.83 examples/s]32616 examples [00:35, 921.23 examples/s]32710 examples [00:35, 926.67 examples/s]32809 examples [00:35, 942.37 examples/s]32908 examples [00:35, 955.88 examples/s]33007 examples [00:35, 964.19 examples/s]33106 examples [00:35, 969.23 examples/s]33204 examples [00:35, 966.83 examples/s]33302 examples [00:36, 969.58 examples/s]33401 examples [00:36, 974.06 examples/s]33499 examples [00:36, 958.05 examples/s]33595 examples [00:36, 957.28 examples/s]33691 examples [00:36, 899.43 examples/s]33782 examples [00:36, 886.89 examples/s]33877 examples [00:36, 902.71 examples/s]33974 examples [00:36, 920.39 examples/s]34071 examples [00:36, 933.13 examples/s]34166 examples [00:36, 937.00 examples/s]34260 examples [00:37, 898.84 examples/s]34353 examples [00:37, 906.28 examples/s]34444 examples [00:37, 898.14 examples/s]34540 examples [00:37, 913.94 examples/s]34632 examples [00:37, 911.67 examples/s]34725 examples [00:37, 915.23 examples/s]34819 examples [00:37, 920.70 examples/s]34916 examples [00:37, 933.09 examples/s]35010 examples [00:37, 919.34 examples/s]35103 examples [00:38, 920.15 examples/s]35197 examples [00:38, 924.39 examples/s]35290 examples [00:38, 912.55 examples/s]35382 examples [00:38, 910.13 examples/s]35475 examples [00:38, 913.48 examples/s]35569 examples [00:38, 919.15 examples/s]35663 examples [00:38, 922.27 examples/s]35756 examples [00:38, 917.59 examples/s]35853 examples [00:38, 932.08 examples/s]35951 examples [00:38, 945.54 examples/s]36046 examples [00:39, 908.72 examples/s]36139 examples [00:39, 914.22 examples/s]36232 examples [00:39, 916.81 examples/s]36324 examples [00:39, 901.25 examples/s]36417 examples [00:39, 908.24 examples/s]36508 examples [00:39, 883.96 examples/s]36603 examples [00:39, 901.90 examples/s]36700 examples [00:39, 919.23 examples/s]36793 examples [00:39, 910.12 examples/s]36885 examples [00:39, 900.29 examples/s]36982 examples [00:40, 918.11 examples/s]37075 examples [00:40, 917.27 examples/s]37168 examples [00:40, 919.54 examples/s]37269 examples [00:40, 942.46 examples/s]37368 examples [00:40, 956.14 examples/s]37465 examples [00:40, 957.59 examples/s]37565 examples [00:40, 967.30 examples/s]37666 examples [00:40, 977.94 examples/s]37767 examples [00:40, 985.76 examples/s]37866 examples [00:40, 971.97 examples/s]37964 examples [00:41, 969.16 examples/s]38063 examples [00:41, 973.21 examples/s]38164 examples [00:41, 983.32 examples/s]38263 examples [00:41, 969.82 examples/s]38361 examples [00:41, 924.93 examples/s]38454 examples [00:41, 914.43 examples/s]38546 examples [00:41, 911.25 examples/s]38638 examples [00:41, 907.61 examples/s]38737 examples [00:41, 929.62 examples/s]38835 examples [00:42, 943.03 examples/s]38932 examples [00:42, 950.85 examples/s]39031 examples [00:42, 959.57 examples/s]39129 examples [00:42, 963.18 examples/s]39226 examples [00:42, 960.56 examples/s]39323 examples [00:42, 955.19 examples/s]39419 examples [00:42, 932.83 examples/s]39517 examples [00:42, 944.46 examples/s]39616 examples [00:42, 956.13 examples/s]39712 examples [00:42, 952.49 examples/s]39808 examples [00:43, 945.42 examples/s]39903 examples [00:43, 912.39 examples/s]39995 examples [00:43, 906.26 examples/s]40086 examples [00:43, 860.13 examples/s]40176 examples [00:43, 869.29 examples/s]40264 examples [00:43, 871.77 examples/s]40352 examples [00:43, 867.94 examples/s]40445 examples [00:43, 885.18 examples/s]40537 examples [00:43, 892.99 examples/s]40631 examples [00:43, 905.66 examples/s]40722 examples [00:44, 905.60 examples/s]40818 examples [00:44, 919.03 examples/s]40911 examples [00:44, 916.75 examples/s]41004 examples [00:44, 919.76 examples/s]41102 examples [00:44, 936.23 examples/s]41196 examples [00:44, 933.51 examples/s]41290 examples [00:44, 920.97 examples/s]41383 examples [00:44, 914.10 examples/s]41475 examples [00:44, 899.22 examples/s]41570 examples [00:44, 913.57 examples/s]41666 examples [00:45, 926.95 examples/s]41761 examples [00:45, 931.64 examples/s]41855 examples [00:45, 932.74 examples/s]41949 examples [00:45, 933.57 examples/s]42043 examples [00:45, 927.46 examples/s]42136 examples [00:45, 906.88 examples/s]42230 examples [00:45, 916.12 examples/s]42327 examples [00:45, 931.01 examples/s]42426 examples [00:45, 946.96 examples/s]42521 examples [00:46, 913.58 examples/s]42615 examples [00:46, 919.87 examples/s]42711 examples [00:46, 930.73 examples/s]42805 examples [00:46, 924.84 examples/s]42898 examples [00:46, 923.68 examples/s]42991 examples [00:46, 906.01 examples/s]43084 examples [00:46, 910.95 examples/s]43178 examples [00:46, 918.24 examples/s]43274 examples [00:46, 927.00 examples/s]43367 examples [00:46, 901.21 examples/s]43458 examples [00:47, 900.09 examples/s]43549 examples [00:47, 902.91 examples/s]43645 examples [00:47, 917.69 examples/s]43740 examples [00:47, 924.81 examples/s]43833 examples [00:47, 917.65 examples/s]43928 examples [00:47, 924.83 examples/s]44022 examples [00:47, 927.92 examples/s]44120 examples [00:47, 940.96 examples/s]44215 examples [00:47, 941.68 examples/s]44310 examples [00:47, 892.11 examples/s]44400 examples [00:48, 870.30 examples/s]44488 examples [00:48, 869.29 examples/s]44581 examples [00:48, 886.46 examples/s]44672 examples [00:48, 892.63 examples/s]44765 examples [00:48, 903.37 examples/s]44856 examples [00:48, 886.11 examples/s]44945 examples [00:48, 882.08 examples/s]45043 examples [00:48, 906.98 examples/s]45144 examples [00:48, 933.59 examples/s]45243 examples [00:49, 949.23 examples/s]45341 examples [00:49, 955.71 examples/s]45440 examples [00:49, 963.92 examples/s]45538 examples [00:49, 967.51 examples/s]45635 examples [00:49, 961.09 examples/s]45732 examples [00:49, 957.45 examples/s]45828 examples [00:49, 926.48 examples/s]45922 examples [00:49, 929.76 examples/s]46018 examples [00:49, 937.90 examples/s]46112 examples [00:49, 915.62 examples/s]46212 examples [00:50, 939.37 examples/s]46307 examples [00:50, 941.65 examples/s]46402 examples [00:50, 930.66 examples/s]46496 examples [00:50, 911.77 examples/s]46592 examples [00:50, 923.50 examples/s]46689 examples [00:50, 933.70 examples/s]46783 examples [00:50, 922.79 examples/s]46878 examples [00:50, 929.51 examples/s]46973 examples [00:50, 935.39 examples/s]47067 examples [00:50, 917.57 examples/s]47163 examples [00:51, 927.96 examples/s]47256 examples [00:51, 897.54 examples/s]47347 examples [00:51, 892.23 examples/s]47446 examples [00:51, 919.19 examples/s]47546 examples [00:51, 940.70 examples/s]47644 examples [00:51, 951.07 examples/s]47740 examples [00:51, 949.34 examples/s]47836 examples [00:51, 910.51 examples/s]47928 examples [00:51, 903.58 examples/s]48021 examples [00:51, 910.85 examples/s]48118 examples [00:52, 925.93 examples/s]48214 examples [00:52, 933.93 examples/s]48311 examples [00:52, 941.94 examples/s]48406 examples [00:52, 925.25 examples/s]48504 examples [00:52, 938.90 examples/s]48603 examples [00:52, 951.25 examples/s]48699 examples [00:52, 928.11 examples/s]48793 examples [00:52, 910.20 examples/s]48885 examples [00:52, 892.26 examples/s]48983 examples [00:53, 914.88 examples/s]49079 examples [00:53, 927.87 examples/s]49173 examples [00:53, 927.76 examples/s]49271 examples [00:53, 940.37 examples/s]49366 examples [00:53, 925.60 examples/s]49459 examples [00:53, 916.25 examples/s]49552 examples [00:53, 917.44 examples/s]49645 examples [00:53, 918.90 examples/s]49741 examples [00:53, 928.59 examples/s]49836 examples [00:53, 932.10 examples/s]49934 examples [00:54, 944.38 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6424/50000 [00:00<00:00, 64238.38 examples/s] 36%|███▌      | 17767/50000 [00:00<00:00, 73845.14 examples/s] 56%|█████▌    | 28006/50000 [00:00<00:00, 80584.57 examples/s] 79%|███████▊  | 39281/50000 [00:00<00:00, 88124.99 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 798.90 examples/s]166 examples [00:00, 814.71 examples/s]266 examples [00:00, 861.88 examples/s]365 examples [00:00, 896.64 examples/s]465 examples [00:00, 923.23 examples/s]563 examples [00:00, 937.06 examples/s]659 examples [00:00, 941.97 examples/s]749 examples [00:00, 926.28 examples/s]843 examples [00:00, 927.51 examples/s]937 examples [00:01, 928.53 examples/s]1028 examples [00:01, 921.88 examples/s]1129 examples [00:01, 944.81 examples/s]1230 examples [00:01, 961.08 examples/s]1326 examples [00:01, 934.31 examples/s]1420 examples [00:01, 747.29 examples/s]1517 examples [00:01, 802.34 examples/s]1618 examples [00:01, 853.54 examples/s]1718 examples [00:01, 892.06 examples/s]1818 examples [00:02, 921.51 examples/s]1916 examples [00:02, 938.23 examples/s]2012 examples [00:02, 943.95 examples/s]2113 examples [00:02, 960.45 examples/s]2212 examples [00:02, 966.40 examples/s]2310 examples [00:02, 948.78 examples/s]2409 examples [00:02, 956.82 examples/s]2506 examples [00:02, 948.20 examples/s]2602 examples [00:02, 935.68 examples/s]2696 examples [00:02, 914.05 examples/s]2788 examples [00:03, 880.63 examples/s]2877 examples [00:03, 865.25 examples/s]2964 examples [00:03, 843.91 examples/s]3061 examples [00:03, 876.27 examples/s]3156 examples [00:03, 895.35 examples/s]3256 examples [00:03, 922.90 examples/s]3354 examples [00:03, 936.89 examples/s]3451 examples [00:03, 945.13 examples/s]3552 examples [00:03, 961.29 examples/s]3651 examples [00:03, 968.26 examples/s]3749 examples [00:04, 962.97 examples/s]3846 examples [00:04, 963.08 examples/s]3946 examples [00:04, 972.57 examples/s]4047 examples [00:04, 982.29 examples/s]4146 examples [00:04, 983.46 examples/s]4245 examples [00:04, 984.81 examples/s]4344 examples [00:04, 959.90 examples/s]4441 examples [00:04, 926.62 examples/s]4535 examples [00:04, 923.73 examples/s]4628 examples [00:05, 918.05 examples/s]4724 examples [00:05, 927.78 examples/s]4819 examples [00:05, 933.16 examples/s]4916 examples [00:05, 941.95 examples/s]5015 examples [00:05, 954.61 examples/s]5113 examples [00:05, 960.39 examples/s]5211 examples [00:05, 965.79 examples/s]5308 examples [00:05, 956.83 examples/s]5406 examples [00:05, 960.86 examples/s]5506 examples [00:05, 970.45 examples/s]5606 examples [00:06, 977.90 examples/s]5707 examples [00:06, 984.98 examples/s]5806 examples [00:06, 939.66 examples/s]5901 examples [00:06, 876.34 examples/s]5990 examples [00:06, 859.90 examples/s]6081 examples [00:06, 871.39 examples/s]6180 examples [00:06, 902.13 examples/s]6272 examples [00:06, 906.17 examples/s]6372 examples [00:06, 931.23 examples/s]6471 examples [00:06, 946.00 examples/s]6567 examples [00:07, 946.88 examples/s]6669 examples [00:07, 965.46 examples/s]6766 examples [00:07, 966.70 examples/s]6867 examples [00:07, 978.73 examples/s]6967 examples [00:07, 984.23 examples/s]7067 examples [00:07, 986.84 examples/s]7168 examples [00:07, 992.77 examples/s]7268 examples [00:07, 977.20 examples/s]7366 examples [00:07, 965.38 examples/s]7463 examples [00:07, 943.21 examples/s]7559 examples [00:08, 947.50 examples/s]7658 examples [00:08, 957.56 examples/s]7754 examples [00:08, 956.10 examples/s]7854 examples [00:08, 967.90 examples/s]7954 examples [00:08, 975.63 examples/s]8052 examples [00:08, 971.38 examples/s]8150 examples [00:08, 966.30 examples/s]8247 examples [00:08, 961.70 examples/s]8344 examples [00:08, 930.66 examples/s]8438 examples [00:09, 922.08 examples/s]8531 examples [00:09, 919.32 examples/s]8624 examples [00:09, 904.69 examples/s]8715 examples [00:09, 896.68 examples/s]8809 examples [00:09, 907.26 examples/s]8900 examples [00:09, 903.39 examples/s]8992 examples [00:09, 908.01 examples/s]9085 examples [00:09, 907.37 examples/s]9180 examples [00:09, 919.49 examples/s]9277 examples [00:09, 933.13 examples/s]9375 examples [00:10, 944.20 examples/s]9471 examples [00:10, 947.47 examples/s]9566 examples [00:10, 940.60 examples/s]9663 examples [00:10, 946.57 examples/s]9761 examples [00:10, 954.53 examples/s]9859 examples [00:10, 960.12 examples/s]9956 examples [00:10, 961.02 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteED9EOH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteED9EOH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd54a72a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd54a72a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd54a72a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd5b6bcdda0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd52bb0f940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd54a72a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd54a72a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd54a72a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd52dd593c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd52dd59278>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fd5228c76a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fd5228c76a8> 

  function with postional parmater data_info <function split_train_valid at 0x7fd5228c76a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fd5228c77b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fd5228c77b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fd5228c77b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4cf8c315fe5788618c0a6f0896541a1aee48c53175323d300d6e41020b637e5a
  Stored in directory: /tmp/pip-ephem-wheel-cache-h_4sp_q9/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:17:27, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:02:13, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:10:48, 26.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:26:06, 37.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:29:36, 53.1kB/s].vector_cache/glove.6B.zip:   1%|          | 7.65M/862M [00:01<3:07:56, 75.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:10:54, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:31:06, 154kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:03:42, 220kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<44:22, 314kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:01<31:03, 446kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:01<21:40, 636kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:01<15:15, 899kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:02<10:40, 1.28MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:02<07:35, 1.79MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:29, 2.46MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<05:45, 2.34MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<05:51, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:04<04:29, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<05:40, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<05:34, 2.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:14, 3.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<05:26, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<04:09, 3.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<05:50, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<06:50, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:23, 2.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:10<03:54, 3.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<21:08, 623kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<16:08, 815kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<11:33, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<11:08, 1.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<10:28, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:59, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<07:41, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<06:32, 1.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<04:54, 2.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<06:25, 2.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<05:39, 2.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<05:32, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<04:09, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<05:55, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<06:49, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<05:25, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<05:26, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<04:05, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:48, 2.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:40, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:14, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<03:48, 3.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<12:50, 982kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:20, 1.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:31, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<08:10, 1.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:21, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:23, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:41, 2.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<07:14, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:21, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:45, 2.61MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:14, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:54, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<05:21, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:54, 3.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<09:44, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<08:06, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:55, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:01, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:25, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:43, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:08, 2.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<12:22, 986kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<09:54, 1.23MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:53, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:46, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:02, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:22, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:30, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:09, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:44, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<06:29, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:40, 3.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<10:58, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:44, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<06:27, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:16, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:31, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:51, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:00, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:26, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:03, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:34, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:19, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:55, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<03:35, 3.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:56, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:28, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<05:31, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:34, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:59, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<03:54, 2.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<11:01, 1.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<08:56, 1.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<06:32, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:55, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<10:13, 1.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:19, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:03, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:50, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:56, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:26, 2.55MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:45, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:00, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:19, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:42, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:12, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<03:37, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:09, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:44, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:35, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:45, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:35, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<03:19, 3.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<10:33:48, 17.3kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<7:24:32, 24.7kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<5:10:43, 35.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<3:39:20, 49.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<2:34:34, 70.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<1:48:12, 100kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<1:18:03, 139kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<55:43, 194kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<39:10, 276kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<29:53, 360kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<23:06, 466kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<16:38, 646kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<11:42, 914kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<25:14, 424kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<18:46, 569kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<13:23, 797kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<11:49, 899kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:26, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:50, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:10, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:12, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:36, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:39, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:02, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:45, 2.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:03, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:40, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:27, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:12, 3.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<36:05, 287kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<26:19, 393kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<18:36, 555kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<15:23, 668kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<11:49, 869kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:28, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<08:21, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<07:50, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:56, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:17, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<08:15, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:49, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:58, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<05:51, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:09, 1.64MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:49, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:58, 2.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:31, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<03:25, 2.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:41, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:18, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:15, 3.04MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:36, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:09, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:04, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<02:57, 3.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<1:36:15, 102kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<1:08:21, 143kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<47:56, 204kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<35:43, 273kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<25:59, 374kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<18:23, 528kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<15:06, 640kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<12:32, 770kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<09:12, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:31, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<11:34, 830kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<09:05, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:35, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:49, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:44, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:15, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:10, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:38, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:12, 2.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<1:09:17, 136kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<49:26, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<34:45, 269kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<26:24, 353kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<20:29, 455kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<14:43, 632kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<10:26, 889kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<10:19, 897kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<08:10, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:54, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:16, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:20, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:51, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:06<03:28, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<14:22, 634kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<11:00, 828kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:55, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:37, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:15, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:32, 1.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:58, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<1:06:55, 134kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<47:45, 188kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<33:33, 267kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<25:29, 350kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<19:39, 453kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<14:07, 630kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<10:01, 885kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<09:53, 895kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<07:49, 1.13MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:41, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:01, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:00, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:39, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<03:20, 2.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<35:54, 242kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<26:00, 335kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<18:22, 472kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<14:50, 582kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<12:08, 711kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:52, 972kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<06:18, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<08:48, 973kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<07:02, 1.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:08, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<05:34, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:46, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:30, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:26, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:54, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:47, 2.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:46, 3.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<05:16, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:34, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:22, 2.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:16, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:45, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:41, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:40, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<08:00, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:27, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:41, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:10, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:19, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:30, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:32, 3.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<49:37, 163kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<36:22, 222kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<25:46, 313kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<18:10, 443kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<14:47, 543kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<11:09, 718kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:57, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:25, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:53, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:10, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:44, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:27, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:30, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:20, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:25, 3.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<16:28, 475kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<12:20, 633kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<08:48, 884kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:56, 976kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:12, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:22, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:54, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:56, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:08, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:57, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:24, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:29, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:31, 2.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<55:26, 136kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<39:33, 190kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<27:46, 270kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<21:05, 354kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<15:24, 485kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<10:56, 680kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:22, 790kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:03, 918kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:01, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<04:16, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<55:23, 132kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<39:29, 186kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<27:41, 264kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<21:00, 346kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<16:11, 449kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<11:41, 620kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<08:13, 876kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<57:04, 126kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<40:40, 177kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<28:31, 251kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<21:32, 331kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<15:47, 451kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<11:11, 634kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<09:28, 746kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<07:20, 960kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<05:16, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:20, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:27, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:15, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:53, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:20, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:24, 2.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<45:18, 151kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<32:24, 211kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<22:44, 300kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<17:26, 389kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<12:46, 531kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<09:05, 743kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<07:54, 849kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<06:54, 972kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:10, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<03:40, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<26:33, 250kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<19:15, 345kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<13:34, 488kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<11:00, 598kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<09:02, 728kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:36, 993kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<04:40, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<07:24, 879kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:51, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:13, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:27, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:29, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:28, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:29, 2.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<24:46, 257kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<17:59, 354kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<12:42, 499kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<10:12, 617kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<08:22, 751kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<06:09, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<05:18, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:20, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:10, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:39, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:49, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:07, 2.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<06:30, 938kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:10, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:01, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:05, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:10, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:16, 2.62MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<39:46, 150kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<29:55, 199kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<21:11, 280kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<14:46, 399kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<1:16:16, 77.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<53:55, 109kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<37:44, 155kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<27:37, 211kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<20:35, 282kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<14:40, 395kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<10:15, 561kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<29:09, 197kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<20:58, 274kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<14:45, 387kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<11:34, 491kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<09:14, 614kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<06:44, 839kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:35, 1.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:28, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:16, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:33, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:37, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:49, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:01, 2.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<21:08, 259kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<15:21, 356kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<10:48, 503kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<08:47, 615kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:42, 804kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:48, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:31, 1.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:14, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:14, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:18, 2.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<20:38, 255kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<14:58, 351kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<10:32, 496kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<08:33, 607kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<06:30, 797kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:40, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:26, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:37, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:38, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:01, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:38, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:56, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:31, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:49, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:12, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:36, 3.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:36, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:49, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:11, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:20, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:40, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:18, 2.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:44, 2.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:18, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:38, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:03, 2.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:28, 3.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<05:09, 911kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:04, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:57, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:08, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:08, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:23, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:46, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:26, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<01:37, 2.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:11, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<01:59, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:30, 2.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:04, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<01:54, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:26, 3.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:01, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:18, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:47, 2.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:25<01:19, 3.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:13, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:59, 2.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:28, 2.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:01, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:50, 2.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:22, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:47, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:19, 3.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<00:58, 4.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<15:50, 258kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<11:29, 354kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<08:04, 501kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<06:33, 612kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<05:24, 742kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:56, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:47, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:03, 970kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:11, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:19, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:30, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:32, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:24, 2.72MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:31, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:05, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:33, 2.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:45, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:18, 2.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:46, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:00, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:34, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:07, 3.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:38, 775kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:37, 992kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:36, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:38, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:35, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:59, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:24, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<13:29, 257kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<09:47, 353kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<06:52, 499kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<05:33, 611kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:34, 741kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<03:20, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:21, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<04:31, 734kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:30, 947kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:31, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:30, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:05, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:31, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:48, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:54, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:28, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:03, 2.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<03:34, 873kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:48, 1.11MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:01, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:06, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:06, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:37, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:09, 2.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<22:05, 135kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<15:44, 189kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<10:59, 268kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<08:17, 352kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<06:24, 454kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:35, 630kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<03:14, 887kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<03:14, 877kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:33, 1.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:51, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:55, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:37, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:12, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:27, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:18, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:58, 2.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:17, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:26, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:48, 3.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:33, 1.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:00, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:27, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:35, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:37, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:15, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:53, 2.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<18:02, 135kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<12:50, 189kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<08:56, 268kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<06:43, 352kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<04:56, 478kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<03:28, 671kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:56, 783kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:32, 904kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:52, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:18, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<02:33, 871kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:00, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:27, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:09, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:49, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<15:22, 136kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<10:56, 190kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<07:37, 270kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<05:42, 354kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<04:11, 481kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<02:56, 676kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:28, 787kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:07, 915kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:34, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<01:06, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:34, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:17, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:56, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:04, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:06, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:51, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:36, 2.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:09, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:58, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:53, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:58, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:45, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:41<00:32, 2.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<11:54, 135kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<08:27, 190kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<05:51, 269kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<04:22, 353kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<03:22, 455kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:24, 632kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:40, 893kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:58, 745kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:31, 958kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:05, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:04, 1.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:02, 1.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:47, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:32, 2.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<08:56, 149kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<06:22, 209kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<04:24, 296kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<03:17, 385kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<02:25, 520kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:41, 729kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:26, 836kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:14, 960kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:54, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:38, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:53, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:44, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:36, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:38, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:29, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:20, 2.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<04:03, 245kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<02:55, 338kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:00, 477kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:34, 588kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:17, 711kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:56, 964kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:37, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<03:31, 243kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<02:32, 334kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:44, 472kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:21, 582kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<01:06, 706kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:48, 965kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:32, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:55, 773kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:43, 991kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:30, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:29, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:28, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:21, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:14, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:12, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:12, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:08, 3.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:14, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:06, 3.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:47, 477kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:34, 637kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:23, 888kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:18, 978kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:14, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:09, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:03, 2.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:26, 119kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:59, 167kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:35, 237kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:19, 313kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:14, 408kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:09, 565kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:02, 799kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:16, 125kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:09, 176kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 250kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 592/400000 [00:00<01:07, 5884.09it/s]  0%|          | 1224/400000 [00:00<01:06, 6008.17it/s]  0%|          | 1930/400000 [00:00<01:03, 6287.39it/s]  1%|          | 2642/400000 [00:00<01:00, 6515.10it/s]  1%|          | 3377/400000 [00:00<00:58, 6742.96it/s]  1%|          | 4030/400000 [00:00<00:59, 6675.37it/s]  1%|          | 4654/400000 [00:00<01:00, 6536.05it/s]  1%|▏         | 5314/400000 [00:00<01:00, 6554.59it/s]  1%|▏         | 5991/400000 [00:00<00:59, 6616.54it/s]  2%|▏         | 6728/400000 [00:01<00:57, 6824.07it/s]  2%|▏         | 7404/400000 [00:01<00:57, 6804.07it/s]  2%|▏         | 8075/400000 [00:01<00:58, 6708.40it/s]  2%|▏         | 8740/400000 [00:01<00:59, 6549.89it/s]  2%|▏         | 9396/400000 [00:01<00:59, 6552.88it/s]  3%|▎         | 10049/400000 [00:01<00:59, 6542.53it/s]  3%|▎         | 10718/400000 [00:01<00:59, 6583.88it/s]  3%|▎         | 11407/400000 [00:01<00:58, 6672.45it/s]  3%|▎         | 12132/400000 [00:01<00:56, 6834.72it/s]  3%|▎         | 12875/400000 [00:01<00:55, 7002.02it/s]  3%|▎         | 13609/400000 [00:02<00:54, 7099.91it/s]  4%|▎         | 14321/400000 [00:02<00:55, 7001.94it/s]  4%|▍         | 15023/400000 [00:02<00:56, 6787.12it/s]  4%|▍         | 15705/400000 [00:02<00:57, 6675.64it/s]  4%|▍         | 16375/400000 [00:02<00:57, 6640.55it/s]  4%|▍         | 17105/400000 [00:02<00:56, 6824.96it/s]  4%|▍         | 17828/400000 [00:02<00:55, 6939.38it/s]  5%|▍         | 18562/400000 [00:02<00:54, 7053.45it/s]  5%|▍         | 19270/400000 [00:02<00:54, 6964.12it/s]  5%|▍         | 19969/400000 [00:02<00:55, 6835.73it/s]  5%|▌         | 20655/400000 [00:03<00:57, 6627.68it/s]  5%|▌         | 21327/400000 [00:03<00:56, 6652.91it/s]  6%|▌         | 22052/400000 [00:03<00:55, 6819.45it/s]  6%|▌         | 22794/400000 [00:03<00:53, 6988.39it/s]  6%|▌         | 23496/400000 [00:03<00:54, 6899.57it/s]  6%|▌         | 24229/400000 [00:03<00:53, 7022.29it/s]  6%|▌         | 24934/400000 [00:03<00:53, 7022.31it/s]  6%|▋         | 25648/400000 [00:03<00:53, 7056.40it/s]  7%|▋         | 26385/400000 [00:03<00:52, 7146.29it/s]  7%|▋         | 27129/400000 [00:03<00:51, 7229.89it/s]  7%|▋         | 27869/400000 [00:04<00:51, 7279.79it/s]  7%|▋         | 28598/400000 [00:04<00:51, 7281.68it/s]  7%|▋         | 29327/400000 [00:04<00:52, 7013.89it/s]  8%|▊         | 30031/400000 [00:04<00:53, 6858.59it/s]  8%|▊         | 30720/400000 [00:04<00:54, 6752.98it/s]  8%|▊         | 31463/400000 [00:04<00:53, 6940.25it/s]  8%|▊         | 32185/400000 [00:04<00:52, 7021.77it/s]  8%|▊         | 32893/400000 [00:04<00:52, 7037.01it/s]  8%|▊         | 33614/400000 [00:04<00:51, 7087.93it/s]  9%|▊         | 34345/400000 [00:04<00:51, 7152.92it/s]  9%|▉         | 35085/400000 [00:05<00:50, 7222.62it/s]  9%|▉         | 35809/400000 [00:05<00:52, 6979.08it/s]  9%|▉         | 36510/400000 [00:05<00:53, 6798.27it/s]  9%|▉         | 37193/400000 [00:05<00:54, 6675.79it/s]  9%|▉         | 37863/400000 [00:05<00:55, 6550.85it/s] 10%|▉         | 38539/400000 [00:05<00:54, 6611.11it/s] 10%|▉         | 39244/400000 [00:05<00:53, 6734.76it/s] 10%|▉         | 39956/400000 [00:05<00:52, 6844.85it/s] 10%|█         | 40643/400000 [00:05<00:53, 6703.47it/s] 10%|█         | 41316/400000 [00:06<00:53, 6679.40it/s] 10%|█         | 41986/400000 [00:06<00:54, 6593.75it/s] 11%|█         | 42648/400000 [00:06<00:54, 6599.95it/s] 11%|█         | 43372/400000 [00:06<00:52, 6777.64it/s] 11%|█         | 44096/400000 [00:06<00:51, 6909.15it/s] 11%|█         | 44789/400000 [00:06<00:52, 6819.31it/s] 11%|█▏        | 45473/400000 [00:06<00:52, 6761.04it/s] 12%|█▏        | 46151/400000 [00:06<00:54, 6550.65it/s] 12%|█▏        | 46827/400000 [00:06<00:53, 6611.13it/s] 12%|█▏        | 47534/400000 [00:06<00:52, 6740.74it/s] 12%|█▏        | 48275/400000 [00:07<00:50, 6927.39it/s] 12%|█▏        | 49029/400000 [00:07<00:49, 7098.17it/s] 12%|█▏        | 49742/400000 [00:07<00:49, 7089.42it/s] 13%|█▎        | 50453/400000 [00:07<00:49, 7044.69it/s] 13%|█▎        | 51178/400000 [00:07<00:49, 7104.67it/s] 13%|█▎        | 51890/400000 [00:07<00:49, 6990.95it/s] 13%|█▎        | 52609/400000 [00:07<00:49, 7049.39it/s] 13%|█▎        | 53330/400000 [00:07<00:48, 7096.36it/s] 14%|█▎        | 54065/400000 [00:07<00:48, 7168.56it/s] 14%|█▎        | 54794/400000 [00:07<00:47, 7201.84it/s] 14%|█▍        | 55535/400000 [00:08<00:47, 7262.84it/s] 14%|█▍        | 56275/400000 [00:08<00:47, 7300.86it/s] 14%|█▍        | 57006/400000 [00:08<00:47, 7188.54it/s] 14%|█▍        | 57726/400000 [00:08<00:48, 7071.75it/s] 15%|█▍        | 58454/400000 [00:08<00:47, 7132.60it/s] 15%|█▍        | 59169/400000 [00:08<00:48, 7063.32it/s] 15%|█▍        | 59899/400000 [00:08<00:47, 7132.29it/s] 15%|█▌        | 60624/400000 [00:08<00:47, 7165.59it/s] 15%|█▌        | 61374/400000 [00:08<00:46, 7260.94it/s] 16%|█▌        | 62113/400000 [00:08<00:46, 7297.34it/s] 16%|█▌        | 62850/400000 [00:09<00:46, 7318.61it/s] 16%|█▌        | 63583/400000 [00:09<00:47, 7013.72it/s] 16%|█▌        | 64304/400000 [00:09<00:47, 7069.26it/s] 16%|█▋        | 65041/400000 [00:09<00:46, 7155.97it/s] 16%|█▋        | 65772/400000 [00:09<00:46, 7200.12it/s] 17%|█▋        | 66494/400000 [00:09<00:46, 7101.55it/s] 17%|█▋        | 67248/400000 [00:09<00:46, 7225.72it/s] 17%|█▋        | 67972/400000 [00:09<00:46, 7083.29it/s] 17%|█▋        | 68715/400000 [00:09<00:46, 7182.45it/s] 17%|█▋        | 69446/400000 [00:10<00:45, 7219.37it/s] 18%|█▊        | 70169/400000 [00:10<00:46, 7161.62it/s] 18%|█▊        | 70917/400000 [00:10<00:45, 7252.69it/s] 18%|█▊        | 71644/400000 [00:10<00:45, 7240.09it/s] 18%|█▊        | 72369/400000 [00:10<00:46, 6998.38it/s] 18%|█▊        | 73106/400000 [00:10<00:46, 7104.88it/s] 18%|█▊        | 73849/400000 [00:10<00:45, 7197.84it/s] 19%|█▊        | 74571/400000 [00:10<00:45, 7181.90it/s] 19%|█▉        | 75294/400000 [00:10<00:45, 7196.16it/s] 19%|█▉        | 76036/400000 [00:10<00:44, 7260.39it/s] 19%|█▉        | 76763/400000 [00:11<00:44, 7238.14it/s] 19%|█▉        | 77496/400000 [00:11<00:44, 7262.85it/s] 20%|█▉        | 78235/400000 [00:11<00:44, 7298.67it/s] 20%|█▉        | 78966/400000 [00:11<00:44, 7270.32it/s] 20%|█▉        | 79695/400000 [00:11<00:44, 7276.00it/s] 20%|██        | 80423/400000 [00:11<00:44, 7214.30it/s] 20%|██        | 81145/400000 [00:11<00:44, 7214.08it/s] 20%|██        | 81867/400000 [00:11<00:44, 7164.66it/s] 21%|██        | 82584/400000 [00:11<00:45, 6952.21it/s] 21%|██        | 83329/400000 [00:11<00:44, 7092.06it/s] 21%|██        | 84074/400000 [00:12<00:43, 7195.16it/s] 21%|██        | 84826/400000 [00:12<00:43, 7289.39it/s] 21%|██▏       | 85562/400000 [00:12<00:43, 7308.15it/s] 22%|██▏       | 86294/400000 [00:12<00:43, 7283.20it/s] 22%|██▏       | 87024/400000 [00:12<00:44, 7018.19it/s] 22%|██▏       | 87729/400000 [00:12<00:45, 6929.66it/s] 22%|██▏       | 88425/400000 [00:12<00:45, 6865.44it/s] 22%|██▏       | 89114/400000 [00:12<00:45, 6799.39it/s] 22%|██▏       | 89796/400000 [00:12<00:45, 6787.16it/s] 23%|██▎       | 90549/400000 [00:12<00:44, 6991.69it/s] 23%|██▎       | 91287/400000 [00:13<00:43, 7101.91it/s] 23%|██▎       | 92032/400000 [00:13<00:42, 7202.83it/s] 23%|██▎       | 92790/400000 [00:13<00:42, 7310.99it/s] 23%|██▎       | 93523/400000 [00:13<00:43, 7123.55it/s] 24%|██▎       | 94258/400000 [00:13<00:42, 7187.57it/s] 24%|██▍       | 95003/400000 [00:13<00:41, 7262.54it/s] 24%|██▍       | 95731/400000 [00:13<00:42, 7240.72it/s] 24%|██▍       | 96476/400000 [00:13<00:41, 7301.84it/s] 24%|██▍       | 97207/400000 [00:13<00:41, 7245.04it/s] 24%|██▍       | 97957/400000 [00:13<00:41, 7319.30it/s] 25%|██▍       | 98690/400000 [00:14<00:41, 7231.65it/s] 25%|██▍       | 99437/400000 [00:14<00:41, 7299.75it/s] 25%|██▌       | 100187/400000 [00:14<00:40, 7357.43it/s] 25%|██▌       | 100924/400000 [00:14<00:40, 7322.96it/s] 25%|██▌       | 101675/400000 [00:14<00:40, 7375.62it/s] 26%|██▌       | 102413/400000 [00:14<00:40, 7356.11it/s] 26%|██▌       | 103156/400000 [00:14<00:40, 7377.74it/s] 26%|██▌       | 103894/400000 [00:14<00:40, 7329.47it/s] 26%|██▌       | 104628/400000 [00:14<00:41, 7110.42it/s] 26%|██▋       | 105341/400000 [00:15<00:41, 7056.83it/s] 27%|██▋       | 106048/400000 [00:15<00:42, 6850.07it/s] 27%|██▋       | 106769/400000 [00:15<00:42, 6953.63it/s] 27%|██▋       | 107467/400000 [00:15<00:42, 6936.61it/s] 27%|██▋       | 108162/400000 [00:15<00:42, 6817.24it/s] 27%|██▋       | 108859/400000 [00:15<00:42, 6860.26it/s] 27%|██▋       | 109601/400000 [00:15<00:41, 7018.78it/s] 28%|██▊       | 110350/400000 [00:15<00:40, 7152.34it/s] 28%|██▊       | 111067/400000 [00:15<00:40, 7144.90it/s] 28%|██▊       | 111810/400000 [00:15<00:39, 7228.06it/s] 28%|██▊       | 112562/400000 [00:16<00:39, 7312.89it/s] 28%|██▊       | 113305/400000 [00:16<00:39, 7345.88it/s] 29%|██▊       | 114041/400000 [00:16<00:39, 7322.86it/s] 29%|██▊       | 114774/400000 [00:16<00:39, 7139.75it/s] 29%|██▉       | 115502/400000 [00:16<00:39, 7179.45it/s] 29%|██▉       | 116239/400000 [00:16<00:39, 7233.68it/s] 29%|██▉       | 116980/400000 [00:16<00:38, 7283.18it/s] 29%|██▉       | 117710/400000 [00:16<00:38, 7288.07it/s] 30%|██▉       | 118440/400000 [00:16<00:40, 6939.29it/s] 30%|██▉       | 119192/400000 [00:16<00:39, 7101.76it/s] 30%|██▉       | 119906/400000 [00:17<00:39, 7106.26it/s] 30%|███       | 120655/400000 [00:17<00:38, 7215.68it/s] 30%|███       | 121379/400000 [00:17<00:39, 7121.53it/s] 31%|███       | 122094/400000 [00:17<00:38, 7128.33it/s] 31%|███       | 122841/400000 [00:17<00:38, 7225.97it/s] 31%|███       | 123588/400000 [00:17<00:37, 7296.64it/s] 31%|███       | 124319/400000 [00:17<00:37, 7299.25it/s] 31%|███▏      | 125050/400000 [00:17<00:37, 7293.19it/s] 31%|███▏      | 125780/400000 [00:17<00:38, 7117.01it/s] 32%|███▏      | 126524/400000 [00:17<00:37, 7208.84it/s] 32%|███▏      | 127271/400000 [00:18<00:37, 7282.68it/s] 32%|███▏      | 128016/400000 [00:18<00:37, 7329.89it/s] 32%|███▏      | 128750/400000 [00:18<00:37, 7325.51it/s] 32%|███▏      | 129484/400000 [00:18<00:36, 7322.85it/s] 33%|███▎      | 130234/400000 [00:18<00:36, 7374.39it/s] 33%|███▎      | 130977/400000 [00:18<00:36, 7389.98it/s] 33%|███▎      | 131731/400000 [00:18<00:36, 7433.49it/s] 33%|███▎      | 132481/400000 [00:18<00:35, 7452.84it/s] 33%|███▎      | 133227/400000 [00:18<00:36, 7293.83it/s] 33%|███▎      | 133970/400000 [00:18<00:36, 7331.50it/s] 34%|███▎      | 134716/400000 [00:19<00:36, 7368.08it/s] 34%|███▍      | 135462/400000 [00:19<00:35, 7394.84it/s] 34%|███▍      | 136205/400000 [00:19<00:35, 7402.76it/s] 34%|███▍      | 136946/400000 [00:19<00:35, 7326.61it/s] 34%|███▍      | 137688/400000 [00:19<00:35, 7352.87it/s] 35%|███▍      | 138433/400000 [00:19<00:35, 7379.87it/s] 35%|███▍      | 139174/400000 [00:19<00:35, 7387.29it/s] 35%|███▍      | 139913/400000 [00:19<00:36, 7099.05it/s] 35%|███▌      | 140626/400000 [00:19<00:38, 6800.11it/s] 35%|███▌      | 141311/400000 [00:20<00:39, 6503.86it/s] 35%|███▌      | 141968/400000 [00:20<00:40, 6432.43it/s] 36%|███▌      | 142616/400000 [00:20<00:40, 6406.51it/s] 36%|███▌      | 143314/400000 [00:20<00:39, 6567.81it/s] 36%|███▌      | 143988/400000 [00:20<00:38, 6618.03it/s] 36%|███▌      | 144679/400000 [00:20<00:38, 6702.89it/s] 36%|███▋      | 145420/400000 [00:20<00:36, 6899.14it/s] 37%|███▋      | 146147/400000 [00:20<00:36, 7004.13it/s] 37%|███▋      | 146893/400000 [00:20<00:35, 7133.53it/s] 37%|███▋      | 147610/400000 [00:20<00:35, 7143.69it/s] 37%|███▋      | 148354/400000 [00:21<00:34, 7229.16it/s] 37%|███▋      | 149103/400000 [00:21<00:34, 7304.54it/s] 37%|███▋      | 149839/400000 [00:21<00:34, 7318.61it/s] 38%|███▊      | 150579/400000 [00:21<00:33, 7340.97it/s] 38%|███▊      | 151314/400000 [00:21<00:33, 7331.49it/s] 38%|███▊      | 152061/400000 [00:21<00:33, 7371.52it/s] 38%|███▊      | 152806/400000 [00:21<00:33, 7392.86it/s] 38%|███▊      | 153554/400000 [00:21<00:33, 7417.25it/s] 39%|███▊      | 154296/400000 [00:21<00:33, 7361.72it/s] 39%|███▉      | 155033/400000 [00:21<00:34, 7041.56it/s] 39%|███▉      | 155741/400000 [00:22<00:35, 6898.96it/s] 39%|███▉      | 156458/400000 [00:22<00:34, 6977.73it/s] 39%|███▉      | 157186/400000 [00:22<00:34, 7065.73it/s] 39%|███▉      | 157937/400000 [00:22<00:33, 7192.83it/s] 40%|███▉      | 158659/400000 [00:22<00:34, 7067.94it/s] 40%|███▉      | 159408/400000 [00:22<00:33, 7188.46it/s] 40%|████      | 160129/400000 [00:22<00:33, 7116.86it/s] 40%|████      | 160858/400000 [00:22<00:33, 7165.24it/s] 40%|████      | 161596/400000 [00:22<00:32, 7226.36it/s] 41%|████      | 162320/400000 [00:22<00:32, 7214.04it/s] 41%|████      | 163046/400000 [00:23<00:32, 7225.04it/s] 41%|████      | 163790/400000 [00:23<00:32, 7287.52it/s] 41%|████      | 164539/400000 [00:23<00:32, 7345.98it/s] 41%|████▏     | 165291/400000 [00:23<00:31, 7396.37it/s] 42%|████▏     | 166032/400000 [00:23<00:32, 7295.78it/s] 42%|████▏     | 166777/400000 [00:23<00:31, 7339.88it/s] 42%|████▏     | 167512/400000 [00:23<00:32, 7202.91it/s] 42%|████▏     | 168234/400000 [00:23<00:33, 6965.30it/s] 42%|████▏     | 168933/400000 [00:23<00:33, 6896.69it/s] 42%|████▏     | 169625/400000 [00:24<00:33, 6819.73it/s] 43%|████▎     | 170314/400000 [00:24<00:33, 6837.98it/s] 43%|████▎     | 171069/400000 [00:24<00:32, 7035.14it/s] 43%|████▎     | 171789/400000 [00:24<00:32, 7064.65it/s] 43%|████▎     | 172550/400000 [00:24<00:31, 7217.23it/s] 43%|████▎     | 173293/400000 [00:24<00:31, 7279.61it/s] 44%|████▎     | 174031/400000 [00:24<00:30, 7308.92it/s] 44%|████▎     | 174763/400000 [00:24<00:31, 7066.64it/s] 44%|████▍     | 175473/400000 [00:24<00:33, 6796.27it/s] 44%|████▍     | 176157/400000 [00:24<00:33, 6717.31it/s] 44%|████▍     | 176832/400000 [00:25<00:33, 6657.99it/s] 44%|████▍     | 177507/400000 [00:25<00:33, 6684.13it/s] 45%|████▍     | 178221/400000 [00:25<00:32, 6813.27it/s] 45%|████▍     | 178970/400000 [00:25<00:31, 7001.76it/s] 45%|████▍     | 179689/400000 [00:25<00:31, 7055.42it/s] 45%|████▌     | 180404/400000 [00:25<00:31, 7082.58it/s] 45%|████▌     | 181141/400000 [00:25<00:30, 7165.42it/s] 45%|████▌     | 181891/400000 [00:25<00:30, 7260.61it/s] 46%|████▌     | 182619/400000 [00:25<00:30, 7153.72it/s] 46%|████▌     | 183374/400000 [00:25<00:29, 7267.59it/s] 46%|████▌     | 184102/400000 [00:26<00:29, 7233.52it/s] 46%|████▌     | 184855/400000 [00:26<00:29, 7318.81it/s] 46%|████▋     | 185615/400000 [00:26<00:28, 7398.27it/s] 47%|████▋     | 186373/400000 [00:26<00:28, 7449.92it/s] 47%|████▋     | 187119/400000 [00:26<00:28, 7450.77it/s] 47%|████▋     | 187865/400000 [00:26<00:29, 7214.35it/s] 47%|████▋     | 188611/400000 [00:26<00:29, 7284.52it/s] 47%|████▋     | 189341/400000 [00:26<00:29, 7181.93it/s] 48%|████▊     | 190061/400000 [00:26<00:29, 7021.32it/s] 48%|████▊     | 190765/400000 [00:26<00:30, 6882.02it/s] 48%|████▊     | 191456/400000 [00:27<00:30, 6808.62it/s] 48%|████▊     | 192206/400000 [00:27<00:29, 7001.90it/s] 48%|████▊     | 192946/400000 [00:27<00:29, 7115.11it/s] 48%|████▊     | 193660/400000 [00:27<00:29, 6957.24it/s] 49%|████▊     | 194358/400000 [00:27<00:30, 6833.11it/s] 49%|████▉     | 195063/400000 [00:27<00:29, 6894.79it/s] 49%|████▉     | 195812/400000 [00:27<00:28, 7063.08it/s] 49%|████▉     | 196554/400000 [00:27<00:28, 7164.60it/s] 49%|████▉     | 197273/400000 [00:27<00:28, 7132.84it/s] 49%|████▉     | 197988/400000 [00:27<00:28, 7125.17it/s] 50%|████▉     | 198716/400000 [00:28<00:28, 7169.57it/s] 50%|████▉     | 199471/400000 [00:28<00:27, 7276.94it/s] 50%|█████     | 200232/400000 [00:28<00:27, 7371.24it/s] 50%|█████     | 200989/400000 [00:28<00:26, 7428.53it/s] 50%|█████     | 201733/400000 [00:28<00:27, 7315.22it/s] 51%|█████     | 202466/400000 [00:28<00:27, 7225.03it/s] 51%|█████     | 203218/400000 [00:28<00:26, 7310.15it/s] 51%|█████     | 203957/400000 [00:28<00:26, 7332.95it/s] 51%|█████     | 204711/400000 [00:28<00:26, 7393.54it/s] 51%|█████▏    | 205451/400000 [00:29<00:26, 7282.10it/s] 52%|█████▏    | 206180/400000 [00:29<00:26, 7233.03it/s] 52%|█████▏    | 206925/400000 [00:29<00:26, 7295.71it/s] 52%|█████▏    | 207672/400000 [00:29<00:26, 7344.81it/s] 52%|█████▏    | 208428/400000 [00:29<00:25, 7406.31it/s] 52%|█████▏    | 209170/400000 [00:29<00:25, 7404.77it/s] 52%|█████▏    | 209911/400000 [00:29<00:25, 7380.43it/s] 53%|█████▎    | 210661/400000 [00:29<00:25, 7415.16it/s] 53%|█████▎    | 211419/400000 [00:29<00:25, 7461.72it/s] 53%|█████▎    | 212166/400000 [00:29<00:25, 7384.79it/s] 53%|█████▎    | 212905/400000 [00:30<00:25, 7380.19it/s] 53%|█████▎    | 213649/400000 [00:30<00:25, 7396.41it/s] 54%|█████▎    | 214389/400000 [00:30<00:25, 7286.25it/s] 54%|█████▍    | 215135/400000 [00:30<00:25, 7336.54it/s] 54%|█████▍    | 215881/400000 [00:30<00:24, 7373.04it/s] 54%|█████▍    | 216622/400000 [00:30<00:24, 7383.64it/s] 54%|█████▍    | 217361/400000 [00:30<00:24, 7378.35it/s] 55%|█████▍    | 218111/400000 [00:30<00:24, 7411.87it/s] 55%|█████▍    | 218853/400000 [00:30<00:24, 7382.31it/s] 55%|█████▍    | 219595/400000 [00:30<00:24, 7389.46it/s] 55%|█████▌    | 220335/400000 [00:31<00:24, 7383.92it/s] 55%|█████▌    | 221075/400000 [00:31<00:24, 7386.63it/s] 55%|█████▌    | 221817/400000 [00:31<00:24, 7395.83it/s] 56%|█████▌    | 222557/400000 [00:31<00:24, 7363.08it/s] 56%|█████▌    | 223294/400000 [00:31<00:24, 7115.77it/s] 56%|█████▌    | 224008/400000 [00:31<00:25, 6948.47it/s] 56%|█████▌    | 224722/400000 [00:31<00:25, 7003.14it/s] 56%|█████▋    | 225480/400000 [00:31<00:24, 7165.94it/s] 57%|█████▋    | 226199/400000 [00:31<00:24, 7152.73it/s] 57%|█████▋    | 226932/400000 [00:31<00:24, 7203.91it/s] 57%|█████▋    | 227654/400000 [00:32<00:23, 7207.31it/s] 57%|█████▋    | 228376/400000 [00:32<00:23, 7159.15it/s] 57%|█████▋    | 229131/400000 [00:32<00:23, 7271.38it/s] 57%|█████▋    | 229881/400000 [00:32<00:23, 7336.82it/s] 58%|█████▊    | 230643/400000 [00:32<00:22, 7418.72it/s] 58%|█████▊    | 231386/400000 [00:32<00:22, 7350.11it/s] 58%|█████▊    | 232122/400000 [00:32<00:23, 7269.14it/s] 58%|█████▊    | 232850/400000 [00:32<00:23, 7161.95it/s] 58%|█████▊    | 233598/400000 [00:32<00:22, 7252.48it/s] 59%|█████▊    | 234356/400000 [00:32<00:22, 7346.98it/s] 59%|█████▉    | 235092/400000 [00:33<00:22, 7344.34it/s] 59%|█████▉    | 235828/400000 [00:33<00:22, 7223.94it/s] 59%|█████▉    | 236569/400000 [00:33<00:22, 7277.57it/s] 59%|█████▉    | 237315/400000 [00:33<00:22, 7330.07it/s] 60%|█████▉    | 238060/400000 [00:33<00:21, 7365.64it/s] 60%|█████▉    | 238798/400000 [00:33<00:22, 7280.45it/s] 60%|█████▉    | 239549/400000 [00:33<00:21, 7345.71it/s] 60%|██████    | 240294/400000 [00:33<00:21, 7375.12it/s] 60%|██████    | 241032/400000 [00:33<00:21, 7373.63it/s] 60%|██████    | 241770/400000 [00:33<00:21, 7305.67it/s] 61%|██████    | 242510/400000 [00:34<00:21, 7332.42it/s] 61%|██████    | 243244/400000 [00:34<00:22, 7031.87it/s] 61%|██████    | 243978/400000 [00:34<00:21, 7119.35it/s] 61%|██████    | 244731/400000 [00:34<00:21, 7237.53it/s] 61%|██████▏   | 245488/400000 [00:34<00:21, 7331.76it/s] 62%|██████▏   | 246223/400000 [00:34<00:21, 7314.11it/s] 62%|██████▏   | 246969/400000 [00:34<00:20, 7355.73it/s] 62%|██████▏   | 247723/400000 [00:34<00:20, 7408.70it/s] 62%|██████▏   | 248470/400000 [00:34<00:20, 7425.97it/s] 62%|██████▏   | 249228/400000 [00:34<00:20, 7469.61it/s] 62%|██████▏   | 249976/400000 [00:35<00:20, 7325.19it/s] 63%|██████▎   | 250710/400000 [00:35<00:20, 7293.30it/s] 63%|██████▎   | 251445/400000 [00:35<00:20, 7308.13it/s] 63%|██████▎   | 252192/400000 [00:35<00:20, 7355.71it/s] 63%|██████▎   | 252928/400000 [00:35<00:20, 7342.39it/s] 63%|██████▎   | 253663/400000 [00:35<00:20, 7264.09it/s] 64%|██████▎   | 254411/400000 [00:35<00:19, 7324.88it/s] 64%|██████▍   | 255164/400000 [00:35<00:19, 7384.81it/s] 64%|██████▍   | 255903/400000 [00:35<00:19, 7371.89it/s] 64%|██████▍   | 256653/400000 [00:36<00:19, 7409.74it/s] 64%|██████▍   | 257395/400000 [00:36<00:19, 7302.84it/s] 65%|██████▍   | 258138/400000 [00:36<00:19, 7339.36it/s] 65%|██████▍   | 258883/400000 [00:36<00:19, 7371.36it/s] 65%|██████▍   | 259628/400000 [00:36<00:18, 7392.15it/s] 65%|██████▌   | 260375/400000 [00:36<00:18, 7413.55it/s] 65%|██████▌   | 261117/400000 [00:36<00:18, 7394.35it/s] 65%|██████▌   | 261857/400000 [00:36<00:18, 7347.02it/s] 66%|██████▌   | 262596/400000 [00:36<00:18, 7358.71it/s] 66%|██████▌   | 263343/400000 [00:36<00:18, 7391.39it/s] 66%|██████▌   | 264083/400000 [00:37<00:18, 7328.30it/s] 66%|██████▌   | 264819/400000 [00:37<00:18, 7335.83it/s] 66%|██████▋   | 265553/400000 [00:37<00:18, 7104.92it/s] 67%|██████▋   | 266266/400000 [00:37<00:19, 6992.50it/s] 67%|██████▋   | 266967/400000 [00:37<00:19, 6985.88it/s] 67%|██████▋   | 267715/400000 [00:37<00:18, 7127.02it/s] 67%|██████▋   | 268470/400000 [00:37<00:18, 7248.65it/s] 67%|██████▋   | 269197/400000 [00:37<00:18, 7152.02it/s] 67%|██████▋   | 269939/400000 [00:37<00:17, 7229.38it/s] 68%|██████▊   | 270664/400000 [00:37<00:17, 7186.06it/s] 68%|██████▊   | 271414/400000 [00:38<00:17, 7276.87it/s] 68%|██████▊   | 272155/400000 [00:38<00:17, 7315.32it/s] 68%|██████▊   | 272888/400000 [00:38<00:17, 7231.42it/s] 68%|██████▊   | 273629/400000 [00:38<00:17, 7283.13it/s] 69%|██████▊   | 274361/400000 [00:38<00:17, 7293.19it/s] 69%|██████▉   | 275108/400000 [00:38<00:17, 7344.27it/s] 69%|██████▉   | 275860/400000 [00:38<00:16, 7394.02it/s] 69%|██████▉   | 276600/400000 [00:38<00:16, 7363.50it/s] 69%|██████▉   | 277343/400000 [00:38<00:16, 7382.54it/s] 70%|██████▉   | 278103/400000 [00:38<00:16, 7445.91it/s] 70%|██████▉   | 278848/400000 [00:39<00:16, 7354.35it/s] 70%|██████▉   | 279590/400000 [00:39<00:16, 7371.87it/s] 70%|███████   | 280330/400000 [00:39<00:16, 7377.43it/s] 70%|███████   | 281068/400000 [00:39<00:16, 7314.80it/s] 70%|███████   | 281827/400000 [00:39<00:15, 7392.51it/s] 71%|███████   | 282567/400000 [00:39<00:15, 7341.75it/s] 71%|███████   | 283318/400000 [00:39<00:15, 7390.57it/s] 71%|███████   | 284058/400000 [00:39<00:15, 7357.07it/s] 71%|███████   | 284806/400000 [00:39<00:15, 7391.52it/s] 71%|███████▏  | 285548/400000 [00:39<00:15, 7399.52it/s] 72%|███████▏  | 286289/400000 [00:40<00:15, 7259.93it/s] 72%|███████▏  | 287016/400000 [00:40<00:15, 7246.79it/s] 72%|███████▏  | 287742/400000 [00:40<00:15, 7149.79it/s] 72%|███████▏  | 288458/400000 [00:40<00:15, 7003.22it/s] 72%|███████▏  | 289160/400000 [00:40<00:15, 6930.64it/s] 72%|███████▏  | 289855/400000 [00:40<00:16, 6878.39it/s] 73%|███████▎  | 290550/400000 [00:40<00:15, 6899.14it/s] 73%|███████▎  | 291263/400000 [00:40<00:15, 6964.77it/s] 73%|███████▎  | 292018/400000 [00:40<00:15, 7128.86it/s] 73%|███████▎  | 292758/400000 [00:40<00:14, 7207.24it/s] 73%|███████▎  | 293518/400000 [00:41<00:14, 7320.42it/s] 74%|███████▎  | 294279/400000 [00:41<00:14, 7402.93it/s] 74%|███████▍  | 295021/400000 [00:41<00:14, 7379.57it/s] 74%|███████▍  | 295760/400000 [00:41<00:14, 7341.46it/s] 74%|███████▍  | 296515/400000 [00:41<00:13, 7402.57it/s] 74%|███████▍  | 297269/400000 [00:41<00:13, 7442.19it/s] 75%|███████▍  | 298014/400000 [00:41<00:14, 7273.69it/s] 75%|███████▍  | 298747/400000 [00:41<00:13, 7290.34it/s] 75%|███████▍  | 299477/400000 [00:41<00:14, 7102.58it/s] 75%|███████▌  | 300189/400000 [00:42<00:14, 6924.76it/s] 75%|███████▌  | 300884/400000 [00:42<00:14, 6869.74it/s] 75%|███████▌  | 301573/400000 [00:42<00:14, 6819.16it/s] 76%|███████▌  | 302278/400000 [00:42<00:14, 6884.38it/s] 76%|███████▌  | 302979/400000 [00:42<00:14, 6919.49it/s] 76%|███████▌  | 303730/400000 [00:42<00:13, 7085.21it/s] 76%|███████▌  | 304441/400000 [00:42<00:13, 6978.35it/s] 76%|███████▋  | 305162/400000 [00:42<00:13, 7045.65it/s] 76%|███████▋  | 305905/400000 [00:42<00:13, 7155.84it/s] 77%|███████▋  | 306652/400000 [00:42<00:12, 7244.60it/s] 77%|███████▋  | 307404/400000 [00:43<00:12, 7322.94it/s] 77%|███████▋  | 308145/400000 [00:43<00:12, 7347.57it/s] 77%|███████▋  | 308881/400000 [00:43<00:13, 6974.22it/s] 77%|███████▋  | 309583/400000 [00:43<00:13, 6839.97it/s] 78%|███████▊  | 310339/400000 [00:43<00:12, 7040.41it/s] 78%|███████▊  | 311048/400000 [00:43<00:12, 6861.80it/s] 78%|███████▊  | 311784/400000 [00:43<00:12, 7002.06it/s] 78%|███████▊  | 312507/400000 [00:43<00:12, 7068.42it/s] 78%|███████▊  | 313243/400000 [00:43<00:12, 7152.51it/s] 78%|███████▊  | 313961/400000 [00:43<00:12, 7151.15it/s] 79%|███████▊  | 314678/400000 [00:44<00:12, 6968.78it/s] 79%|███████▉  | 315377/400000 [00:44<00:12, 6880.88it/s] 79%|███████▉  | 316067/400000 [00:44<00:12, 6814.88it/s] 79%|███████▉  | 316787/400000 [00:44<00:12, 6925.00it/s] 79%|███████▉  | 317517/400000 [00:44<00:11, 7033.21it/s] 80%|███████▉  | 318273/400000 [00:44<00:11, 7182.90it/s] 80%|███████▉  | 319032/400000 [00:44<00:11, 7298.22it/s] 80%|███████▉  | 319764/400000 [00:44<00:11, 7130.07it/s] 80%|████████  | 320480/400000 [00:44<00:11, 6855.56it/s] 80%|████████  | 321170/400000 [00:45<00:11, 6738.61it/s] 80%|████████  | 321847/400000 [00:45<00:11, 6713.20it/s] 81%|████████  | 322521/400000 [00:45<00:11, 6664.52it/s] 81%|████████  | 323251/400000 [00:45<00:11, 6842.00it/s] 81%|████████  | 323994/400000 [00:45<00:10, 7008.35it/s] 81%|████████  | 324698/400000 [00:45<00:10, 6998.04it/s] 81%|████████▏ | 325400/400000 [00:45<00:10, 6877.09it/s] 82%|████████▏ | 326090/400000 [00:45<00:10, 6816.16it/s] 82%|████████▏ | 326773/400000 [00:45<00:10, 6742.64it/s] 82%|████████▏ | 327449/400000 [00:45<00:10, 6615.96it/s] 82%|████████▏ | 328112/400000 [00:46<00:11, 6530.19it/s] 82%|████████▏ | 328789/400000 [00:46<00:10, 6599.21it/s] 82%|████████▏ | 329470/400000 [00:46<00:10, 6658.93it/s] 83%|████████▎ | 330157/400000 [00:46<00:10, 6719.18it/s] 83%|████████▎ | 330895/400000 [00:46<00:10, 6903.03it/s] 83%|████████▎ | 331649/400000 [00:46<00:09, 7081.11it/s] 83%|████████▎ | 332392/400000 [00:46<00:09, 7180.50it/s] 83%|████████▎ | 333142/400000 [00:46<00:09, 7271.41it/s] 83%|████████▎ | 333871/400000 [00:46<00:09, 7167.11it/s] 84%|████████▎ | 334608/400000 [00:46<00:09, 7226.71it/s] 84%|████████▍ | 335333/400000 [00:47<00:08, 7233.24it/s] 84%|████████▍ | 336082/400000 [00:47<00:08, 7306.70it/s] 84%|████████▍ | 336825/400000 [00:47<00:08, 7341.28it/s] 84%|████████▍ | 337560/400000 [00:47<00:08, 7323.36it/s] 85%|████████▍ | 338293/400000 [00:47<00:08, 7196.75it/s] 85%|████████▍ | 339038/400000 [00:47<00:08, 7267.86it/s] 85%|████████▍ | 339770/400000 [00:47<00:08, 7283.29it/s] 85%|████████▌ | 340499/400000 [00:47<00:08, 7264.54it/s] 85%|████████▌ | 341230/400000 [00:47<00:08, 7276.47it/s] 85%|████████▌ | 341962/400000 [00:47<00:07, 7286.59it/s] 86%|████████▌ | 342691/400000 [00:48<00:07, 7247.12it/s] 86%|████████▌ | 343443/400000 [00:48<00:07, 7326.57it/s] 86%|████████▌ | 344189/400000 [00:48<00:07, 7364.51it/s] 86%|████████▌ | 344926/400000 [00:48<00:07, 7152.35it/s] 86%|████████▋ | 345677/400000 [00:48<00:07, 7253.74it/s] 87%|████████▋ | 346404/400000 [00:48<00:07, 7248.47it/s] 87%|████████▋ | 347148/400000 [00:48<00:07, 7303.82it/s] 87%|████████▋ | 347894/400000 [00:48<00:07, 7349.94it/s] 87%|████████▋ | 348633/400000 [00:48<00:06, 7359.27it/s] 87%|████████▋ | 349384/400000 [00:48<00:06, 7401.88it/s] 88%|████████▊ | 350125/400000 [00:49<00:06, 7396.10it/s] 88%|████████▊ | 350865/400000 [00:49<00:06, 7308.03it/s] 88%|████████▊ | 351611/400000 [00:49<00:06, 7351.70it/s] 88%|████████▊ | 352347/400000 [00:49<00:06, 7324.69it/s] 88%|████████▊ | 353087/400000 [00:49<00:06, 7346.66it/s] 88%|████████▊ | 353832/400000 [00:49<00:06, 7374.76it/s] 89%|████████▊ | 354570/400000 [00:49<00:06, 7347.94it/s] 89%|████████▉ | 355305/400000 [00:49<00:06, 7329.61it/s] 89%|████████▉ | 356039/400000 [00:49<00:06, 7315.02it/s] 89%|████████▉ | 356773/400000 [00:49<00:05, 7320.03it/s] 89%|████████▉ | 357506/400000 [00:50<00:05, 7260.18it/s] 90%|████████▉ | 358233/400000 [00:50<00:05, 7112.25it/s] 90%|████████▉ | 358946/400000 [00:50<00:05, 7018.13it/s] 90%|████████▉ | 359649/400000 [00:50<00:05, 6829.56it/s] 90%|█████████ | 360376/400000 [00:50<00:05, 6953.18it/s] 90%|█████████ | 361119/400000 [00:50<00:05, 7088.85it/s] 90%|█████████ | 361875/400000 [00:50<00:05, 7223.02it/s] 91%|█████████ | 362633/400000 [00:50<00:05, 7325.29it/s] 91%|█████████ | 363368/400000 [00:50<00:05, 7282.36it/s] 91%|█████████ | 364124/400000 [00:51<00:04, 7360.77it/s] 91%|█████████ | 364862/400000 [00:51<00:04, 7350.05it/s] 91%|█████████▏| 365598/400000 [00:51<00:04, 7267.16it/s] 92%|█████████▏| 366326/400000 [00:51<00:04, 7103.98it/s] 92%|█████████▏| 367038/400000 [00:51<00:04, 6963.46it/s] 92%|█████████▏| 367736/400000 [00:51<00:04, 6939.22it/s] 92%|█████████▏| 368478/400000 [00:51<00:04, 7075.68it/s] 92%|█████████▏| 369190/400000 [00:51<00:04, 7088.58it/s] 92%|█████████▏| 369900/400000 [00:51<00:04, 6970.25it/s] 93%|█████████▎| 370599/400000 [00:51<00:04, 6711.03it/s] 93%|█████████▎| 371273/400000 [00:52<00:04, 6679.50it/s] 93%|█████████▎| 371943/400000 [00:52<00:04, 6597.91it/s] 93%|█████████▎| 372614/400000 [00:52<00:04, 6628.66it/s] 93%|█████████▎| 373284/400000 [00:52<00:04, 6639.91it/s] 93%|█████████▎| 373949/400000 [00:52<00:03, 6628.22it/s] 94%|█████████▎| 374624/400000 [00:52<00:03, 6662.61it/s] 94%|█████████▍| 375374/400000 [00:52<00:03, 6892.73it/s] 94%|█████████▍| 376124/400000 [00:52<00:03, 7062.09it/s] 94%|█████████▍| 376850/400000 [00:52<00:03, 7118.78it/s] 94%|█████████▍| 377564/400000 [00:52<00:03, 6985.35it/s] 95%|█████████▍| 378305/400000 [00:53<00:03, 7105.28it/s] 95%|█████████▍| 379070/400000 [00:53<00:02, 7259.81it/s] 95%|█████████▍| 379823/400000 [00:53<00:02, 7336.50it/s] 95%|█████████▌| 380559/400000 [00:53<00:02, 7331.76it/s] 95%|█████████▌| 381301/400000 [00:53<00:02, 7350.71it/s] 96%|█████████▌| 382037/400000 [00:53<00:02, 7316.99it/s] 96%|█████████▌| 382803/400000 [00:53<00:02, 7413.99it/s] 96%|█████████▌| 383551/400000 [00:53<00:02, 7433.54it/s] 96%|█████████▌| 384295/400000 [00:53<00:02, 7343.42it/s] 96%|█████████▋| 385030/400000 [00:53<00:02, 7070.68it/s] 96%|█████████▋| 385778/400000 [00:54<00:01, 7188.49it/s] 97%|█████████▋| 386539/400000 [00:54<00:01, 7307.55it/s] 97%|█████████▋| 387295/400000 [00:54<00:01, 7381.49it/s] 97%|█████████▋| 388057/400000 [00:54<00:01, 7449.98it/s] 97%|█████████▋| 388804/400000 [00:54<00:01, 7418.24it/s] 97%|█████████▋| 389547/400000 [00:54<00:01, 7241.95it/s] 98%|█████████▊| 390304/400000 [00:54<00:01, 7335.72it/s] 98%|█████████▊| 391063/400000 [00:54<00:01, 7407.83it/s] 98%|█████████▊| 391824/400000 [00:54<00:01, 7466.87it/s] 98%|█████████▊| 392572/400000 [00:54<00:01, 7288.31it/s] 98%|█████████▊| 393303/400000 [00:55<00:00, 7014.02it/s] 99%|█████████▊| 394015/400000 [00:55<00:00, 7045.12it/s] 99%|█████████▊| 394722/400000 [00:55<00:00, 6994.21it/s] 99%|█████████▉| 395477/400000 [00:55<00:00, 7150.20it/s] 99%|█████████▉| 396215/400000 [00:55<00:00, 7215.31it/s] 99%|█████████▉| 396966/400000 [00:55<00:00, 7300.22it/s] 99%|█████████▉| 397698/400000 [00:55<00:00, 7171.15it/s]100%|█████████▉| 398417/400000 [00:55<00:00, 6877.15it/s]100%|█████████▉| 399109/400000 [00:55<00:00, 6736.53it/s]100%|█████████▉| 399794/400000 [00:56<00:00, 6767.70it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7133.45it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fd5228720f0>, <torchtext.data.dataset.TabularDataset object at 0x7fd522872240>, <torchtext.vocab.Vocab object at 0x7fd522872160>), {}) 

