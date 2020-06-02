
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc694e30378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc694e30378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc706e800d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc706e800d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc69e6169d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc69e6169d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc6b47648c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc6b47648c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc6b47648c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3301376/9912422 [00:00<00:00, 32957216.62it/s]9920512it [00:00, 32994538.26it/s]                             
0it [00:00, ?it/s]32768it [00:00, 478791.28it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7616535.68it/s]          
0it [00:00, ?it/s]8192it [00:00, 114915.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc694e35978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc694fc0d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc6b4764510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc6b4764510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc6b4764510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:44,  3.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:44,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:20,  6.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 12.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 16.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 33.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 46.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 67.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 73.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 78.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 82.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.66 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.49s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.66 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ url]
0 examples [00:00, ? examples/s]2020-06-02 06:10:05.855947: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-02 06:10:05.871548: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-02 06:10:05.871786: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5563a800a0e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-02 06:10:05.871806: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
17 examples [00:00, 168.61 examples/s]107 examples [00:00, 222.96 examples/s]200 examples [00:00, 288.65 examples/s]284 examples [00:00, 359.14 examples/s]369 examples [00:00, 434.36 examples/s]454 examples [00:00, 508.25 examples/s]544 examples [00:00, 584.24 examples/s]625 examples [00:00, 636.71 examples/s]714 examples [00:00, 694.82 examples/s]804 examples [00:01, 744.72 examples/s]893 examples [00:01, 782.46 examples/s]979 examples [00:01, 800.78 examples/s]1071 examples [00:01, 831.24 examples/s]1158 examples [00:01, 827.19 examples/s]1244 examples [00:01, 832.48 examples/s]1335 examples [00:01, 852.25 examples/s]1427 examples [00:01, 870.88 examples/s]1518 examples [00:01, 880.21 examples/s]1611 examples [00:01, 892.50 examples/s]1704 examples [00:02, 901.48 examples/s]1795 examples [00:02, 891.34 examples/s]1885 examples [00:02, 889.47 examples/s]1978 examples [00:02, 900.70 examples/s]2070 examples [00:02, 906.33 examples/s]2161 examples [00:02, 901.40 examples/s]2257 examples [00:02, 916.51 examples/s]2349 examples [00:02, 910.30 examples/s]2441 examples [00:02, 907.01 examples/s]2532 examples [00:02, 904.80 examples/s]2623 examples [00:03, 893.95 examples/s]2713 examples [00:03, 877.97 examples/s]2801 examples [00:03, 841.10 examples/s]2889 examples [00:03, 851.86 examples/s]2986 examples [00:03, 882.38 examples/s]3075 examples [00:03, 865.68 examples/s]3170 examples [00:03, 886.37 examples/s]3263 examples [00:03, 897.22 examples/s]3354 examples [00:03, 873.57 examples/s]3444 examples [00:03, 880.18 examples/s]3533 examples [00:04, 837.96 examples/s]3626 examples [00:04, 861.55 examples/s]3717 examples [00:04, 873.97 examples/s]3808 examples [00:04, 883.59 examples/s]3897 examples [00:04, 882.82 examples/s]3987 examples [00:04, 887.87 examples/s]4077 examples [00:04, 891.44 examples/s]4167 examples [00:04, 889.90 examples/s]4257 examples [00:04, 889.90 examples/s]4347 examples [00:05, 892.22 examples/s]4437 examples [00:05, 893.89 examples/s]4527 examples [00:05, 891.89 examples/s]4617 examples [00:05, 856.41 examples/s]4706 examples [00:05, 865.53 examples/s]4797 examples [00:05, 876.54 examples/s]4885 examples [00:05, 868.71 examples/s]4974 examples [00:05, 872.45 examples/s]5066 examples [00:05, 883.52 examples/s]5156 examples [00:05, 885.97 examples/s]5247 examples [00:06, 893.02 examples/s]5338 examples [00:06, 896.72 examples/s]5430 examples [00:06, 902.05 examples/s]5522 examples [00:06, 905.77 examples/s]5613 examples [00:06, 902.38 examples/s]5704 examples [00:06, 902.29 examples/s]5795 examples [00:06, 886.19 examples/s]5887 examples [00:06, 893.28 examples/s]5978 examples [00:06, 897.95 examples/s]6068 examples [00:06, 874.51 examples/s]6160 examples [00:07, 886.95 examples/s]6249 examples [00:07, 880.45 examples/s]6342 examples [00:07, 891.97 examples/s]6433 examples [00:07, 896.13 examples/s]6530 examples [00:07, 915.99 examples/s]6622 examples [00:07, 905.43 examples/s]6715 examples [00:07, 911.06 examples/s]6809 examples [00:07, 917.84 examples/s]6901 examples [00:07, 884.96 examples/s]6990 examples [00:07, 882.94 examples/s]7081 examples [00:08, 888.36 examples/s]7172 examples [00:08, 893.77 examples/s]7262 examples [00:08, 886.37 examples/s]7351 examples [00:08, 865.74 examples/s]7438 examples [00:08, 859.38 examples/s]7525 examples [00:08, 850.01 examples/s]7611 examples [00:08, 839.02 examples/s]7697 examples [00:08, 843.42 examples/s]7782 examples [00:08, 820.17 examples/s]7865 examples [00:09, 815.79 examples/s]7948 examples [00:09, 818.61 examples/s]8036 examples [00:09, 835.20 examples/s]8122 examples [00:09, 841.83 examples/s]8213 examples [00:09, 857.78 examples/s]8301 examples [00:09, 863.68 examples/s]8388 examples [00:09, 864.57 examples/s]8475 examples [00:09, 839.71 examples/s]8563 examples [00:09, 851.10 examples/s]8650 examples [00:09, 855.05 examples/s]8736 examples [00:10, 833.65 examples/s]8827 examples [00:10, 854.54 examples/s]8918 examples [00:10, 869.05 examples/s]9011 examples [00:10, 884.24 examples/s]9104 examples [00:10, 896.54 examples/s]9195 examples [00:10, 897.98 examples/s]9285 examples [00:10, 884.55 examples/s]9376 examples [00:10, 891.27 examples/s]9468 examples [00:10, 899.13 examples/s]9560 examples [00:10, 905.29 examples/s]9654 examples [00:11, 915.15 examples/s]9746 examples [00:11, 915.00 examples/s]9838 examples [00:11, 908.39 examples/s]9931 examples [00:11, 913.76 examples/s]10023 examples [00:11, 862.01 examples/s]10116 examples [00:11, 879.98 examples/s]10205 examples [00:11, 853.14 examples/s]10291 examples [00:11, 823.58 examples/s]10385 examples [00:11, 854.14 examples/s]10472 examples [00:12, 850.47 examples/s]10564 examples [00:12, 869.50 examples/s]10657 examples [00:12, 885.72 examples/s]10749 examples [00:12, 895.19 examples/s]10839 examples [00:12, 891.69 examples/s]10929 examples [00:12, 888.75 examples/s]11021 examples [00:12, 895.74 examples/s]11111 examples [00:12, 876.19 examples/s]11201 examples [00:12, 880.86 examples/s]11294 examples [00:12, 895.03 examples/s]11385 examples [00:13, 897.17 examples/s]11475 examples [00:13, 888.22 examples/s]11564 examples [00:13, 885.63 examples/s]11654 examples [00:13, 889.65 examples/s]11744 examples [00:13, 874.48 examples/s]11839 examples [00:13, 894.03 examples/s]11934 examples [00:13, 907.68 examples/s]12025 examples [00:13, 906.76 examples/s]12116 examples [00:13, 901.99 examples/s]12207 examples [00:13, 903.85 examples/s]12298 examples [00:14, 905.38 examples/s]12390 examples [00:14, 907.64 examples/s]12481 examples [00:14, 908.26 examples/s]12572 examples [00:14, 900.69 examples/s]12665 examples [00:14, 907.00 examples/s]12756 examples [00:14, 901.25 examples/s]12848 examples [00:14, 904.61 examples/s]12939 examples [00:14, 905.30 examples/s]13030 examples [00:14, 898.96 examples/s]13120 examples [00:14, 893.91 examples/s]13210 examples [00:15, 891.42 examples/s]13300 examples [00:15, 884.42 examples/s]13389 examples [00:15, 878.43 examples/s]13477 examples [00:15, 843.19 examples/s]13562 examples [00:15, 791.57 examples/s]13649 examples [00:15, 812.42 examples/s]13734 examples [00:15, 821.60 examples/s]13820 examples [00:15, 831.98 examples/s]13912 examples [00:15, 854.80 examples/s]14003 examples [00:15, 868.22 examples/s]14093 examples [00:16, 874.95 examples/s]14181 examples [00:16, 872.02 examples/s]14269 examples [00:16, 871.66 examples/s]14357 examples [00:16, 872.12 examples/s]14446 examples [00:16, 876.71 examples/s]14536 examples [00:16, 876.86 examples/s]14624 examples [00:16, 863.40 examples/s]14711 examples [00:16, 793.22 examples/s]14800 examples [00:16, 819.14 examples/s]14891 examples [00:17, 843.50 examples/s]14983 examples [00:17, 862.57 examples/s]15071 examples [00:17, 867.30 examples/s]15159 examples [00:17, 854.10 examples/s]15245 examples [00:17, 823.34 examples/s]15328 examples [00:17, 819.12 examples/s]15417 examples [00:17, 830.47 examples/s]15505 examples [00:17, 843.99 examples/s]15594 examples [00:17, 854.38 examples/s]15683 examples [00:17, 864.37 examples/s]15777 examples [00:18, 884.69 examples/s]15866 examples [00:18, 878.09 examples/s]15954 examples [00:18, 861.83 examples/s]16048 examples [00:18, 881.76 examples/s]16137 examples [00:18, 856.58 examples/s]16223 examples [00:18, 838.62 examples/s]16315 examples [00:18, 860.47 examples/s]16402 examples [00:18, 852.83 examples/s]16494 examples [00:18, 870.28 examples/s]16582 examples [00:19, 872.79 examples/s]16670 examples [00:19, 846.11 examples/s]16756 examples [00:19, 849.13 examples/s]16842 examples [00:19, 793.93 examples/s]16934 examples [00:19, 826.44 examples/s]17023 examples [00:19, 843.85 examples/s]17117 examples [00:19, 869.58 examples/s]17207 examples [00:19, 876.17 examples/s]17296 examples [00:19, 879.27 examples/s]17385 examples [00:19, 881.24 examples/s]17478 examples [00:20, 893.55 examples/s]17569 examples [00:20, 896.39 examples/s]17659 examples [00:20, 895.40 examples/s]17749 examples [00:20, 863.51 examples/s]17838 examples [00:20, 868.67 examples/s]17932 examples [00:20, 887.68 examples/s]18022 examples [00:20, 875.09 examples/s]18111 examples [00:20, 877.84 examples/s]18199 examples [00:20, 871.58 examples/s]18287 examples [00:20, 856.54 examples/s]18373 examples [00:21, 843.81 examples/s]18460 examples [00:21, 849.79 examples/s]18553 examples [00:21, 870.60 examples/s]18643 examples [00:21, 876.51 examples/s]18735 examples [00:21, 887.84 examples/s]18828 examples [00:21, 899.39 examples/s]18921 examples [00:21, 905.28 examples/s]19013 examples [00:21, 908.49 examples/s]19104 examples [00:21, 897.47 examples/s]19194 examples [00:21, 885.39 examples/s]19286 examples [00:22, 894.04 examples/s]19376 examples [00:22, 891.35 examples/s]19466 examples [00:22, 844.85 examples/s]19558 examples [00:22, 864.86 examples/s]19645 examples [00:22, 843.68 examples/s]19737 examples [00:22, 863.33 examples/s]19826 examples [00:22, 870.67 examples/s]19918 examples [00:22, 884.31 examples/s]20007 examples [00:22, 831.60 examples/s]20098 examples [00:23, 853.65 examples/s]20192 examples [00:23, 876.44 examples/s]20281 examples [00:23, 875.15 examples/s]20373 examples [00:23, 887.30 examples/s]20463 examples [00:23, 887.64 examples/s]20554 examples [00:23, 891.95 examples/s]20644 examples [00:23, 865.22 examples/s]20731 examples [00:23, 845.67 examples/s]20816 examples [00:23, 823.15 examples/s]20909 examples [00:23, 851.40 examples/s]20999 examples [00:24, 863.52 examples/s]21088 examples [00:24, 869.08 examples/s]21176 examples [00:24, 871.78 examples/s]21266 examples [00:24, 878.20 examples/s]21359 examples [00:24, 892.54 examples/s]21453 examples [00:24, 903.73 examples/s]21544 examples [00:24, 900.39 examples/s]21635 examples [00:24, 900.31 examples/s]21728 examples [00:24, 906.71 examples/s]21819 examples [00:25, 893.35 examples/s]21911 examples [00:25, 901.03 examples/s]22003 examples [00:25, 905.48 examples/s]22094 examples [00:25, 857.24 examples/s]22181 examples [00:25, 822.13 examples/s]22278 examples [00:25, 859.46 examples/s]22365 examples [00:25, 855.39 examples/s]22452 examples [00:25, 843.34 examples/s]22545 examples [00:25, 867.11 examples/s]22633 examples [00:25, 866.88 examples/s]22721 examples [00:26, 869.69 examples/s]22810 examples [00:26, 873.09 examples/s]22898 examples [00:26, 851.69 examples/s]22984 examples [00:26, 781.31 examples/s]23064 examples [00:26, 776.74 examples/s]23155 examples [00:26, 810.75 examples/s]23247 examples [00:26, 839.15 examples/s]23338 examples [00:26, 857.61 examples/s]23428 examples [00:26, 866.75 examples/s]23516 examples [00:27, 860.50 examples/s]23603 examples [00:27, 820.47 examples/s]23688 examples [00:27, 827.69 examples/s]23773 examples [00:27, 833.35 examples/s]23857 examples [00:27, 825.00 examples/s]23947 examples [00:27, 844.89 examples/s]24040 examples [00:27, 867.93 examples/s]24131 examples [00:27, 879.37 examples/s]24221 examples [00:27, 884.88 examples/s]24312 examples [00:27, 889.95 examples/s]24406 examples [00:28, 902.28 examples/s]24497 examples [00:28, 878.96 examples/s]24586 examples [00:28, 878.85 examples/s]24675 examples [00:28, 858.40 examples/s]24762 examples [00:28, 848.84 examples/s]24848 examples [00:28, 852.15 examples/s]24935 examples [00:28, 856.05 examples/s]25025 examples [00:28, 866.32 examples/s]25112 examples [00:28, 865.31 examples/s]25202 examples [00:28, 873.86 examples/s]25291 examples [00:29, 877.76 examples/s]25379 examples [00:29, 862.71 examples/s]25472 examples [00:29, 879.46 examples/s]25561 examples [00:29, 877.81 examples/s]25649 examples [00:29, 875.62 examples/s]25738 examples [00:29, 879.39 examples/s]25828 examples [00:29, 883.56 examples/s]25921 examples [00:29, 894.82 examples/s]26012 examples [00:29, 897.20 examples/s]26102 examples [00:29, 892.10 examples/s]26192 examples [00:30, 891.84 examples/s]26282 examples [00:30, 887.11 examples/s]26373 examples [00:30, 893.75 examples/s]26463 examples [00:30, 873.91 examples/s]26551 examples [00:30, 840.08 examples/s]26636 examples [00:30, 823.55 examples/s]26719 examples [00:30, 805.05 examples/s]26813 examples [00:30, 839.46 examples/s]26900 examples [00:30, 847.26 examples/s]26986 examples [00:31, 829.78 examples/s]27070 examples [00:31, 825.56 examples/s]27159 examples [00:31, 841.67 examples/s]27246 examples [00:31, 848.17 examples/s]27340 examples [00:31, 871.53 examples/s]27428 examples [00:31, 846.90 examples/s]27514 examples [00:31, 848.87 examples/s]27604 examples [00:31, 861.07 examples/s]27691 examples [00:31, 858.25 examples/s]27785 examples [00:31, 880.88 examples/s]27874 examples [00:32, 867.39 examples/s]27961 examples [00:32, 861.22 examples/s]28048 examples [00:32, 862.45 examples/s]28135 examples [00:32, 860.50 examples/s]28222 examples [00:32, 853.95 examples/s]28311 examples [00:32, 863.43 examples/s]28399 examples [00:32, 868.24 examples/s]28488 examples [00:32, 872.54 examples/s]28576 examples [00:32, 852.64 examples/s]28666 examples [00:32, 863.75 examples/s]28756 examples [00:33, 870.51 examples/s]28846 examples [00:33, 876.65 examples/s]28938 examples [00:33, 879.27 examples/s]29026 examples [00:33, 870.68 examples/s]29116 examples [00:33, 876.81 examples/s]29207 examples [00:33, 885.38 examples/s]29296 examples [00:33, 884.58 examples/s]29385 examples [00:33, 865.14 examples/s]29472 examples [00:33, 845.61 examples/s]29566 examples [00:34, 863.49 examples/s]29653 examples [00:34, 852.31 examples/s]29741 examples [00:34, 860.08 examples/s]29828 examples [00:34, 861.53 examples/s]29921 examples [00:34, 880.81 examples/s]30010 examples [00:34, 811.26 examples/s]30102 examples [00:34, 839.70 examples/s]30193 examples [00:34, 857.68 examples/s]30287 examples [00:34, 879.17 examples/s]30376 examples [00:34, 871.92 examples/s]30465 examples [00:35, 877.15 examples/s]30554 examples [00:35, 868.64 examples/s]30646 examples [00:35, 881.10 examples/s]30735 examples [00:35, 859.04 examples/s]30822 examples [00:35, 842.41 examples/s]30909 examples [00:35, 848.87 examples/s]30995 examples [00:35, 831.08 examples/s]31087 examples [00:35, 854.08 examples/s]31179 examples [00:35, 871.67 examples/s]31270 examples [00:35, 882.44 examples/s]31360 examples [00:36, 887.62 examples/s]31450 examples [00:36, 889.10 examples/s]31542 examples [00:36, 896.94 examples/s]31633 examples [00:36, 900.81 examples/s]31724 examples [00:36, 895.40 examples/s]31814 examples [00:36, 866.44 examples/s]31901 examples [00:36, 862.53 examples/s]31990 examples [00:36, 870.54 examples/s]32080 examples [00:36, 876.63 examples/s]32174 examples [00:36, 894.45 examples/s]32264 examples [00:37, 878.47 examples/s]32353 examples [00:37, 852.54 examples/s]32439 examples [00:37, 848.17 examples/s]32525 examples [00:37, 849.65 examples/s]32614 examples [00:37, 859.91 examples/s]32701 examples [00:37, 851.08 examples/s]32791 examples [00:37, 862.59 examples/s]32880 examples [00:37, 868.26 examples/s]32969 examples [00:37, 872.48 examples/s]33057 examples [00:38, 873.76 examples/s]33146 examples [00:38, 876.93 examples/s]33234 examples [00:38, 876.32 examples/s]33323 examples [00:38, 879.84 examples/s]33414 examples [00:38, 887.66 examples/s]33505 examples [00:38, 893.30 examples/s]33596 examples [00:38, 897.23 examples/s]33686 examples [00:38, 883.37 examples/s]33775 examples [00:38, 870.12 examples/s]33863 examples [00:38, 861.74 examples/s]33956 examples [00:39, 879.75 examples/s]34050 examples [00:39, 894.70 examples/s]34140 examples [00:39, 880.17 examples/s]34229 examples [00:39, 875.84 examples/s]34317 examples [00:39, 869.62 examples/s]34405 examples [00:39, 870.60 examples/s]34494 examples [00:39, 874.04 examples/s]34582 examples [00:39, 861.98 examples/s]34673 examples [00:39, 873.90 examples/s]34765 examples [00:39, 884.68 examples/s]34857 examples [00:40, 892.15 examples/s]34948 examples [00:40, 894.59 examples/s]35042 examples [00:40, 904.53 examples/s]35133 examples [00:40, 901.16 examples/s]35224 examples [00:40, 881.52 examples/s]35313 examples [00:40, 862.46 examples/s]35400 examples [00:40, 848.79 examples/s]35487 examples [00:40, 853.78 examples/s]35578 examples [00:40, 867.93 examples/s]35671 examples [00:40, 883.42 examples/s]35760 examples [00:41, 870.39 examples/s]35848 examples [00:41, 820.06 examples/s]35938 examples [00:41, 839.75 examples/s]36026 examples [00:41, 850.70 examples/s]36117 examples [00:41, 866.61 examples/s]36206 examples [00:41, 871.36 examples/s]36300 examples [00:41, 888.26 examples/s]36390 examples [00:41, 891.73 examples/s]36480 examples [00:41, 894.03 examples/s]36570 examples [00:42, 894.29 examples/s]36660 examples [00:42, 891.54 examples/s]36750 examples [00:42, 883.72 examples/s]36839 examples [00:42, 852.82 examples/s]36925 examples [00:42, 849.07 examples/s]37015 examples [00:42, 863.15 examples/s]37102 examples [00:42, 863.88 examples/s]37191 examples [00:42, 869.41 examples/s]37279 examples [00:42, 872.28 examples/s]37371 examples [00:42, 885.08 examples/s]37465 examples [00:43, 898.24 examples/s]37557 examples [00:43, 903.52 examples/s]37648 examples [00:43, 884.10 examples/s]37737 examples [00:43, 855.59 examples/s]37828 examples [00:43, 869.44 examples/s]37916 examples [00:43, 857.27 examples/s]38005 examples [00:43, 861.56 examples/s]38092 examples [00:43, 852.01 examples/s]38179 examples [00:43, 856.24 examples/s]38265 examples [00:43, 845.32 examples/s]38356 examples [00:44, 861.74 examples/s]38446 examples [00:44, 870.92 examples/s]38536 examples [00:44, 877.61 examples/s]38624 examples [00:44, 872.73 examples/s]38717 examples [00:44, 887.44 examples/s]38809 examples [00:44, 895.03 examples/s]38903 examples [00:44, 906.87 examples/s]38994 examples [00:44, 907.38 examples/s]39085 examples [00:44, 906.80 examples/s]39176 examples [00:45, 888.91 examples/s]39269 examples [00:45, 900.76 examples/s]39365 examples [00:45, 916.10 examples/s]39457 examples [00:45, 892.65 examples/s]39552 examples [00:45, 906.90 examples/s]39648 examples [00:45, 919.60 examples/s]39741 examples [00:45, 914.43 examples/s]39833 examples [00:45, 908.16 examples/s]39924 examples [00:45, 906.69 examples/s]40015 examples [00:45, 852.90 examples/s]40108 examples [00:46, 872.67 examples/s]40200 examples [00:46, 883.76 examples/s]40292 examples [00:46, 893.09 examples/s]40383 examples [00:46, 897.14 examples/s]40473 examples [00:46, 889.89 examples/s]40563 examples [00:46, 890.67 examples/s]40655 examples [00:46, 896.92 examples/s]40745 examples [00:46, 870.42 examples/s]40837 examples [00:46, 884.29 examples/s]40930 examples [00:46, 894.84 examples/s]41024 examples [00:47, 905.90 examples/s]41117 examples [00:47, 911.65 examples/s]41209 examples [00:47, 904.50 examples/s]41300 examples [00:47, 882.16 examples/s]41389 examples [00:47, 873.32 examples/s]41477 examples [00:47, 828.39 examples/s]41563 examples [00:47, 836.40 examples/s]41654 examples [00:47, 856.43 examples/s]41741 examples [00:47, 837.44 examples/s]41831 examples [00:48, 853.20 examples/s]41922 examples [00:48, 867.10 examples/s]42013 examples [00:48, 877.72 examples/s]42102 examples [00:48, 881.29 examples/s]42191 examples [00:48, 879.62 examples/s]42280 examples [00:48, 858.01 examples/s]42372 examples [00:48, 873.94 examples/s]42462 examples [00:48, 880.98 examples/s]42554 examples [00:48, 890.92 examples/s]42644 examples [00:48, 883.44 examples/s]42736 examples [00:49, 893.72 examples/s]42826 examples [00:49, 855.12 examples/s]42918 examples [00:49, 871.69 examples/s]43008 examples [00:49, 879.98 examples/s]43102 examples [00:49, 894.65 examples/s]43192 examples [00:49, 885.00 examples/s]43281 examples [00:49, 884.09 examples/s]43372 examples [00:49, 888.91 examples/s]43461 examples [00:49, 872.45 examples/s]43549 examples [00:49, 869.28 examples/s]43639 examples [00:50, 877.96 examples/s]43728 examples [00:50, 880.78 examples/s]43817 examples [00:50, 871.05 examples/s]43907 examples [00:50, 876.33 examples/s]43995 examples [00:50, 874.28 examples/s]44083 examples [00:50, 864.72 examples/s]44170 examples [00:50, 854.76 examples/s]44257 examples [00:50, 858.71 examples/s]44346 examples [00:50, 866.25 examples/s]44435 examples [00:50, 870.39 examples/s]44525 examples [00:51, 878.21 examples/s]44613 examples [00:51, 877.01 examples/s]44703 examples [00:51, 881.96 examples/s]44792 examples [00:51, 878.70 examples/s]44880 examples [00:51, 837.15 examples/s]44966 examples [00:51, 843.25 examples/s]45057 examples [00:51, 860.63 examples/s]45149 examples [00:51, 876.93 examples/s]45237 examples [00:51, 844.03 examples/s]45328 examples [00:52, 861.45 examples/s]45417 examples [00:52, 869.52 examples/s]45505 examples [00:52, 869.36 examples/s]45593 examples [00:52, 860.57 examples/s]45680 examples [00:52, 853.46 examples/s]45766 examples [00:52, 855.30 examples/s]45854 examples [00:52, 861.23 examples/s]45945 examples [00:52, 874.69 examples/s]46033 examples [00:52, 863.49 examples/s]46120 examples [00:52, 815.65 examples/s]46213 examples [00:53, 844.89 examples/s]46303 examples [00:53, 860.65 examples/s]46390 examples [00:53, 819.93 examples/s]46481 examples [00:53, 843.40 examples/s]46573 examples [00:53, 864.76 examples/s]46665 examples [00:53, 878.25 examples/s]46755 examples [00:53, 882.34 examples/s]46848 examples [00:53, 893.62 examples/s]46938 examples [00:53, 869.84 examples/s]47028 examples [00:53, 877.26 examples/s]47116 examples [00:54, 873.87 examples/s]47208 examples [00:54, 885.80 examples/s]47299 examples [00:54, 890.08 examples/s]47390 examples [00:54, 895.54 examples/s]47480 examples [00:54, 873.43 examples/s]47571 examples [00:54, 882.85 examples/s]47660 examples [00:54, 880.80 examples/s]47753 examples [00:54, 893.01 examples/s]47843 examples [00:54, 889.65 examples/s]47933 examples [00:55, 857.11 examples/s]48022 examples [00:55, 865.58 examples/s]48109 examples [00:55, 865.08 examples/s]48199 examples [00:55, 873.70 examples/s]48291 examples [00:55, 884.62 examples/s]48380 examples [00:55, 870.15 examples/s]48468 examples [00:55, 869.13 examples/s]48556 examples [00:55, 861.27 examples/s]48644 examples [00:55, 864.85 examples/s]48731 examples [00:55, 830.19 examples/s]48820 examples [00:56, 845.24 examples/s]48908 examples [00:56, 855.18 examples/s]48995 examples [00:56, 858.31 examples/s]49085 examples [00:56, 870.08 examples/s]49175 examples [00:56, 876.48 examples/s]49266 examples [00:56, 883.87 examples/s]49358 examples [00:56, 892.03 examples/s]49450 examples [00:56, 897.67 examples/s]49542 examples [00:56, 901.89 examples/s]49633 examples [00:56, 903.07 examples/s]49724 examples [00:57, 898.65 examples/s]49815 examples [00:57, 899.61 examples/s]49905 examples [00:57, 854.28 examples/s]49991 examples [00:57, 851.90 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5528/50000 [00:00<00:00, 55275.44 examples/s] 33%|███▎      | 16348/50000 [00:00<00:00, 64780.69 examples/s] 55%|█████▍    | 27402/50000 [00:00<00:00, 73965.64 examples/s] 77%|███████▋  | 38458/50000 [00:00<00:00, 82119.38 examples/s] 99%|█████████▉| 49610/50000 [00:00<00:00, 89171.20 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 716.95 examples/s]167 examples [00:00, 771.93 examples/s]259 examples [00:00, 809.73 examples/s]350 examples [00:00, 837.27 examples/s]430 examples [00:00, 824.09 examples/s]524 examples [00:00, 853.78 examples/s]610 examples [00:00, 854.69 examples/s]705 examples [00:00, 879.94 examples/s]801 examples [00:00, 900.48 examples/s]894 examples [00:01, 907.61 examples/s]984 examples [00:01, 877.22 examples/s]1071 examples [00:01, 872.79 examples/s]1158 examples [00:01, 866.54 examples/s]1248 examples [00:01, 875.23 examples/s]1339 examples [00:01, 882.99 examples/s]1428 examples [00:01, 688.63 examples/s]1507 examples [00:01, 716.18 examples/s]1597 examples [00:01, 761.80 examples/s]1684 examples [00:02, 789.28 examples/s]1773 examples [00:02, 816.77 examples/s]1864 examples [00:02, 841.32 examples/s]1951 examples [00:02, 843.61 examples/s]2037 examples [00:02, 831.34 examples/s]2127 examples [00:02, 849.96 examples/s]2218 examples [00:02, 866.49 examples/s]2310 examples [00:02, 881.88 examples/s]2399 examples [00:02, 873.08 examples/s]2488 examples [00:02, 875.93 examples/s]2576 examples [00:03, 860.26 examples/s]2664 examples [00:03, 865.85 examples/s]2751 examples [00:03, 855.05 examples/s]2840 examples [00:03, 864.97 examples/s]2931 examples [00:03, 876.29 examples/s]3023 examples [00:03, 888.91 examples/s]3116 examples [00:03, 900.66 examples/s]3208 examples [00:03, 904.05 examples/s]3299 examples [00:03, 904.51 examples/s]3391 examples [00:03, 908.73 examples/s]3483 examples [00:04, 910.26 examples/s]3575 examples [00:04, 878.45 examples/s]3664 examples [00:04, 877.57 examples/s]3755 examples [00:04, 884.65 examples/s]3847 examples [00:04, 894.02 examples/s]3940 examples [00:04, 901.58 examples/s]4031 examples [00:04, 848.31 examples/s]4117 examples [00:04, 815.78 examples/s]4207 examples [00:04, 839.26 examples/s]4296 examples [00:05, 850.98 examples/s]4388 examples [00:05, 869.99 examples/s]4479 examples [00:05, 879.36 examples/s]4570 examples [00:05, 885.95 examples/s]4659 examples [00:05, 865.57 examples/s]4746 examples [00:05, 861.98 examples/s]4836 examples [00:05, 871.95 examples/s]4928 examples [00:05, 884.00 examples/s]5020 examples [00:05, 892.64 examples/s]5110 examples [00:05, 893.98 examples/s]5202 examples [00:06, 899.07 examples/s]5293 examples [00:06, 901.31 examples/s]5384 examples [00:06, 887.12 examples/s]5475 examples [00:06, 893.00 examples/s]5565 examples [00:06, 892.95 examples/s]5658 examples [00:06, 901.15 examples/s]5751 examples [00:06, 908.16 examples/s]5842 examples [00:06, 905.86 examples/s]5933 examples [00:06, 895.83 examples/s]6023 examples [00:06, 886.99 examples/s]6112 examples [00:07, 884.70 examples/s]6205 examples [00:07, 893.99 examples/s]6295 examples [00:07, 867.28 examples/s]6386 examples [00:07, 877.18 examples/s]6477 examples [00:07, 884.78 examples/s]6570 examples [00:07, 896.41 examples/s]6662 examples [00:07, 902.84 examples/s]6753 examples [00:07, 899.80 examples/s]6845 examples [00:07, 904.01 examples/s]6936 examples [00:07, 875.80 examples/s]7026 examples [00:08, 882.67 examples/s]7119 examples [00:08, 895.86 examples/s]7210 examples [00:08, 898.44 examples/s]7302 examples [00:08, 902.24 examples/s]7393 examples [00:08, 903.54 examples/s]7484 examples [00:08, 899.05 examples/s]7575 examples [00:08, 901.25 examples/s]7666 examples [00:08, 897.78 examples/s]7756 examples [00:08, 872.15 examples/s]7845 examples [00:08, 877.30 examples/s]7933 examples [00:09, 868.99 examples/s]8023 examples [00:09, 876.81 examples/s]8111 examples [00:09, 877.73 examples/s]8199 examples [00:09, 854.42 examples/s]8292 examples [00:09, 873.82 examples/s]8385 examples [00:09, 888.52 examples/s]8475 examples [00:09, 886.20 examples/s]8564 examples [00:09, 881.32 examples/s]8657 examples [00:09, 892.66 examples/s]8747 examples [00:10, 874.28 examples/s]8838 examples [00:10, 883.56 examples/s]8927 examples [00:10, 885.12 examples/s]9016 examples [00:10, 870.57 examples/s]9105 examples [00:10, 874.74 examples/s]9193 examples [00:10, 866.34 examples/s]9280 examples [00:10, 867.30 examples/s]9371 examples [00:10, 878.12 examples/s]9459 examples [00:10, 842.54 examples/s]9544 examples [00:10, 818.70 examples/s]9637 examples [00:11, 846.45 examples/s]9723 examples [00:11, 847.62 examples/s]9813 examples [00:11, 861.29 examples/s]9900 examples [00:11, 830.16 examples/s]9984 examples [00:11, 822.89 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]100%|█████████▉| 9970/10000 [00:00<00:00, 99699.38 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteACND51/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteACND51/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc6b47648c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc6b47648c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc6b47648c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc720c07da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc69406b1d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc6b47648c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc6b47648c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc6b47648c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc69dd1b710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc695cd22b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc6889af730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc6889af730> 

  function with postional parmater data_info <function split_train_valid at 0x7fc6889af730> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc6889af840> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc6889af840> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc6889af840> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a5384b39f7de4f7ee343ab89ffbdb37c966eb676faedf999ba31ad506335d528
  Stored in directory: /tmp/pip-ephem-wheel-cache-jl8fqb8v/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:22:39, 9.44kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<17:59:53, 13.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:39:11, 18.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:51:52, 27.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.36M/862M [00:01<6:11:25, 38.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.62M/862M [00:01<4:19:08, 55.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.36M/862M [00:01<3:01:18, 78.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<2:06:20, 112kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:01<1:28:15, 160kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.3M/862M [00:01<1:01:40, 228kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.7M/862M [00:02<43:07, 324kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:02<30:10, 462kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:02<21:08, 656kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.6M/862M [00:02<14:50, 930kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<10:27, 1.31MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<07:24, 1.85MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<05:17, 2.58MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<03:49, 3.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:02<02:54, 4.67MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:02<02:16, 5.95MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<02:46, 4.85MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:03<02:05, 6.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<06:57, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<09:14, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<07:24, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<05:28, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<06:48, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:05, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<05:33, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:07, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:41, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:16, 2.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<05:54, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:31, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<05:09, 2.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<06:27, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<05:05, 2.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<05:46, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<06:19, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<05:00, 2.61MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<05:41, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<06:20, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<04:56, 2.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<03:36, 3.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<13:11, 982kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<11:34, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:19<08:36, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<16:02, 803kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<13:33, 950kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<09:58, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<07:07, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:23<14:01, 913kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<12:02, 1.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:23<08:59, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<08:25, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<08:12, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<06:18, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:32, 1.94MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:52, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:22, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:53, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:23, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<04:56, 2.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<03:37, 3.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:17, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<08:41, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<06:38, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:44, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:00, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:27, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:54, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:22, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:01, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<03:38, 3.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<23:44, 519kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<18:51, 654kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<13:44, 896kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:39, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<10:19, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:46, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:28, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:27, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:46, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:03, 2.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:49, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<07:17, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:21, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:53, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:54, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:17, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<03:49, 3.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<16:53, 709kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:59, 856kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:14, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:16, 1.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<20:14, 589kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<18:19, 650kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<13:54, 856kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<09:58, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:53, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:03, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:47, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:58, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:15, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:39, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:55, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:15, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:48, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:35, 3.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:20, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:07, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:38, 2.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<03:23, 3.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<14:29, 799kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<12:15, 944kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:06, 1.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:17, 1.39MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:52, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:56, 1.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:21, 2.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:21, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:08, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:30, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:45, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:05, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:45, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:42, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:30, 2.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:02, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:33, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:18, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:08, 3.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<11:47, 946kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:11, 1.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:33, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<05:24, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<13:53, 798kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<11:43, 945kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:39, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<06:10, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<15:19, 719kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:42, 867kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:22, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<08:23, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:51, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:58, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:00, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:06, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:45, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:26, 3.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<28:47, 376kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<22:06, 489kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<15:56, 678kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<12:55, 832kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<10:59, 978kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:09, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:29, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:10, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:29, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:37, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:40, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:19, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:39, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:00, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:06, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:44, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:28, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:17, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:45, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:08, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:03, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:34, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:04, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:00, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:31, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:03, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<03:59, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:30, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:00, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<03:57, 2.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:28, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:53, 2.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:52, 2.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:23, 2.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:53, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<03:47, 2.65MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<02:47, 3.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:09, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:48, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:12, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:18, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:17, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:39, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:58, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:55, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:23, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:50, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:49, 2.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:18, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:26, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:22, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<03:58, 2.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:17, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:23, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:12, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:03, 3.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<27:03, 354kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<21:51, 438kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<16:00, 597kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<11:20, 840kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:13, 847kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<10:45, 883kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:07, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:49, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:49, 1.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:39, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:55, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:23, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:02, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:13, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:01, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:39, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:51, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:46, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:16, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:51, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:05, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:12, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:52, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:36, 2.54MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:52, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:38, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:28, 2.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:22, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:31, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:28, 2.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:16, 2.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:58, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<06:37, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:08, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:49, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:37, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:39, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:27, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:15, 2.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:49, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:39, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:17, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<06:03, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:30, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<05:02, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:37, 2.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:07, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:39, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:43, 2.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:18, 1.63MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:57, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:39, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:24, 2.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:54, 1.46MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:21, 1.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:00, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:38, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:27, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:58, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:38, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<04:42, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:28, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:17, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:08, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<05:27, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:16, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:07, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:50, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:31, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:19, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:07, 2.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:53, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:34, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:19, 2.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:10, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:59, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:46, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<05:26, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<05:54, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:40, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:23, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<05:37, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<06:06, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:49, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:30, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<05:30, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<05:54, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:38, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:21, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:38, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<05:58, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:37, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:34, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<06:01, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<04:42, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:25, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:36, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:55, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:37, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:21, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:41, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:51, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:30, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:15, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<04:48, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<05:19, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:11, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:03, 2.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:23, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:38, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:20, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:08, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:49, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:19, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:11, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:01, 2.46MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:28, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:38, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:24, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:12, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:24, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:35, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:22, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:09, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:43, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<05:53, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:29, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:17, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<04:06, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<04:38, 1.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:37, 1.99MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<02:43, 2.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<04:08, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:16, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:23, 2.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<04:29, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:52, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:45, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:45, 2.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:54, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<04:26, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:25, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:33, 2.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:24, 2.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:04, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:17, 3.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:25, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:10, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:18, 2.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:05, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<05:08, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:55, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:49, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:58, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:59, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:51, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:47, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:37, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<05:44, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<04:24, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:10, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:21, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<04:27, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:24, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:27, 2.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:48, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<04:41, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:33, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:35, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:03, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:24, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:25, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:28, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:54, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:18, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:24, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:27, 2.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:10, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:09, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:59, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:51, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<05:18, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:11, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<04:00, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:52, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:34, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:01, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:47, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:39, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:38, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:45, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:00, 3.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:59, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:53, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:41, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:48, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:55, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:03, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:11, 2.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<10:03, 587kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:20, 707kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<06:06, 964kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:23, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:30, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:23, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:25, 2.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:46, 1.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:32, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:41, 2.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:53, 1.97MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:12, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:32, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<01:49, 3.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<30:52, 182kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<22:44, 247kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<16:08, 348kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<11:17, 493kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<10:53, 510kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<08:23, 661kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<06:03, 914kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:13, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:43, 958kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:27, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:12, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<02:19, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<12:53, 420kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<10:02, 539kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<07:16, 743kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:55, 903kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:09, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:51, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:32, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:30, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:42, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:44, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<02:55, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:16, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:38, 3.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:47, 888kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:00, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:44, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:26, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:23, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:38, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:54, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:12, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:21, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:48, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:56, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:14, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:39, 2.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:46, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:53, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:12, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<01:35, 3.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<04:18, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:54, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:57, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:50, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:53, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:14, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:19, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:32, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:59, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:26, 3.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<05:00, 916kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:22, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:14, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:20, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:01, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:57, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:16, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:19, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:27, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:55, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:17, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:48, 2.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:58, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:11, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:42, 2.53MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:13, 3.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:10, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<03:43, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:46, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:58, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:59, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:36, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:40, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:09, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:00, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:17, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:15, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:20, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:47, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:18, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:43, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:39, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:01, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:26, 2.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<06:30, 600kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<05:18, 735kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:51, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:43, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:07, 933kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:36, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:41, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:29, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:26, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:51, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:20, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:13, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:45, 985kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:59, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:21, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:19, 1.56MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:46, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:16, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:33, 1.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:11, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:23, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:13, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:14, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:42, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:14, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:31, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:24, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:49, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:54, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:29, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:44, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:22, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:28, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:38, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:16, 2.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<00:55, 3.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:05, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:38, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:09, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:41, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:12, 2.55MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:38, 547kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:27, 565kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:08, 743kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:58, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:07, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:09, 953kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:41, 818kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:57, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:07, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:32, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<03:07, 944kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:37, 813kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:52, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:05, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:30, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<03:21, 859kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<03:37, 794kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:51, 1.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:04, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:29, 1.88MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<03:28, 808kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<03:40, 763kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:52, 972kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:04, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:29, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:41, 742kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:48, 720kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:57, 923kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:08, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:31, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:43, 716kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:47, 703kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:56, 904kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:06, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:32, 734kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:32, 735kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:44, 945kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:58, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:24, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<04:12, 602kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<04:03, 622kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<03:06, 812kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:13, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:34, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<04:06, 600kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:57, 621kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:01, 811kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:10, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:32, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<04:13, 566kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<04:01, 595kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:03, 780kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:11, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:33, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<04:24, 527kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<04:06, 564kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:07, 743kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:13, 1.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:35, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:04, 553kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:46, 598kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:49, 793kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:01, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:26, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:56, 744kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:56, 742kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:17, 953kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:38, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:09, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:48, 752kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<02:49, 748kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:09, 976kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:33, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:49, 726kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<02:52, 711kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:13, 918kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:35, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:07, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:20, 846kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:33<02:30, 787kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:57, 1.00MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:24, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:00, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<03:11, 599kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<03:04, 620kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:20, 810kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:40, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:11, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:19, 554kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:08, 585kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:23, 768kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:42, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:12, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:15, 543kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:00, 590kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:16, 773kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:37, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:08, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:33, 479kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:10, 535kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<02:23, 709kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:41, 988kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:11, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<03:50, 424kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<03:21, 487kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<02:28, 656kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:45, 912kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:13, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:47, 270kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<04:41, 333kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<03:25, 454kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:24, 637kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:40, 894kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<12:25, 120kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<09:15, 161kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<06:35, 226kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<04:35, 319kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<03:09, 453kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<09:08, 156kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<06:55, 206kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<04:56, 287kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<03:26, 406kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<02:21, 575kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<16:34, 81.9kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<12:05, 112kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<08:31, 158kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<05:54, 224kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<04:03, 319kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<04:21, 295kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<03:29, 368kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:32, 505kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:46, 707kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:28, 822kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:27, 833kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:07, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:47, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:48, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:57, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:45, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:32, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:23, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<05:13, 207kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<04:00, 270kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<02:51, 373kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<01:58, 528kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:36, 629kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:27, 696kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:05, 922kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:45, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:46, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:49, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:38, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:33, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:40, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:22, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:16, 2.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:56, 856kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:52, 919kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:39, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:27, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:32, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:35, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:27, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<03:02, 219kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<02:17, 290kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:36, 403kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<01:04, 571kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:57, 619kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:49, 713kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:36, 950kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:24, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:27, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:25, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:19, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:12, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:21, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:22, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:17, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:16, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:05, 2.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:20, 723kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:17, 865kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:11, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:06, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:13, 787kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:11, 928kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:04, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:08, 818kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 906kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.67MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.46MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 1.90MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<151:49:45,  1.37s/it]  0%|          | 755/400000 [00:01<106:05:04,  1.05it/s]  0%|          | 1479/400000 [00:01<74:07:45,  1.49it/s]  1%|          | 2135/400000 [00:01<51:48:36,  2.13it/s]  1%|          | 2798/400000 [00:01<36:12:41,  3.05it/s]  1%|          | 3529/400000 [00:01<25:18:21,  4.35it/s]  1%|          | 4271/400000 [00:01<17:41:07,  6.22it/s]  1%|▏         | 5028/400000 [00:02<12:21:37,  8.88it/s]  1%|▏         | 5787/400000 [00:02<8:38:24, 12.67it/s]   2%|▏         | 6542/400000 [00:02<6:02:26, 18.09it/s]  2%|▏         | 7268/400000 [00:02<4:13:30, 25.82it/s]  2%|▏         | 8016/400000 [00:02<2:57:23, 36.83it/s]  2%|▏         | 8777/400000 [00:02<2:04:11, 52.50it/s]  2%|▏         | 9540/400000 [00:02<1:27:00, 74.79it/s]  3%|▎         | 10294/400000 [00:02<1:01:03, 106.39it/s]  3%|▎         | 11046/400000 [00:02<42:54, 151.06it/s]    3%|▎         | 11795/400000 [00:02<30:14, 213.89it/s]  3%|▎         | 12549/400000 [00:03<21:23, 301.88it/s]  3%|▎         | 13304/400000 [00:03<15:12, 423.99it/s]  4%|▎         | 14069/400000 [00:03<10:52, 591.65it/s]  4%|▎         | 14823/400000 [00:03<07:51, 817.68it/s]  4%|▍         | 15576/400000 [00:03<05:44, 1114.99it/s]  4%|▍         | 16324/400000 [00:03<04:16, 1496.13it/s]  4%|▍         | 17071/400000 [00:03<03:14, 1968.24it/s]  4%|▍         | 17817/400000 [00:03<02:31, 2525.77it/s]  5%|▍         | 18573/400000 [00:03<02:00, 3155.82it/s]  5%|▍         | 19322/400000 [00:03<01:39, 3815.70it/s]  5%|▌         | 20070/400000 [00:04<01:25, 4468.02it/s]  5%|▌         | 20832/400000 [00:04<01:14, 5099.89it/s]  5%|▌         | 21588/400000 [00:04<01:06, 5650.59it/s]  6%|▌         | 22341/400000 [00:04<01:01, 6096.35it/s]  6%|▌         | 23092/400000 [00:04<00:58, 6457.96it/s]  6%|▌         | 23852/400000 [00:04<00:55, 6761.80it/s]  6%|▌         | 24606/400000 [00:04<00:54, 6885.35it/s]  6%|▋         | 25351/400000 [00:04<00:53, 7044.53it/s]  7%|▋         | 26116/400000 [00:04<00:51, 7213.95it/s]  7%|▋         | 26873/400000 [00:04<00:50, 7316.57it/s]  7%|▋         | 27625/400000 [00:05<00:51, 7263.67it/s]  7%|▋         | 28380/400000 [00:05<00:50, 7345.60it/s]  7%|▋         | 29125/400000 [00:05<00:50, 7361.30it/s]  7%|▋         | 29886/400000 [00:05<00:49, 7431.91it/s]  8%|▊         | 30635/400000 [00:05<00:50, 7355.22it/s]  8%|▊         | 31375/400000 [00:05<00:51, 7199.85it/s]  8%|▊         | 32126/400000 [00:05<00:50, 7289.27it/s]  8%|▊         | 32891/400000 [00:05<00:49, 7391.95it/s]  8%|▊         | 33633/400000 [00:05<00:49, 7380.30it/s]  9%|▊         | 34373/400000 [00:06<00:49, 7338.87it/s]  9%|▉         | 35141/400000 [00:06<00:49, 7436.08it/s]  9%|▉         | 35907/400000 [00:06<00:48, 7499.51it/s]  9%|▉         | 36658/400000 [00:06<00:48, 7496.64it/s]  9%|▉         | 37409/400000 [00:06<00:48, 7471.02it/s] 10%|▉         | 38157/400000 [00:06<00:48, 7450.46it/s] 10%|▉         | 38916/400000 [00:06<00:48, 7489.51it/s] 10%|▉         | 39666/400000 [00:06<00:48, 7406.25it/s] 10%|█         | 40417/400000 [00:06<00:48, 7435.06it/s] 10%|█         | 41161/400000 [00:06<00:48, 7413.93it/s] 10%|█         | 41903/400000 [00:07<00:49, 7203.08it/s] 11%|█         | 42625/400000 [00:07<00:50, 7135.95it/s] 11%|█         | 43340/400000 [00:07<00:50, 7093.36it/s] 11%|█         | 44051/400000 [00:07<00:50, 7063.91it/s] 11%|█         | 44759/400000 [00:07<00:50, 7033.94it/s] 11%|█▏        | 45463/400000 [00:07<00:50, 7020.76it/s] 12%|█▏        | 46166/400000 [00:07<00:51, 6863.94it/s] 12%|█▏        | 46872/400000 [00:07<00:51, 6919.08it/s] 12%|█▏        | 47567/400000 [00:07<00:50, 6927.77it/s] 12%|█▏        | 48270/400000 [00:07<00:50, 6957.51it/s] 12%|█▏        | 48967/400000 [00:08<00:50, 6958.87it/s] 12%|█▏        | 49668/400000 [00:08<00:50, 6973.77it/s] 13%|█▎        | 50366/400000 [00:08<00:50, 6911.62it/s] 13%|█▎        | 51059/400000 [00:08<00:50, 6914.43it/s] 13%|█▎        | 51751/400000 [00:08<00:50, 6914.60it/s] 13%|█▎        | 52464/400000 [00:08<00:49, 6976.87it/s] 13%|█▎        | 53205/400000 [00:08<00:48, 7100.19it/s] 13%|█▎        | 53945/400000 [00:08<00:48, 7186.21it/s] 14%|█▎        | 54686/400000 [00:08<00:47, 7250.26it/s] 14%|█▍        | 55437/400000 [00:08<00:47, 7324.74it/s] 14%|█▍        | 56182/400000 [00:09<00:46, 7361.08it/s] 14%|█▍        | 56946/400000 [00:09<00:46, 7437.19it/s] 14%|█▍        | 57708/400000 [00:09<00:45, 7489.38it/s] 15%|█▍        | 58465/400000 [00:09<00:45, 7513.24it/s] 15%|█▍        | 59217/400000 [00:09<00:45, 7452.26it/s] 15%|█▍        | 59963/400000 [00:09<00:45, 7424.08it/s] 15%|█▌        | 60716/400000 [00:09<00:45, 7454.59it/s] 15%|█▌        | 61474/400000 [00:09<00:45, 7489.51it/s] 16%|█▌        | 62231/400000 [00:09<00:44, 7512.12it/s] 16%|█▌        | 62993/400000 [00:09<00:44, 7542.12it/s] 16%|█▌        | 63748/400000 [00:10<00:45, 7427.35it/s] 16%|█▌        | 64504/400000 [00:10<00:44, 7464.45it/s] 16%|█▋        | 65258/400000 [00:10<00:44, 7484.54it/s] 17%|█▋        | 66017/400000 [00:10<00:44, 7515.79it/s] 17%|█▋        | 66778/400000 [00:10<00:44, 7541.96it/s] 17%|█▋        | 67533/400000 [00:10<00:44, 7490.23it/s] 17%|█▋        | 68283/400000 [00:10<00:44, 7417.04it/s] 17%|█▋        | 69049/400000 [00:10<00:44, 7487.01it/s] 17%|█▋        | 69811/400000 [00:10<00:43, 7524.54it/s] 18%|█▊        | 70570/400000 [00:10<00:43, 7542.48it/s] 18%|█▊        | 71325/400000 [00:11<00:43, 7529.30it/s] 18%|█▊        | 72079/400000 [00:11<00:43, 7528.33it/s] 18%|█▊        | 72841/400000 [00:11<00:43, 7553.64it/s] 18%|█▊        | 73605/400000 [00:11<00:43, 7577.08it/s] 19%|█▊        | 74369/400000 [00:11<00:42, 7594.94it/s] 19%|█▉        | 75129/400000 [00:11<00:42, 7580.03it/s] 19%|█▉        | 75888/400000 [00:11<00:42, 7547.78it/s] 19%|█▉        | 76643/400000 [00:11<00:42, 7522.71it/s] 19%|█▉        | 77396/400000 [00:11<00:42, 7511.03it/s] 20%|█▉        | 78148/400000 [00:11<00:43, 7437.98it/s] 20%|█▉        | 78892/400000 [00:12<00:44, 7222.69it/s] 20%|█▉        | 79641/400000 [00:12<00:43, 7300.16it/s] 20%|██        | 80393/400000 [00:12<00:43, 7363.83it/s] 20%|██        | 81131/400000 [00:12<00:43, 7350.57it/s] 20%|██        | 81868/400000 [00:12<00:43, 7354.93it/s] 21%|██        | 82604/400000 [00:12<00:43, 7290.70it/s] 21%|██        | 83358/400000 [00:12<00:43, 7361.11it/s] 21%|██        | 84111/400000 [00:12<00:42, 7409.35it/s] 21%|██        | 84853/400000 [00:12<00:42, 7349.83it/s] 21%|██▏       | 85589/400000 [00:12<00:42, 7351.68it/s] 22%|██▏       | 86325/400000 [00:13<00:42, 7305.98it/s] 22%|██▏       | 87080/400000 [00:13<00:42, 7374.63it/s] 22%|██▏       | 87821/400000 [00:13<00:42, 7384.03it/s] 22%|██▏       | 88568/400000 [00:13<00:42, 7408.53it/s] 22%|██▏       | 89320/400000 [00:13<00:41, 7439.59it/s] 23%|██▎       | 90065/400000 [00:13<00:42, 7285.33it/s] 23%|██▎       | 90795/400000 [00:13<00:42, 7245.81it/s] 23%|██▎       | 91548/400000 [00:13<00:42, 7327.81it/s] 23%|██▎       | 92282/400000 [00:13<00:42, 7271.33it/s] 23%|██▎       | 93016/400000 [00:13<00:42, 7290.55it/s] 23%|██▎       | 93746/400000 [00:14<00:42, 7194.94it/s] 24%|██▎       | 94467/400000 [00:14<00:44, 6895.20it/s] 24%|██▍       | 95160/400000 [00:14<00:44, 6856.19it/s] 24%|██▍       | 95896/400000 [00:14<00:43, 6997.55it/s] 24%|██▍       | 96655/400000 [00:14<00:42, 7162.84it/s] 24%|██▍       | 97375/400000 [00:14<00:42, 7172.61it/s] 25%|██▍       | 98117/400000 [00:14<00:41, 7243.36it/s] 25%|██▍       | 98876/400000 [00:14<00:41, 7343.19it/s] 25%|██▍       | 99639/400000 [00:14<00:40, 7426.85it/s] 25%|██▌       | 100395/400000 [00:15<00:40, 7463.72it/s] 25%|██▌       | 101143/400000 [00:15<00:40, 7302.79it/s] 25%|██▌       | 101893/400000 [00:15<00:40, 7358.66it/s] 26%|██▌       | 102630/400000 [00:15<00:41, 7184.39it/s] 26%|██▌       | 103351/400000 [00:15<00:41, 7175.45it/s] 26%|██▌       | 104070/400000 [00:15<00:41, 7178.42it/s] 26%|██▌       | 104817/400000 [00:15<00:40, 7262.40it/s] 26%|██▋       | 105583/400000 [00:15<00:39, 7377.23it/s] 27%|██▋       | 106341/400000 [00:15<00:39, 7435.02it/s] 27%|██▋       | 107095/400000 [00:15<00:39, 7464.54it/s] 27%|██▋       | 107933/400000 [00:16<00:37, 7716.76it/s] 27%|██▋       | 108708/400000 [00:16<00:37, 7711.88it/s] 27%|██▋       | 109481/400000 [00:16<00:37, 7698.17it/s] 28%|██▊       | 110253/400000 [00:16<00:37, 7676.67it/s] 28%|██▊       | 111022/400000 [00:16<00:37, 7620.67it/s] 28%|██▊       | 111785/400000 [00:16<00:37, 7620.45it/s] 28%|██▊       | 112548/400000 [00:16<00:37, 7589.86it/s] 28%|██▊       | 113308/400000 [00:16<00:37, 7586.50it/s] 29%|██▊       | 114076/400000 [00:16<00:37, 7612.71it/s] 29%|██▊       | 114838/400000 [00:16<00:37, 7599.00it/s] 29%|██▉       | 115599/400000 [00:17<00:38, 7398.85it/s] 29%|██▉       | 116349/400000 [00:17<00:38, 7426.06it/s] 29%|██▉       | 117115/400000 [00:17<00:37, 7494.42it/s] 29%|██▉       | 117877/400000 [00:17<00:37, 7529.45it/s] 30%|██▉       | 118631/400000 [00:17<00:37, 7520.06it/s] 30%|██▉       | 119392/400000 [00:17<00:37, 7546.17it/s] 30%|███       | 120147/400000 [00:17<00:37, 7483.61it/s] 30%|███       | 120902/400000 [00:17<00:37, 7501.38it/s] 30%|███       | 121664/400000 [00:17<00:36, 7534.31it/s] 31%|███       | 122427/400000 [00:17<00:36, 7560.17it/s] 31%|███       | 123195/400000 [00:18<00:36, 7595.40it/s] 31%|███       | 123955/400000 [00:18<00:36, 7466.31it/s] 31%|███       | 124705/400000 [00:18<00:36, 7475.31it/s] 31%|███▏      | 125461/400000 [00:18<00:36, 7499.51it/s] 32%|███▏      | 126219/400000 [00:18<00:36, 7522.00it/s] 32%|███▏      | 126973/400000 [00:18<00:36, 7524.67it/s] 32%|███▏      | 127726/400000 [00:18<00:37, 7181.28it/s] 32%|███▏      | 128465/400000 [00:18<00:37, 7241.97it/s] 32%|███▏      | 129231/400000 [00:18<00:36, 7360.47it/s] 32%|███▏      | 129992/400000 [00:18<00:36, 7432.54it/s] 33%|███▎      | 130761/400000 [00:19<00:35, 7505.97it/s] 33%|███▎      | 131513/400000 [00:19<00:35, 7476.32it/s] 33%|███▎      | 132274/400000 [00:19<00:35, 7515.16it/s] 33%|███▎      | 133027/400000 [00:19<00:35, 7476.80it/s] 33%|███▎      | 133784/400000 [00:19<00:35, 7504.10it/s] 34%|███▎      | 134537/400000 [00:19<00:35, 7509.48it/s] 34%|███▍      | 135289/400000 [00:19<00:35, 7504.50it/s] 34%|███▍      | 136045/400000 [00:19<00:35, 7518.17it/s] 34%|███▍      | 136797/400000 [00:19<00:35, 7431.52it/s] 34%|███▍      | 137548/400000 [00:19<00:35, 7454.61it/s] 35%|███▍      | 138302/400000 [00:20<00:34, 7479.84it/s] 35%|███▍      | 139056/400000 [00:20<00:34, 7495.18it/s] 35%|███▍      | 139811/400000 [00:20<00:34, 7510.39it/s] 35%|███▌      | 140563/400000 [00:20<00:36, 7064.84it/s] 35%|███▌      | 141275/400000 [00:20<00:37, 6850.20it/s] 35%|███▌      | 141966/400000 [00:20<00:37, 6848.85it/s] 36%|███▌      | 142704/400000 [00:20<00:36, 6997.83it/s] 36%|███▌      | 143450/400000 [00:20<00:35, 7128.79it/s] 36%|███▌      | 144196/400000 [00:20<00:35, 7224.35it/s] 36%|███▌      | 144949/400000 [00:21<00:34, 7311.44it/s] 36%|███▋      | 145683/400000 [00:21<00:34, 7286.09it/s] 37%|███▋      | 146413/400000 [00:21<00:34, 7279.23it/s] 37%|███▋      | 147161/400000 [00:21<00:34, 7337.12it/s] 37%|███▋      | 147918/400000 [00:21<00:34, 7402.71it/s] 37%|███▋      | 148674/400000 [00:21<00:33, 7448.00it/s] 37%|███▋      | 149430/400000 [00:21<00:33, 7479.28it/s] 38%|███▊      | 150179/400000 [00:21<00:33, 7389.47it/s] 38%|███▊      | 150927/400000 [00:21<00:33, 7414.55it/s] 38%|███▊      | 151683/400000 [00:21<00:33, 7455.82it/s] 38%|███▊      | 152429/400000 [00:22<00:33, 7450.01it/s] 38%|███▊      | 153175/400000 [00:22<00:33, 7404.71it/s] 38%|███▊      | 153916/400000 [00:22<00:33, 7369.59it/s] 39%|███▊      | 154654/400000 [00:22<00:33, 7360.46it/s] 39%|███▉      | 155395/400000 [00:22<00:33, 7373.45it/s] 39%|███▉      | 156158/400000 [00:22<00:32, 7448.24it/s] 39%|███▉      | 156921/400000 [00:22<00:32, 7501.66it/s] 39%|███▉      | 157672/400000 [00:22<00:32, 7503.96it/s] 40%|███▉      | 158423/400000 [00:22<00:32, 7449.97it/s] 40%|███▉      | 159178/400000 [00:22<00:32, 7478.71it/s] 40%|███▉      | 159927/400000 [00:23<00:32, 7438.54it/s] 40%|████      | 160672/400000 [00:23<00:32, 7421.08it/s] 40%|████      | 161422/400000 [00:23<00:32, 7443.71it/s] 41%|████      | 162176/400000 [00:23<00:31, 7471.95it/s] 41%|████      | 162924/400000 [00:23<00:31, 7412.79it/s] 41%|████      | 163668/400000 [00:23<00:31, 7420.62it/s] 41%|████      | 164421/400000 [00:23<00:31, 7451.05it/s] 41%|████▏     | 165168/400000 [00:23<00:31, 7456.12it/s] 41%|████▏     | 165914/400000 [00:23<00:31, 7445.82it/s] 42%|████▏     | 166659/400000 [00:23<00:31, 7443.62it/s] 42%|████▏     | 167404/400000 [00:24<00:31, 7431.00it/s] 42%|████▏     | 168172/400000 [00:24<00:30, 7503.55it/s] 42%|████▏     | 168923/400000 [00:24<00:30, 7493.74it/s] 42%|████▏     | 169673/400000 [00:24<00:30, 7495.51it/s] 43%|████▎     | 170423/400000 [00:24<00:30, 7454.76it/s] 43%|████▎     | 171169/400000 [00:24<00:30, 7396.87it/s] 43%|████▎     | 171909/400000 [00:24<00:31, 7351.38it/s] 43%|████▎     | 172645/400000 [00:24<00:30, 7339.29it/s] 43%|████▎     | 173397/400000 [00:24<00:30, 7391.18it/s] 44%|████▎     | 174153/400000 [00:24<00:30, 7440.48it/s] 44%|████▎     | 174920/400000 [00:25<00:29, 7507.24it/s] 44%|████▍     | 175672/400000 [00:25<00:29, 7491.43it/s] 44%|████▍     | 176422/400000 [00:25<00:29, 7479.29it/s] 44%|████▍     | 177171/400000 [00:25<00:29, 7470.18it/s] 44%|████▍     | 177924/400000 [00:25<00:29, 7485.99it/s] 45%|████▍     | 178673/400000 [00:25<00:30, 7280.67it/s] 45%|████▍     | 179414/400000 [00:25<00:30, 7316.59it/s] 45%|████▌     | 180147/400000 [00:25<00:30, 7316.03it/s] 45%|████▌     | 180903/400000 [00:25<00:29, 7387.52it/s] 45%|████▌     | 181658/400000 [00:25<00:29, 7434.12it/s] 46%|████▌     | 182417/400000 [00:26<00:29, 7478.64it/s] 46%|████▌     | 183177/400000 [00:26<00:28, 7513.29it/s] 46%|████▌     | 183929/400000 [00:26<00:28, 7483.37it/s] 46%|████▌     | 184678/400000 [00:26<00:28, 7463.79it/s] 46%|████▋     | 185435/400000 [00:26<00:28, 7493.82it/s] 47%|████▋     | 186190/400000 [00:26<00:28, 7508.41it/s] 47%|████▋     | 186942/400000 [00:26<00:28, 7511.83it/s] 47%|████▋     | 187694/400000 [00:26<00:28, 7510.82it/s] 47%|████▋     | 188446/400000 [00:26<00:28, 7453.00it/s] 47%|████▋     | 189195/400000 [00:26<00:28, 7461.50it/s] 47%|████▋     | 189952/400000 [00:27<00:28, 7493.11it/s] 48%|████▊     | 190709/400000 [00:27<00:27, 7514.41it/s] 48%|████▊     | 191461/400000 [00:27<00:27, 7512.35it/s] 48%|████▊     | 192219/400000 [00:27<00:27, 7531.26it/s] 48%|████▊     | 192973/400000 [00:27<00:27, 7499.43it/s] 48%|████▊     | 193724/400000 [00:27<00:27, 7444.21it/s] 49%|████▊     | 194476/400000 [00:27<00:27, 7466.64it/s] 49%|████▉     | 195223/400000 [00:27<00:27, 7328.24it/s] 49%|████▉     | 195969/400000 [00:27<00:27, 7359.58it/s] 49%|████▉     | 196720/400000 [00:27<00:27, 7401.59it/s] 49%|████▉     | 197461/400000 [00:28<00:27, 7398.64it/s] 50%|████▉     | 198216/400000 [00:28<00:27, 7442.06it/s] 50%|████▉     | 198967/400000 [00:28<00:26, 7461.10it/s] 50%|████▉     | 199714/400000 [00:28<00:26, 7462.24it/s] 50%|█████     | 200468/400000 [00:28<00:26, 7483.98it/s] 50%|█████     | 201217/400000 [00:28<00:26, 7426.89it/s] 50%|█████     | 201969/400000 [00:28<00:26, 7452.30it/s] 51%|█████     | 202715/400000 [00:28<00:27, 7257.97it/s] 51%|█████     | 203463/400000 [00:28<00:26, 7322.05it/s] 51%|█████     | 204197/400000 [00:29<00:27, 7179.62it/s] 51%|█████     | 204953/400000 [00:29<00:26, 7288.56it/s] 51%|█████▏    | 205709/400000 [00:29<00:26, 7366.72it/s] 52%|█████▏    | 206465/400000 [00:29<00:26, 7422.92it/s] 52%|█████▏    | 207216/400000 [00:29<00:25, 7446.29it/s] 52%|█████▏    | 207978/400000 [00:29<00:25, 7494.59it/s] 52%|█████▏    | 208728/400000 [00:29<00:25, 7490.10it/s] 52%|█████▏    | 209478/400000 [00:29<00:25, 7450.24it/s] 53%|█████▎    | 210224/400000 [00:29<00:25, 7387.33it/s] 53%|█████▎    | 210974/400000 [00:29<00:25, 7420.51it/s] 53%|█████▎    | 211727/400000 [00:30<00:25, 7450.15it/s] 53%|█████▎    | 212473/400000 [00:30<00:25, 7427.75it/s] 53%|█████▎    | 213228/400000 [00:30<00:25, 7462.25it/s] 53%|█████▎    | 213975/400000 [00:30<00:25, 7388.13it/s] 54%|█████▎    | 214715/400000 [00:30<00:25, 7385.61it/s] 54%|█████▍    | 215464/400000 [00:30<00:24, 7415.17it/s] 54%|█████▍    | 216218/400000 [00:30<00:24, 7450.51it/s] 54%|█████▍    | 216970/400000 [00:30<00:24, 7470.35it/s] 54%|█████▍    | 217718/400000 [00:30<00:24, 7469.76it/s] 55%|█████▍    | 218466/400000 [00:30<00:24, 7419.19it/s] 55%|█████▍    | 219220/400000 [00:31<00:24, 7452.57it/s] 55%|█████▍    | 219969/400000 [00:31<00:24, 7462.34it/s] 55%|█████▌    | 220720/400000 [00:31<00:23, 7473.72it/s] 55%|█████▌    | 221468/400000 [00:31<00:23, 7466.59it/s] 56%|█████▌    | 222215/400000 [00:31<00:23, 7460.32it/s] 56%|█████▌    | 222962/400000 [00:31<00:23, 7420.73it/s] 56%|█████▌    | 223711/400000 [00:31<00:23, 7441.02it/s] 56%|█████▌    | 224463/400000 [00:31<00:23, 7463.23it/s] 56%|█████▋    | 225212/400000 [00:31<00:23, 7469.24it/s] 56%|█████▋    | 225959/400000 [00:31<00:23, 7445.55it/s] 57%|█████▋    | 226704/400000 [00:32<00:23, 7432.89it/s] 57%|█████▋    | 227448/400000 [00:32<00:23, 7416.64it/s] 57%|█████▋    | 228202/400000 [00:32<00:23, 7450.26it/s] 57%|█████▋    | 228948/400000 [00:32<00:23, 7433.50it/s] 57%|█████▋    | 229692/400000 [00:32<00:23, 7190.49it/s] 58%|█████▊    | 230413/400000 [00:32<00:23, 7124.12it/s] 58%|█████▊    | 231127/400000 [00:32<00:24, 7022.59it/s] 58%|█████▊    | 231831/400000 [00:32<00:24, 6989.73it/s] 58%|█████▊    | 232531/400000 [00:32<00:23, 6983.18it/s] 58%|█████▊    | 233231/400000 [00:32<00:23, 6985.97it/s] 58%|█████▊    | 233931/400000 [00:33<00:23, 6976.80it/s] 59%|█████▊    | 234633/400000 [00:33<00:23, 6986.86it/s] 59%|█████▉    | 235332/400000 [00:33<00:23, 6948.42it/s] 59%|█████▉    | 236028/400000 [00:33<00:23, 6948.24it/s] 59%|█████▉    | 236723/400000 [00:33<00:23, 6942.73it/s] 59%|█████▉    | 237423/400000 [00:33<00:23, 6958.12it/s] 60%|█████▉    | 238119/400000 [00:33<00:23, 6952.75it/s] 60%|█████▉    | 238819/400000 [00:33<00:23, 6966.65it/s] 60%|█████▉    | 239521/400000 [00:33<00:22, 6980.32it/s] 60%|██████    | 240220/400000 [00:33<00:23, 6756.04it/s] 60%|██████    | 240898/400000 [00:34<00:24, 6588.59it/s] 60%|██████    | 241660/400000 [00:34<00:23, 6867.22it/s] 61%|██████    | 242425/400000 [00:34<00:22, 7083.69it/s] 61%|██████    | 243182/400000 [00:34<00:21, 7220.54it/s] 61%|██████    | 243914/400000 [00:34<00:21, 7247.78it/s] 61%|██████    | 244679/400000 [00:34<00:21, 7362.21it/s] 61%|██████▏   | 245427/400000 [00:34<00:20, 7395.68it/s] 62%|██████▏   | 246187/400000 [00:34<00:20, 7454.33it/s] 62%|██████▏   | 246935/400000 [00:34<00:20, 7460.38it/s] 62%|██████▏   | 247690/400000 [00:34<00:20, 7484.91it/s] 62%|██████▏   | 248440/400000 [00:35<00:20, 7465.42it/s] 62%|██████▏   | 249206/400000 [00:35<00:20, 7522.23it/s] 62%|██████▏   | 249967/400000 [00:35<00:19, 7547.68it/s] 63%|██████▎   | 250730/400000 [00:35<00:19, 7572.17it/s] 63%|██████▎   | 251488/400000 [00:35<00:19, 7549.57it/s] 63%|██████▎   | 252255/400000 [00:35<00:19, 7583.16it/s] 63%|██████▎   | 253014/400000 [00:35<00:19, 7542.55it/s] 63%|██████▎   | 253778/400000 [00:35<00:19, 7569.70it/s] 64%|██████▎   | 254536/400000 [00:35<00:19, 7561.61it/s] 64%|██████▍   | 255293/400000 [00:35<00:19, 7512.17it/s] 64%|██████▍   | 256047/400000 [00:36<00:19, 7517.99it/s] 64%|██████▍   | 256799/400000 [00:36<00:19, 7445.47it/s] 64%|██████▍   | 257562/400000 [00:36<00:18, 7497.89it/s] 65%|██████▍   | 258325/400000 [00:36<00:18, 7534.10it/s] 65%|██████▍   | 259079/400000 [00:36<00:18, 7490.25it/s] 65%|██████▍   | 259846/400000 [00:36<00:18, 7538.73it/s] 65%|██████▌   | 260601/400000 [00:36<00:18, 7514.77it/s] 65%|██████▌   | 261353/400000 [00:36<00:18, 7447.33it/s] 66%|██████▌   | 262106/400000 [00:36<00:18, 7470.04it/s] 66%|██████▌   | 262854/400000 [00:36<00:18, 7457.39it/s] 66%|██████▌   | 263608/400000 [00:37<00:18, 7479.41it/s] 66%|██████▌   | 264362/400000 [00:37<00:18, 7497.40it/s] 66%|██████▋   | 265112/400000 [00:37<00:18, 7479.66it/s] 66%|██████▋   | 265861/400000 [00:37<00:17, 7455.59it/s] 67%|██████▋   | 266607/400000 [00:37<00:18, 7313.30it/s] 67%|██████▋   | 267339/400000 [00:37<00:18, 7177.59it/s] 67%|██████▋   | 268058/400000 [00:37<00:18, 7086.49it/s] 67%|██████▋   | 268768/400000 [00:37<00:18, 7057.06it/s] 67%|██████▋   | 269475/400000 [00:37<00:18, 7042.57it/s] 68%|██████▊   | 270180/400000 [00:38<00:18, 6954.69it/s] 68%|██████▊   | 270877/400000 [00:38<00:18, 6953.12it/s] 68%|██████▊   | 271576/400000 [00:38<00:18, 6961.46it/s] 68%|██████▊   | 272273/400000 [00:38<00:18, 6958.48it/s] 68%|██████▊   | 272973/400000 [00:38<00:18, 6968.74it/s] 68%|██████▊   | 273671/400000 [00:38<00:18, 6917.36it/s] 69%|██████▊   | 274363/400000 [00:38<00:18, 6832.58it/s] 69%|██████▉   | 275110/400000 [00:38<00:17, 7009.73it/s] 69%|██████▉   | 275862/400000 [00:38<00:17, 7153.68it/s] 69%|██████▉   | 276604/400000 [00:38<00:17, 7230.27it/s] 69%|██████▉   | 277352/400000 [00:39<00:16, 7301.18it/s] 70%|██████▉   | 278084/400000 [00:39<00:16, 7304.24it/s] 70%|██████▉   | 278816/400000 [00:39<00:16, 7282.61it/s] 70%|██████▉   | 279570/400000 [00:39<00:16, 7356.13it/s] 70%|███████   | 280325/400000 [00:39<00:16, 7412.88it/s] 70%|███████   | 281074/400000 [00:39<00:15, 7435.68it/s] 70%|███████   | 281831/400000 [00:39<00:15, 7474.70it/s] 71%|███████   | 282579/400000 [00:39<00:15, 7446.19it/s] 71%|███████   | 283334/400000 [00:39<00:15, 7474.70it/s] 71%|███████   | 284091/400000 [00:39<00:15, 7502.27it/s] 71%|███████   | 284842/400000 [00:40<00:15, 7492.03it/s] 71%|███████▏  | 285593/400000 [00:40<00:15, 7495.22it/s] 72%|███████▏  | 286343/400000 [00:40<00:15, 7461.71it/s] 72%|███████▏  | 287090/400000 [00:40<00:15, 7408.32it/s] 72%|███████▏  | 287842/400000 [00:40<00:15, 7439.73it/s] 72%|███████▏  | 288587/400000 [00:40<00:14, 7436.18it/s] 72%|███████▏  | 289343/400000 [00:40<00:14, 7471.82it/s] 73%|███████▎  | 290103/400000 [00:40<00:14, 7509.45it/s] 73%|███████▎  | 290855/400000 [00:40<00:14, 7493.99it/s] 73%|███████▎  | 291605/400000 [00:40<00:14, 7479.42it/s] 73%|███████▎  | 292354/400000 [00:41<00:14, 7456.18it/s] 73%|███████▎  | 293122/400000 [00:41<00:14, 7519.20it/s] 73%|███████▎  | 293889/400000 [00:41<00:14, 7561.30it/s] 74%|███████▎  | 294657/400000 [00:41<00:13, 7596.06it/s] 74%|███████▍  | 295417/400000 [00:41<00:13, 7554.86it/s] 74%|███████▍  | 296173/400000 [00:41<00:13, 7511.07it/s] 74%|███████▍  | 296925/400000 [00:41<00:13, 7449.26it/s] 74%|███████▍  | 297671/400000 [00:41<00:14, 7250.45it/s] 75%|███████▍  | 298398/400000 [00:41<00:14, 7170.00it/s] 75%|███████▍  | 299117/400000 [00:41<00:14, 7096.74it/s] 75%|███████▍  | 299828/400000 [00:42<00:14, 6975.89it/s] 75%|███████▌  | 300527/400000 [00:42<00:14, 6973.00it/s] 75%|███████▌  | 301226/400000 [00:42<00:14, 6972.99it/s] 75%|███████▌  | 301926/400000 [00:42<00:14, 6980.02it/s] 76%|███████▌  | 302625/400000 [00:42<00:14, 6822.81it/s] 76%|███████▌  | 303309/400000 [00:42<00:14, 6810.09it/s] 76%|███████▌  | 304035/400000 [00:42<00:13, 6938.11it/s] 76%|███████▌  | 304795/400000 [00:42<00:13, 7122.00it/s] 76%|███████▋  | 305538/400000 [00:42<00:13, 7210.98it/s] 77%|███████▋  | 306291/400000 [00:42<00:12, 7303.08it/s] 77%|███████▋  | 307051/400000 [00:43<00:12, 7389.30it/s] 77%|███████▋  | 307811/400000 [00:43<00:12, 7451.04it/s] 77%|███████▋  | 308558/400000 [00:43<00:12, 7404.35it/s] 77%|███████▋  | 309314/400000 [00:43<00:12, 7449.73it/s] 78%|███████▊  | 310060/400000 [00:43<00:12, 7426.47it/s] 78%|███████▊  | 310804/400000 [00:43<00:12, 7410.84it/s] 78%|███████▊  | 311558/400000 [00:43<00:11, 7447.34it/s] 78%|███████▊  | 312303/400000 [00:43<00:11, 7424.83it/s] 78%|███████▊  | 313046/400000 [00:43<00:11, 7425.74it/s] 78%|███████▊  | 313802/400000 [00:43<00:11, 7464.23it/s] 79%|███████▊  | 314549/400000 [00:44<00:11, 7447.74it/s] 79%|███████▉  | 315306/400000 [00:44<00:11, 7483.31it/s] 79%|███████▉  | 316062/400000 [00:44<00:11, 7503.49it/s] 79%|███████▉  | 316813/400000 [00:44<00:11, 7428.43it/s] 79%|███████▉  | 317568/400000 [00:44<00:11, 7463.23it/s] 80%|███████▉  | 318315/400000 [00:44<00:11, 7419.17it/s] 80%|███████▉  | 319071/400000 [00:44<00:10, 7459.57it/s] 80%|███████▉  | 319825/400000 [00:44<00:10, 7482.02it/s] 80%|████████  | 320578/400000 [00:44<00:10, 7495.57it/s] 80%|████████  | 321328/400000 [00:44<00:10, 7425.70it/s] 81%|████████  | 322072/400000 [00:45<00:10, 7429.77it/s] 81%|████████  | 322823/400000 [00:45<00:10, 7452.74it/s] 81%|████████  | 323572/400000 [00:45<00:10, 7462.82it/s] 81%|████████  | 324322/400000 [00:45<00:10, 7472.17it/s] 81%|████████▏ | 325070/400000 [00:45<00:10, 7446.21it/s] 81%|████████▏ | 325815/400000 [00:45<00:09, 7434.09it/s] 82%|████████▏ | 326563/400000 [00:45<00:09, 7446.93it/s] 82%|████████▏ | 327321/400000 [00:45<00:09, 7484.72it/s] 82%|████████▏ | 328070/400000 [00:45<00:09, 7466.54it/s] 82%|████████▏ | 328824/400000 [00:45<00:09, 7486.33it/s] 82%|████████▏ | 329573/400000 [00:46<00:09, 7426.29it/s] 83%|████████▎ | 330343/400000 [00:46<00:09, 7505.18it/s] 83%|████████▎ | 331114/400000 [00:46<00:09, 7563.56it/s] 83%|████████▎ | 331885/400000 [00:46<00:08, 7604.44it/s] 83%|████████▎ | 332655/400000 [00:46<00:08, 7631.47it/s] 83%|████████▎ | 333419/400000 [00:46<00:08, 7595.74it/s] 84%|████████▎ | 334179/400000 [00:46<00:08, 7524.59it/s] 84%|████████▎ | 334936/400000 [00:46<00:08, 7536.56it/s] 84%|████████▍ | 335704/400000 [00:46<00:08, 7578.31it/s] 84%|████████▍ | 336471/400000 [00:46<00:08, 7603.52it/s] 84%|████████▍ | 337232/400000 [00:47<00:08, 7586.72it/s] 84%|████████▍ | 337991/400000 [00:47<00:08, 7483.62it/s] 85%|████████▍ | 338759/400000 [00:47<00:08, 7540.38it/s] 85%|████████▍ | 339515/400000 [00:47<00:08, 7542.35it/s] 85%|████████▌ | 340282/400000 [00:47<00:07, 7580.06it/s] 85%|████████▌ | 341041/400000 [00:47<00:07, 7519.10it/s] 85%|████████▌ | 341794/400000 [00:47<00:07, 7519.92it/s] 86%|████████▌ | 342547/400000 [00:47<00:07, 7396.61it/s] 86%|████████▌ | 343312/400000 [00:47<00:07, 7470.54it/s] 86%|████████▌ | 344071/400000 [00:48<00:07, 7504.33it/s] 86%|████████▌ | 344822/400000 [00:48<00:07, 7485.47it/s] 86%|████████▋ | 345571/400000 [00:48<00:07, 7482.78it/s] 87%|████████▋ | 346320/400000 [00:48<00:07, 7433.94it/s] 87%|████████▋ | 347075/400000 [00:48<00:07, 7467.64it/s] 87%|████████▋ | 347833/400000 [00:48<00:06, 7500.29it/s] 87%|████████▋ | 348586/400000 [00:48<00:06, 7507.24it/s] 87%|████████▋ | 349337/400000 [00:48<00:06, 7480.01it/s] 88%|████████▊ | 350087/400000 [00:48<00:06, 7484.17it/s] 88%|████████▊ | 350836/400000 [00:48<00:06, 7423.92it/s] 88%|████████▊ | 351592/400000 [00:49<00:06, 7462.53it/s] 88%|████████▊ | 352339/400000 [00:49<00:06, 7358.42it/s] 88%|████████▊ | 353076/400000 [00:49<00:06, 7339.64it/s] 88%|████████▊ | 353831/400000 [00:49<00:06, 7399.55it/s] 89%|████████▊ | 354588/400000 [00:49<00:06, 7448.96it/s] 89%|████████▉ | 355334/400000 [00:49<00:06, 7403.12it/s] 89%|████████▉ | 356086/400000 [00:49<00:05, 7432.95it/s] 89%|████████▉ | 356830/400000 [00:49<00:05, 7412.88it/s] 89%|████████▉ | 357572/400000 [00:49<00:05, 7376.75it/s] 90%|████████▉ | 358310/400000 [00:49<00:05, 7048.95it/s] 90%|████████▉ | 359039/400000 [00:50<00:05, 7118.18it/s] 90%|████████▉ | 359785/400000 [00:50<00:05, 7215.76it/s] 90%|█████████ | 360529/400000 [00:50<00:05, 7279.75it/s] 90%|█████████ | 361281/400000 [00:50<00:05, 7349.28it/s] 91%|█████████ | 362031/400000 [00:50<00:05, 7393.09it/s] 91%|█████████ | 362772/400000 [00:50<00:05, 7382.73it/s] 91%|█████████ | 363511/400000 [00:50<00:04, 7353.55it/s] 91%|█████████ | 364256/400000 [00:50<00:04, 7381.58it/s] 91%|█████████ | 364995/400000 [00:50<00:04, 7352.56it/s] 91%|█████████▏| 365731/400000 [00:50<00:04, 7299.77it/s] 92%|█████████▏| 366480/400000 [00:51<00:04, 7354.75it/s] 92%|█████████▏| 367216/400000 [00:51<00:04, 7331.43it/s] 92%|█████████▏| 367950/400000 [00:51<00:04, 7281.78it/s] 92%|█████████▏| 368711/400000 [00:51<00:04, 7376.72it/s] 92%|█████████▏| 369478/400000 [00:51<00:04, 7459.56it/s] 93%|█████████▎| 370232/400000 [00:51<00:03, 7481.63it/s] 93%|█████████▎| 370998/400000 [00:51<00:03, 7533.63it/s] 93%|█████████▎| 371752/400000 [00:51<00:03, 7437.38it/s] 93%|█████████▎| 372497/400000 [00:51<00:03, 7429.21it/s] 93%|█████████▎| 373246/400000 [00:51<00:03, 7446.82it/s] 93%|█████████▎| 373991/400000 [00:52<00:03, 7420.54it/s] 94%|█████████▎| 374743/400000 [00:52<00:03, 7449.24it/s] 94%|█████████▍| 375489/400000 [00:52<00:03, 7394.18it/s] 94%|█████████▍| 376229/400000 [00:52<00:03, 7362.72it/s] 94%|█████████▍| 376985/400000 [00:52<00:03, 7419.81it/s] 94%|█████████▍| 377750/400000 [00:52<00:02, 7487.18it/s] 95%|█████████▍| 378507/400000 [00:52<00:02, 7509.76it/s] 95%|█████████▍| 379259/400000 [00:52<00:02, 7481.26it/s] 95%|█████████▌| 380008/400000 [00:52<00:02, 7414.27it/s] 95%|█████████▌| 380756/400000 [00:52<00:02, 7432.94it/s] 95%|█████████▌| 381522/400000 [00:53<00:02, 7497.32it/s] 96%|█████████▌| 382288/400000 [00:53<00:02, 7543.12it/s] 96%|█████████▌| 383048/400000 [00:53<00:02, 7557.72it/s] 96%|█████████▌| 383804/400000 [00:53<00:02, 7555.27it/s] 96%|█████████▌| 384560/400000 [00:53<00:02, 7499.65it/s] 96%|█████████▋| 385317/400000 [00:53<00:01, 7520.39it/s] 97%|█████████▋| 386075/400000 [00:53<00:01, 7537.90it/s] 97%|█████████▋| 386829/400000 [00:53<00:01, 7522.59it/s] 97%|█████████▋| 387583/400000 [00:53<00:01, 7526.51it/s] 97%|█████████▋| 388342/400000 [00:53<00:01, 7545.40it/s] 97%|█████████▋| 389097/400000 [00:54<00:01, 7515.26it/s] 97%|█████████▋| 389849/400000 [00:54<00:01, 7506.74it/s] 98%|█████████▊| 390600/400000 [00:54<00:01, 7474.53it/s] 98%|█████████▊| 391348/400000 [00:54<00:01, 7284.54it/s] 98%|█████████▊| 392114/400000 [00:54<00:01, 7390.92it/s] 98%|█████████▊| 392874/400000 [00:54<00:00, 7451.58it/s] 98%|█████████▊| 393621/400000 [00:54<00:00, 7410.21it/s] 99%|█████████▊| 394376/400000 [00:54<00:00, 7451.43it/s] 99%|█████████▉| 395145/400000 [00:54<00:00, 7520.52it/s] 99%|█████████▉| 395906/400000 [00:54<00:00, 7547.09it/s] 99%|█████████▉| 396673/400000 [00:55<00:00, 7581.83it/s] 99%|█████████▉| 397432/400000 [00:55<00:00, 7522.40it/s]100%|█████████▉| 398188/400000 [00:55<00:00, 7533.03it/s]100%|█████████▉| 398955/400000 [00:55<00:00, 7571.53it/s]100%|█████████▉| 399724/400000 [00:55<00:00, 7606.54it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7203.26it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc68bfb20f0>, <torchtext.data.dataset.TabularDataset object at 0x7fc68bfb2240>, <torchtext.vocab.Vocab object at 0x7fc68bfb2160>), {}) 

