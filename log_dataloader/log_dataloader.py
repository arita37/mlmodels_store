
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdf705252f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdf705252f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fdfe25770d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fdfe25770d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdf79d0d9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdf79d0d9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdf8fe5b8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 2981888/9912422 [00:00<00:00, 29801997.69it/s]9920512it [00:00, 34487635.95it/s]                             
0it [00:00, ?it/s]32768it [00:00, 419253.83it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8206989.16it/s]          
0it [00:00, ?it/s]8192it [00:00, 273595.29it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdf794129b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdf706b5d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fdf8fe5b510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fdf8fe5b510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fdf8fe5b510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:01,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:01,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:01,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:00,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:42,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:42,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:41,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:41,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:40,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:19,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:18,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  7.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:12,  9.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 13.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 28.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 43.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 65.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 70.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 73.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 76.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.19 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.19 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.16 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ url]
0 examples [00:00, ? examples/s]2020-06-02 18:09:49.703399: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-02 18:09:49.707842: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-02 18:09:49.708049: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fd7bf3b370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-02 18:09:49.708168: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
60 examples [00:00, 594.99 examples/s]144 examples [00:00, 651.84 examples/s]221 examples [00:00, 681.63 examples/s]303 examples [00:00, 716.89 examples/s]387 examples [00:00, 747.08 examples/s]469 examples [00:00, 766.25 examples/s]558 examples [00:00, 797.00 examples/s]642 examples [00:00, 807.40 examples/s]725 examples [00:00, 812.04 examples/s]807 examples [00:01, 811.64 examples/s]891 examples [00:01, 819.48 examples/s]977 examples [00:01, 830.54 examples/s]1061 examples [00:01, 830.73 examples/s]1144 examples [00:01, 829.28 examples/s]1227 examples [00:01, 826.23 examples/s]1310 examples [00:01, 815.21 examples/s]1396 examples [00:01, 825.54 examples/s]1479 examples [00:01, 822.82 examples/s]1563 examples [00:01, 826.25 examples/s]1651 examples [00:02, 840.53 examples/s]1738 examples [00:02, 847.37 examples/s]1823 examples [00:02, 842.40 examples/s]1908 examples [00:02, 844.33 examples/s]1998 examples [00:02, 859.04 examples/s]2085 examples [00:02, 860.37 examples/s]2173 examples [00:02, 865.51 examples/s]2260 examples [00:02, 843.37 examples/s]2345 examples [00:02, 830.43 examples/s]2431 examples [00:02, 839.00 examples/s]2521 examples [00:03, 854.30 examples/s]2607 examples [00:03, 852.23 examples/s]2693 examples [00:03, 842.55 examples/s]2778 examples [00:03, 805.35 examples/s]2860 examples [00:03, 807.28 examples/s]2944 examples [00:03, 814.93 examples/s]3028 examples [00:03, 821.41 examples/s]3111 examples [00:03, 806.82 examples/s]3192 examples [00:03, 786.97 examples/s]3279 examples [00:03, 808.81 examples/s]3369 examples [00:04, 832.40 examples/s]3454 examples [00:04, 836.40 examples/s]3542 examples [00:04, 847.17 examples/s]3627 examples [00:04, 811.85 examples/s]3709 examples [00:04, 809.90 examples/s]3793 examples [00:04, 816.79 examples/s]3875 examples [00:04, 800.49 examples/s]3963 examples [00:04, 820.76 examples/s]4047 examples [00:04, 825.54 examples/s]4130 examples [00:05, 819.85 examples/s]4216 examples [00:05, 829.73 examples/s]4302 examples [00:05, 836.12 examples/s]4386 examples [00:05, 836.27 examples/s]4470 examples [00:05, 830.65 examples/s]4557 examples [00:05, 840.47 examples/s]4642 examples [00:05, 821.49 examples/s]4730 examples [00:05, 836.79 examples/s]4814 examples [00:05, 825.59 examples/s]4903 examples [00:05, 842.26 examples/s]4990 examples [00:06, 849.96 examples/s]5076 examples [00:06, 851.04 examples/s]5162 examples [00:06, 845.37 examples/s]5247 examples [00:06, 826.97 examples/s]5330 examples [00:06, 825.76 examples/s]5418 examples [00:06, 839.25 examples/s]5505 examples [00:06, 846.48 examples/s]5590 examples [00:06, 837.70 examples/s]5674 examples [00:06, 827.10 examples/s]5757 examples [00:06, 825.39 examples/s]5840 examples [00:07, 825.73 examples/s]5925 examples [00:07, 831.46 examples/s]6011 examples [00:07, 837.62 examples/s]6095 examples [00:07, 827.52 examples/s]6178 examples [00:07, 821.26 examples/s]6261 examples [00:07, 816.44 examples/s]6343 examples [00:07, 815.70 examples/s]6425 examples [00:07, 795.77 examples/s]6508 examples [00:07, 804.37 examples/s]6596 examples [00:07, 824.00 examples/s]6681 examples [00:08, 829.28 examples/s]6765 examples [00:08, 809.16 examples/s]6850 examples [00:08, 818.68 examples/s]6933 examples [00:08, 796.13 examples/s]7017 examples [00:08, 808.67 examples/s]7101 examples [00:08, 816.69 examples/s]7183 examples [00:08, 786.74 examples/s]7263 examples [00:08, 785.14 examples/s]7343 examples [00:08, 787.95 examples/s]7427 examples [00:09, 801.19 examples/s]7510 examples [00:09, 806.52 examples/s]7595 examples [00:09, 816.75 examples/s]7680 examples [00:09, 826.03 examples/s]7763 examples [00:09, 820.29 examples/s]7846 examples [00:09, 822.55 examples/s]7930 examples [00:09, 826.90 examples/s]8013 examples [00:09, 815.21 examples/s]8095 examples [00:09, 779.90 examples/s]8174 examples [00:09, 769.09 examples/s]8255 examples [00:10, 779.57 examples/s]8334 examples [00:10, 775.52 examples/s]8420 examples [00:10, 798.18 examples/s]8506 examples [00:10, 814.23 examples/s]8591 examples [00:10, 824.63 examples/s]8676 examples [00:10, 831.52 examples/s]8761 examples [00:10, 836.48 examples/s]8845 examples [00:10, 821.37 examples/s]8928 examples [00:10, 786.36 examples/s]9008 examples [00:10, 775.19 examples/s]9087 examples [00:11, 777.04 examples/s]9165 examples [00:11, 773.20 examples/s]9247 examples [00:11, 785.33 examples/s]9327 examples [00:11, 786.88 examples/s]9415 examples [00:11, 811.58 examples/s]9499 examples [00:11, 818.11 examples/s]9587 examples [00:11, 834.22 examples/s]9671 examples [00:11, 815.71 examples/s]9753 examples [00:11, 801.73 examples/s]9836 examples [00:12, 808.03 examples/s]9917 examples [00:12, 799.19 examples/s]9998 examples [00:12, 772.42 examples/s]10076 examples [00:12, 707.53 examples/s]10153 examples [00:12, 725.00 examples/s]10227 examples [00:12, 726.68 examples/s]10306 examples [00:12, 741.98 examples/s]10393 examples [00:12, 774.32 examples/s]10480 examples [00:12, 799.21 examples/s]10565 examples [00:12, 813.56 examples/s]10650 examples [00:13, 822.53 examples/s]10733 examples [00:13, 802.03 examples/s]10814 examples [00:13, 803.54 examples/s]10895 examples [00:13, 779.65 examples/s]10976 examples [00:13, 787.51 examples/s]11056 examples [00:13, 775.88 examples/s]11142 examples [00:13, 797.51 examples/s]11231 examples [00:13, 821.84 examples/s]11314 examples [00:13, 796.27 examples/s]11396 examples [00:14, 800.78 examples/s]11477 examples [00:14, 788.63 examples/s]11557 examples [00:14, 780.36 examples/s]11636 examples [00:14, 765.53 examples/s]11716 examples [00:14, 775.26 examples/s]11797 examples [00:14, 783.20 examples/s]11880 examples [00:14, 795.67 examples/s]11960 examples [00:14, 774.71 examples/s]12044 examples [00:14, 792.62 examples/s]12124 examples [00:14, 793.45 examples/s]12208 examples [00:15, 806.75 examples/s]12294 examples [00:15, 820.51 examples/s]12377 examples [00:15, 820.55 examples/s]12463 examples [00:15, 830.88 examples/s]12548 examples [00:15, 834.35 examples/s]12632 examples [00:15, 821.69 examples/s]12715 examples [00:15, 822.28 examples/s]12798 examples [00:15, 818.48 examples/s]12880 examples [00:15, 813.88 examples/s]12964 examples [00:15, 821.42 examples/s]13047 examples [00:16, 818.58 examples/s]13129 examples [00:16, 803.57 examples/s]13210 examples [00:16, 779.39 examples/s]13289 examples [00:16, 729.76 examples/s]13363 examples [00:16, 726.46 examples/s]13442 examples [00:16, 742.28 examples/s]13523 examples [00:16, 761.18 examples/s]13605 examples [00:16, 776.43 examples/s]13685 examples [00:16, 782.59 examples/s]13764 examples [00:17, 774.92 examples/s]13844 examples [00:17, 779.81 examples/s]13924 examples [00:17, 785.62 examples/s]14006 examples [00:17, 794.70 examples/s]14087 examples [00:17, 798.32 examples/s]14169 examples [00:17, 804.41 examples/s]14250 examples [00:17, 801.78 examples/s]14332 examples [00:17, 805.35 examples/s]14416 examples [00:17, 813.93 examples/s]14498 examples [00:17, 787.03 examples/s]14583 examples [00:18, 803.87 examples/s]14670 examples [00:18, 819.99 examples/s]14757 examples [00:18, 832.20 examples/s]14846 examples [00:18, 848.22 examples/s]14932 examples [00:18, 831.50 examples/s]15016 examples [00:18, 827.94 examples/s]15100 examples [00:18, 829.54 examples/s]15184 examples [00:18, 822.95 examples/s]15267 examples [00:18, 816.39 examples/s]15349 examples [00:18, 814.27 examples/s]15431 examples [00:19, 775.76 examples/s]15509 examples [00:19, 770.97 examples/s]15593 examples [00:19, 788.49 examples/s]15673 examples [00:19, 780.15 examples/s]15752 examples [00:19, 776.40 examples/s]15833 examples [00:19, 785.50 examples/s]15918 examples [00:19, 802.43 examples/s]15999 examples [00:19, 794.75 examples/s]16079 examples [00:19, 754.24 examples/s]16155 examples [00:20, 715.37 examples/s]16236 examples [00:20, 739.63 examples/s]16324 examples [00:20, 775.90 examples/s]16414 examples [00:20, 807.23 examples/s]16503 examples [00:20, 828.17 examples/s]16590 examples [00:20, 837.55 examples/s]16676 examples [00:20, 842.80 examples/s]16764 examples [00:20, 853.44 examples/s]16850 examples [00:20, 853.13 examples/s]16936 examples [00:20, 840.19 examples/s]17021 examples [00:21, 835.87 examples/s]17105 examples [00:21, 827.62 examples/s]17188 examples [00:21, 808.83 examples/s]17270 examples [00:21, 770.65 examples/s]17355 examples [00:21, 792.85 examples/s]17441 examples [00:21, 810.48 examples/s]17532 examples [00:21, 836.67 examples/s]17620 examples [00:21, 849.00 examples/s]17706 examples [00:21, 780.85 examples/s]17786 examples [00:22, 768.69 examples/s]17864 examples [00:22, 757.12 examples/s]17941 examples [00:22, 760.82 examples/s]18025 examples [00:22, 781.62 examples/s]18106 examples [00:22, 787.44 examples/s]18186 examples [00:22, 781.05 examples/s]18271 examples [00:22, 800.39 examples/s]18363 examples [00:22, 830.47 examples/s]18447 examples [00:22, 831.18 examples/s]18532 examples [00:22, 835.93 examples/s]18616 examples [00:23, 796.41 examples/s]18697 examples [00:23, 774.01 examples/s]18777 examples [00:23, 781.17 examples/s]18856 examples [00:23, 767.49 examples/s]18934 examples [00:23, 739.01 examples/s]19011 examples [00:23, 747.27 examples/s]19087 examples [00:23, 749.92 examples/s]19165 examples [00:23, 757.08 examples/s]19245 examples [00:23, 769.33 examples/s]19323 examples [00:23, 772.21 examples/s]19403 examples [00:24, 778.18 examples/s]19487 examples [00:24, 793.53 examples/s]19575 examples [00:24, 815.80 examples/s]19661 examples [00:24, 827.23 examples/s]19746 examples [00:24, 831.81 examples/s]19830 examples [00:24, 817.13 examples/s]19912 examples [00:24, 815.51 examples/s]19994 examples [00:24, 811.60 examples/s]20076 examples [00:24, 758.00 examples/s]20159 examples [00:25, 778.22 examples/s]20240 examples [00:25, 785.32 examples/s]20323 examples [00:25, 795.67 examples/s]20403 examples [00:25, 787.86 examples/s]20487 examples [00:25, 800.72 examples/s]20570 examples [00:25, 808.33 examples/s]20652 examples [00:25, 807.41 examples/s]20733 examples [00:25, 790.31 examples/s]20816 examples [00:25, 801.62 examples/s]20900 examples [00:25, 809.88 examples/s]20982 examples [00:26, 808.58 examples/s]21063 examples [00:26, 793.41 examples/s]21144 examples [00:26, 795.87 examples/s]21224 examples [00:26, 780.25 examples/s]21307 examples [00:26, 793.90 examples/s]21387 examples [00:26, 793.32 examples/s]21471 examples [00:26, 804.43 examples/s]21557 examples [00:26, 818.80 examples/s]21640 examples [00:26, 815.47 examples/s]21723 examples [00:26, 818.89 examples/s]21805 examples [00:27, 800.65 examples/s]21888 examples [00:27, 808.90 examples/s]21970 examples [00:27, 811.69 examples/s]22052 examples [00:27, 811.74 examples/s]22134 examples [00:27, 802.89 examples/s]22222 examples [00:27, 824.46 examples/s]22306 examples [00:27, 826.91 examples/s]22393 examples [00:27, 838.26 examples/s]22477 examples [00:27, 826.00 examples/s]22560 examples [00:27, 827.00 examples/s]22643 examples [00:28, 817.30 examples/s]22725 examples [00:28, 790.87 examples/s]22805 examples [00:28, 791.21 examples/s]22885 examples [00:28, 793.23 examples/s]22966 examples [00:28, 797.67 examples/s]23046 examples [00:28, 744.53 examples/s]23122 examples [00:28, 722.76 examples/s]23205 examples [00:28, 750.74 examples/s]23283 examples [00:28, 757.54 examples/s]23368 examples [00:29, 780.55 examples/s]23451 examples [00:29, 793.49 examples/s]23532 examples [00:29, 797.01 examples/s]23613 examples [00:29, 800.44 examples/s]23695 examples [00:29, 805.65 examples/s]23778 examples [00:29, 808.83 examples/s]23860 examples [00:29, 794.55 examples/s]23942 examples [00:29, 800.56 examples/s]24025 examples [00:29, 807.10 examples/s]24106 examples [00:29, 793.21 examples/s]24195 examples [00:30, 818.44 examples/s]24278 examples [00:30, 803.77 examples/s]24359 examples [00:30, 802.67 examples/s]24440 examples [00:30, 796.05 examples/s]24520 examples [00:30, 796.93 examples/s]24603 examples [00:30, 803.56 examples/s]24684 examples [00:30, 805.21 examples/s]24767 examples [00:30, 811.42 examples/s]24849 examples [00:30, 810.98 examples/s]24931 examples [00:30, 812.33 examples/s]25016 examples [00:31, 822.60 examples/s]25099 examples [00:31, 804.17 examples/s]25180 examples [00:31, 803.69 examples/s]25261 examples [00:31, 797.48 examples/s]25342 examples [00:31, 799.33 examples/s]25422 examples [00:31, 785.50 examples/s]25501 examples [00:31, 780.91 examples/s]25585 examples [00:31, 796.48 examples/s]25665 examples [00:31, 788.02 examples/s]25753 examples [00:31, 812.12 examples/s]25836 examples [00:32, 816.57 examples/s]25918 examples [00:32, 810.29 examples/s]26000 examples [00:32, 804.61 examples/s]26081 examples [00:32, 780.79 examples/s]26160 examples [00:32, 774.96 examples/s]26241 examples [00:32, 784.66 examples/s]26328 examples [00:32, 808.28 examples/s]26417 examples [00:32, 828.92 examples/s]26505 examples [00:32, 842.64 examples/s]26596 examples [00:33, 860.23 examples/s]26685 examples [00:33, 866.65 examples/s]26772 examples [00:33, 835.10 examples/s]26856 examples [00:33, 827.63 examples/s]26940 examples [00:33, 815.33 examples/s]27030 examples [00:33, 836.69 examples/s]27114 examples [00:33, 824.62 examples/s]27197 examples [00:33, 823.73 examples/s]27283 examples [00:33, 832.04 examples/s]27369 examples [00:33, 838.15 examples/s]27457 examples [00:34, 848.83 examples/s]27543 examples [00:34, 852.05 examples/s]27629 examples [00:34, 812.64 examples/s]27711 examples [00:34, 806.60 examples/s]27792 examples [00:34, 778.73 examples/s]27875 examples [00:34, 791.85 examples/s]27955 examples [00:34, 779.33 examples/s]28038 examples [00:34, 791.47 examples/s]28126 examples [00:34, 813.61 examples/s]28216 examples [00:34, 836.13 examples/s]28301 examples [00:35, 824.72 examples/s]28384 examples [00:35, 801.84 examples/s]28465 examples [00:35, 798.17 examples/s]28546 examples [00:35, 801.31 examples/s]28627 examples [00:35, 793.37 examples/s]28713 examples [00:35, 811.30 examples/s]28802 examples [00:35, 830.08 examples/s]28889 examples [00:35, 840.01 examples/s]28976 examples [00:35, 846.19 examples/s]29061 examples [00:36, 801.53 examples/s]29143 examples [00:36, 804.41 examples/s]29224 examples [00:36, 792.97 examples/s]29304 examples [00:36, 785.38 examples/s]29390 examples [00:36, 805.36 examples/s]29477 examples [00:36, 821.13 examples/s]29560 examples [00:36, 819.77 examples/s]29643 examples [00:36, 819.96 examples/s]29726 examples [00:36, 817.05 examples/s]29808 examples [00:36, 811.92 examples/s]29890 examples [00:37, 803.24 examples/s]29971 examples [00:37, 768.93 examples/s]30049 examples [00:37, 699.16 examples/s]30124 examples [00:37, 712.30 examples/s]30206 examples [00:37, 739.16 examples/s]30285 examples [00:37, 753.05 examples/s]30369 examples [00:37, 776.79 examples/s]30459 examples [00:37, 808.53 examples/s]30541 examples [00:37, 792.46 examples/s]30621 examples [00:38, 788.73 examples/s]30702 examples [00:38, 793.52 examples/s]30782 examples [00:38, 793.21 examples/s]30864 examples [00:38, 797.94 examples/s]30947 examples [00:38, 803.29 examples/s]31028 examples [00:38, 791.21 examples/s]31109 examples [00:38, 794.34 examples/s]31190 examples [00:38, 795.06 examples/s]31272 examples [00:38, 800.35 examples/s]31355 examples [00:38, 807.60 examples/s]31439 examples [00:39, 815.84 examples/s]31521 examples [00:39, 805.44 examples/s]31603 examples [00:39, 809.24 examples/s]31684 examples [00:39, 784.15 examples/s]31763 examples [00:39, 761.75 examples/s]31840 examples [00:39, 758.44 examples/s]31920 examples [00:39, 769.56 examples/s]32003 examples [00:39, 786.63 examples/s]32082 examples [00:39, 771.32 examples/s]32166 examples [00:39, 790.61 examples/s]32249 examples [00:40, 799.95 examples/s]32331 examples [00:40, 804.65 examples/s]32413 examples [00:40, 808.13 examples/s]32500 examples [00:40, 824.39 examples/s]32584 examples [00:40, 827.55 examples/s]32669 examples [00:40, 832.04 examples/s]32753 examples [00:40, 825.07 examples/s]32836 examples [00:40, 818.65 examples/s]32918 examples [00:40, 813.22 examples/s]33001 examples [00:40, 814.96 examples/s]33086 examples [00:41, 824.01 examples/s]33170 examples [00:41, 826.12 examples/s]33255 examples [00:41, 832.47 examples/s]33339 examples [00:41, 820.29 examples/s]33424 examples [00:41, 826.35 examples/s]33511 examples [00:41, 836.68 examples/s]33598 examples [00:41, 845.06 examples/s]33683 examples [00:41, 841.03 examples/s]33768 examples [00:41, 831.73 examples/s]33852 examples [00:42, 829.61 examples/s]33936 examples [00:42, 828.24 examples/s]34022 examples [00:42, 836.46 examples/s]34106 examples [00:42, 832.28 examples/s]34190 examples [00:42, 784.57 examples/s]34270 examples [00:42, 782.65 examples/s]34349 examples [00:42, 716.44 examples/s]34423 examples [00:42, 701.13 examples/s]34495 examples [00:42, 657.33 examples/s]34563 examples [00:43, 599.11 examples/s]34647 examples [00:43, 655.23 examples/s]34733 examples [00:43, 703.68 examples/s]34811 examples [00:43, 722.89 examples/s]34886 examples [00:43, 720.11 examples/s]34963 examples [00:43, 732.55 examples/s]35051 examples [00:43, 770.37 examples/s]35143 examples [00:43, 808.35 examples/s]35226 examples [00:43, 781.95 examples/s]35306 examples [00:43, 769.71 examples/s]35391 examples [00:44, 792.07 examples/s]35478 examples [00:44, 812.66 examples/s]35561 examples [00:44, 816.30 examples/s]35644 examples [00:44, 808.08 examples/s]35726 examples [00:44, 811.06 examples/s]35808 examples [00:44, 811.97 examples/s]35892 examples [00:44, 817.87 examples/s]35977 examples [00:44, 827.07 examples/s]36060 examples [00:44, 823.85 examples/s]36143 examples [00:44, 801.53 examples/s]36224 examples [00:45, 790.53 examples/s]36313 examples [00:45, 816.84 examples/s]36401 examples [00:45, 831.98 examples/s]36485 examples [00:45, 780.61 examples/s]36564 examples [00:45, 753.45 examples/s]36643 examples [00:45, 761.49 examples/s]36726 examples [00:45, 777.93 examples/s]36805 examples [00:45, 774.68 examples/s]36890 examples [00:45, 794.20 examples/s]36974 examples [00:46, 806.32 examples/s]37061 examples [00:46, 821.70 examples/s]37144 examples [00:46, 822.89 examples/s]37232 examples [00:46, 836.93 examples/s]37316 examples [00:46, 829.04 examples/s]37400 examples [00:46, 814.85 examples/s]37482 examples [00:46, 782.61 examples/s]37563 examples [00:46, 790.49 examples/s]37643 examples [00:46, 786.62 examples/s]37726 examples [00:46, 797.52 examples/s]37807 examples [00:47, 801.13 examples/s]37888 examples [00:47, 803.70 examples/s]37972 examples [00:47, 814.02 examples/s]38054 examples [00:47, 810.36 examples/s]38136 examples [00:47, 808.84 examples/s]38217 examples [00:47, 806.17 examples/s]38301 examples [00:47, 814.66 examples/s]38383 examples [00:47, 815.20 examples/s]38465 examples [00:47, 798.36 examples/s]38545 examples [00:47, 793.19 examples/s]38625 examples [00:48, 777.64 examples/s]38706 examples [00:48, 784.91 examples/s]38789 examples [00:48, 797.64 examples/s]38869 examples [00:48, 788.02 examples/s]38948 examples [00:48, 770.02 examples/s]39029 examples [00:48, 780.29 examples/s]39108 examples [00:48, 775.60 examples/s]39186 examples [00:48, 771.35 examples/s]39270 examples [00:48, 788.66 examples/s]39350 examples [00:49, 778.01 examples/s]39428 examples [00:49, 777.60 examples/s]39510 examples [00:49, 788.35 examples/s]39591 examples [00:49, 793.49 examples/s]39672 examples [00:49, 797.15 examples/s]39755 examples [00:49, 803.91 examples/s]39836 examples [00:49, 794.87 examples/s]39921 examples [00:49, 808.71 examples/s]40002 examples [00:49, 768.93 examples/s]40086 examples [00:49, 787.60 examples/s]40166 examples [00:50, 771.00 examples/s]40248 examples [00:50, 783.40 examples/s]40327 examples [00:50, 776.07 examples/s]40408 examples [00:50, 784.90 examples/s]40487 examples [00:50, 774.58 examples/s]40565 examples [00:50, 763.38 examples/s]40647 examples [00:50, 777.85 examples/s]40728 examples [00:50, 785.18 examples/s]40815 examples [00:50, 806.63 examples/s]40904 examples [00:50, 828.88 examples/s]40993 examples [00:51, 843.91 examples/s]41078 examples [00:51, 829.37 examples/s]41162 examples [00:51, 829.56 examples/s]41246 examples [00:51, 823.12 examples/s]41329 examples [00:51, 822.14 examples/s]41412 examples [00:51, 796.12 examples/s]41492 examples [00:51, 786.79 examples/s]41574 examples [00:51, 792.16 examples/s]41654 examples [00:51, 779.63 examples/s]41735 examples [00:52, 786.32 examples/s]41814 examples [00:52, 785.85 examples/s]41894 examples [00:52, 789.72 examples/s]41979 examples [00:52, 806.43 examples/s]42066 examples [00:52, 823.25 examples/s]42157 examples [00:52, 844.23 examples/s]42242 examples [00:52, 836.03 examples/s]42327 examples [00:52, 838.81 examples/s]42412 examples [00:52, 832.28 examples/s]42496 examples [00:52, 833.53 examples/s]42580 examples [00:53, 832.41 examples/s]42664 examples [00:53, 821.71 examples/s]42747 examples [00:53, 798.64 examples/s]42830 examples [00:53, 807.35 examples/s]42913 examples [00:53, 813.03 examples/s]42995 examples [00:53, 811.39 examples/s]43077 examples [00:53, 798.29 examples/s]43165 examples [00:53, 821.03 examples/s]43252 examples [00:53, 833.29 examples/s]43339 examples [00:53, 841.89 examples/s]43425 examples [00:54, 846.78 examples/s]43510 examples [00:54, 837.56 examples/s]43596 examples [00:54, 841.03 examples/s]43681 examples [00:54, 842.42 examples/s]43766 examples [00:54, 821.49 examples/s]43849 examples [00:54, 819.07 examples/s]43938 examples [00:54, 837.82 examples/s]44022 examples [00:54, 837.63 examples/s]44107 examples [00:54, 839.16 examples/s]44192 examples [00:54, 816.09 examples/s]44274 examples [00:55, 815.13 examples/s]44361 examples [00:55, 828.03 examples/s]44446 examples [00:55, 831.41 examples/s]44532 examples [00:55, 837.19 examples/s]44620 examples [00:55, 849.18 examples/s]44706 examples [00:55, 837.95 examples/s]44790 examples [00:55, 831.19 examples/s]44874 examples [00:55, 810.02 examples/s]44956 examples [00:55, 774.40 examples/s]45034 examples [00:56, 749.96 examples/s]45121 examples [00:56, 781.53 examples/s]45200 examples [00:56, 776.63 examples/s]45281 examples [00:56, 786.10 examples/s]45365 examples [00:56, 800.91 examples/s]45447 examples [00:56, 802.88 examples/s]45528 examples [00:56, 783.63 examples/s]45613 examples [00:56, 799.93 examples/s]45698 examples [00:56, 812.17 examples/s]45782 examples [00:56, 818.89 examples/s]45867 examples [00:57, 826.11 examples/s]45950 examples [00:57, 816.82 examples/s]46032 examples [00:57, 796.47 examples/s]46112 examples [00:57, 765.87 examples/s]46189 examples [00:57, 749.51 examples/s]46268 examples [00:57, 760.38 examples/s]46345 examples [00:57, 757.56 examples/s]46436 examples [00:57, 797.61 examples/s]46522 examples [00:57, 815.00 examples/s]46605 examples [00:58, 815.66 examples/s]46688 examples [00:58, 818.58 examples/s]46771 examples [00:58, 810.20 examples/s]46855 examples [00:58, 817.57 examples/s]46938 examples [00:58, 820.88 examples/s]47021 examples [00:58, 821.48 examples/s]47104 examples [00:58, 793.09 examples/s]47185 examples [00:58, 795.85 examples/s]47265 examples [00:58, 786.43 examples/s]47346 examples [00:58, 792.62 examples/s]47428 examples [00:59, 798.16 examples/s]47514 examples [00:59, 815.18 examples/s]47598 examples [00:59, 821.83 examples/s]47684 examples [00:59, 832.23 examples/s]47770 examples [00:59, 837.25 examples/s]47854 examples [00:59, 828.06 examples/s]47937 examples [00:59, 816.04 examples/s]48020 examples [00:59, 820.02 examples/s]48103 examples [00:59, 800.02 examples/s]48184 examples [00:59, 802.81 examples/s]48269 examples [01:00, 813.67 examples/s]48351 examples [01:00, 782.92 examples/s]48431 examples [01:00, 787.02 examples/s]48510 examples [01:00, 769.38 examples/s]48591 examples [01:00, 778.95 examples/s]48670 examples [01:00, 781.82 examples/s]48750 examples [01:00, 785.93 examples/s]48829 examples [01:00, 772.26 examples/s]48915 examples [01:00, 795.08 examples/s]49000 examples [01:00, 809.64 examples/s]49090 examples [01:01, 834.64 examples/s]49174 examples [01:01, 833.93 examples/s]49258 examples [01:01, 824.90 examples/s]49341 examples [01:01, 805.33 examples/s]49422 examples [01:01, 765.93 examples/s]49500 examples [01:01, 758.45 examples/s]49585 examples [01:01, 782.81 examples/s]49664 examples [01:01, 784.59 examples/s]49743 examples [01:01, 773.75 examples/s]49826 examples [01:02, 789.38 examples/s]49909 examples [01:02, 799.24 examples/s]49993 examples [01:02, 810.22 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5693/50000 [00:00<00:00, 56924.63 examples/s] 30%|██▉       | 14895/50000 [00:00<00:00, 64278.99 examples/s] 52%|█████▏    | 25970/50000 [00:00<00:00, 73534.31 examples/s] 75%|███████▌  | 37532/50000 [00:00<00:00, 82547.88 examples/s] 98%|█████████▊| 49097/50000 [00:00<00:00, 90300.94 examples/s]                                                               0 examples [00:00, ? examples/s]66 examples [00:00, 657.26 examples/s]154 examples [00:00, 709.84 examples/s]238 examples [00:00, 743.15 examples/s]322 examples [00:00, 767.83 examples/s]408 examples [00:00, 792.78 examples/s]493 examples [00:00, 805.79 examples/s]571 examples [00:00, 796.16 examples/s]646 examples [00:00, 779.54 examples/s]721 examples [00:00, 627.82 examples/s]794 examples [00:01, 653.74 examples/s]883 examples [00:01, 710.18 examples/s]975 examples [00:01, 760.41 examples/s]1063 examples [00:01, 791.41 examples/s]1146 examples [00:01, 801.82 examples/s]1229 examples [00:01, 809.19 examples/s]1312 examples [00:01, 807.72 examples/s]1394 examples [00:01, 790.66 examples/s]1476 examples [00:01, 798.63 examples/s]1558 examples [00:01, 803.90 examples/s]1639 examples [00:02, 805.09 examples/s]1722 examples [00:02, 810.26 examples/s]1806 examples [00:02, 816.42 examples/s]1895 examples [00:02, 835.63 examples/s]1980 examples [00:02, 838.09 examples/s]2067 examples [00:02, 845.98 examples/s]2157 examples [00:02, 858.57 examples/s]2244 examples [00:02, 861.36 examples/s]2333 examples [00:02, 868.33 examples/s]2420 examples [00:03, 865.47 examples/s]2507 examples [00:03, 851.55 examples/s]2593 examples [00:03, 850.78 examples/s]2679 examples [00:03, 835.47 examples/s]2763 examples [00:03, 829.89 examples/s]2847 examples [00:03, 827.00 examples/s]2930 examples [00:03, 821.24 examples/s]3015 examples [00:03, 827.27 examples/s]3101 examples [00:03, 836.05 examples/s]3187 examples [00:03, 840.80 examples/s]3275 examples [00:04, 850.75 examples/s]3361 examples [00:04, 829.92 examples/s]3448 examples [00:04, 840.08 examples/s]3533 examples [00:04, 839.93 examples/s]3620 examples [00:04, 846.57 examples/s]3709 examples [00:04, 858.62 examples/s]3795 examples [00:04, 856.46 examples/s]3881 examples [00:04, 830.07 examples/s]3965 examples [00:04, 817.82 examples/s]4047 examples [00:04, 804.19 examples/s]4132 examples [00:05, 816.02 examples/s]4214 examples [00:05, 802.42 examples/s]4298 examples [00:05, 810.74 examples/s]4383 examples [00:05, 821.38 examples/s]4466 examples [00:05, 813.26 examples/s]4549 examples [00:05, 817.24 examples/s]4631 examples [00:05, 813.98 examples/s]4713 examples [00:05, 800.93 examples/s]4794 examples [00:05, 779.55 examples/s]4877 examples [00:05, 791.29 examples/s]4959 examples [00:06, 797.18 examples/s]5043 examples [00:06, 807.40 examples/s]5129 examples [00:06, 821.16 examples/s]5212 examples [00:06, 805.85 examples/s]5293 examples [00:06, 781.53 examples/s]5376 examples [00:06, 794.84 examples/s]5458 examples [00:06, 799.94 examples/s]5541 examples [00:06, 805.67 examples/s]5623 examples [00:06, 809.67 examples/s]5708 examples [00:07, 818.68 examples/s]5793 examples [00:07, 823.86 examples/s]5876 examples [00:07, 801.13 examples/s]5958 examples [00:07, 804.91 examples/s]6044 examples [00:07, 819.75 examples/s]6127 examples [00:07, 815.56 examples/s]6209 examples [00:07, 808.94 examples/s]6292 examples [00:07, 813.15 examples/s]6381 examples [00:07, 833.48 examples/s]6467 examples [00:07, 840.63 examples/s]6554 examples [00:08, 848.65 examples/s]6641 examples [00:08, 853.14 examples/s]6727 examples [00:08, 829.90 examples/s]6811 examples [00:08, 830.09 examples/s]6895 examples [00:08, 826.90 examples/s]6979 examples [00:08, 829.55 examples/s]7063 examples [00:08, 827.12 examples/s]7146 examples [00:08, 825.62 examples/s]7229 examples [00:08, 818.36 examples/s]7311 examples [00:08, 810.81 examples/s]7393 examples [00:09, 806.88 examples/s]7474 examples [00:09, 800.08 examples/s]7555 examples [00:09, 799.16 examples/s]7635 examples [00:09, 799.17 examples/s]7717 examples [00:09, 803.27 examples/s]7800 examples [00:09, 810.89 examples/s]7884 examples [00:09, 817.52 examples/s]7966 examples [00:09, 810.33 examples/s]8048 examples [00:09, 810.09 examples/s]8135 examples [00:09, 824.98 examples/s]8218 examples [00:10, 825.09 examples/s]8301 examples [00:10, 821.28 examples/s]8384 examples [00:10, 807.16 examples/s]8466 examples [00:10, 807.99 examples/s]8549 examples [00:10, 811.95 examples/s]8633 examples [00:10, 818.03 examples/s]8717 examples [00:10, 824.46 examples/s]8800 examples [00:10, 826.10 examples/s]8883 examples [00:10, 823.40 examples/s]8966 examples [00:11, 819.78 examples/s]9049 examples [00:11, 821.91 examples/s]9132 examples [00:11, 821.52 examples/s]9215 examples [00:11, 810.46 examples/s]9297 examples [00:11, 805.50 examples/s]9378 examples [00:11, 798.03 examples/s]9458 examples [00:11, 783.91 examples/s]9538 examples [00:11, 788.57 examples/s]9617 examples [00:11, 771.75 examples/s]9695 examples [00:11, 768.08 examples/s]9785 examples [00:12, 802.62 examples/s]9871 examples [00:12, 817.20 examples/s]9962 examples [00:12, 841.47 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUUZAGN/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUUZAGN/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdf8fe5b8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdffc2feda0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdf68a1bf98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdf8fe5b8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdf8fe5b8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdf713cc3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdf713cc278>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fdf2c06b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fdf2c06b620> 

  function with postional parmater data_info <function split_train_valid at 0x7fdf2c06b620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fdf2c06b730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fdf2c06b730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fdf2c06b730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=68022108c9d5a2defb864eaf88496db0e21325c257b5762080517dbc93bb9088
  Stored in directory: /tmp/pip-ephem-wheel-cache-qe2bw93j/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:43:05, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:50:08, 14.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:50:24, 20.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:17:44, 28.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:47:31, 41.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.70M/862M [00:01<4:01:37, 58.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:48:36, 83.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:57:26, 120kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.2M/862M [00:01<1:21:54, 171kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.9M/862M [00:01<57:02, 244kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:02<39:45, 347kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<27:43, 495kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:02<19:27, 702kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<13:36, 998kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<09:38, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<08:08, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:35, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<07:26, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<05:44, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<06:22, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<05:53, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<04:31, 2.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.4M/862M [00:07<03:26, 3.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:30, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:41, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<05:02, 2.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:27, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<05:50, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<04:27, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<05:41, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<05:19, 2.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<04:03, 3.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<05:58, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:50, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<05:28, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<05:55, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<05:30, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<04:11, 3.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<05:56, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:18<05:19, 2.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:19<03:57, 3.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<06:56, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<06:10, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:20<04:36, 2.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<06:13, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:58, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<05:32, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:23<04:00, 3.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<12:17:58, 17.3kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<8:37:39, 24.6kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<6:01:55, 35.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<4:15:34, 49.6kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:01:28, 69.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<2:07:32, 99.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:30:58, 139kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:04:45, 195kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<45:35, 276kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<31:58, 392kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<46:49, 268kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<34:02, 368kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<24:03, 520kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<19:45, 631kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<15:04, 827kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:50, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:30, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:37, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:20, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:20, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:39, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:52, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:16, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:11, 1.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:49, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:16, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:57, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:25, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:03, 2.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:40, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:26, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:30, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:48, 3.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:28, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:10, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:55, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:21, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:59, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:47, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:11, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:56, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:20, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:56, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:45, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:21, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:55, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:44, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:20, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:55, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:41, 3.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:05, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:50, 2.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:30, 3.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:00:53, 17.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<7:43:33, 24.7kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<5:24:01, 35.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<3:48:45, 49.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<2:42:24, 70.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<1:54:09, 99.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:21:22, 139kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<58:05, 194kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<40:52, 276kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<31:08, 361kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<22:55, 490kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<16:18, 687kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<14:01, 796kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<10:56, 1.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<07:55, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<08:09, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:50, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:01, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:08, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:33, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:03, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:40, 2.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<08:24, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:01, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:09, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:31, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:07, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:19, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:40, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:32, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:58, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:40, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:30, 2.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:22, 3.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:51, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:34, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:26, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:12, 3.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<10:12:16, 17.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<7:09:26, 24.5kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<5:00:08, 35.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<3:31:50, 49.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<2:29:15, 70.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<1:44:29, 100kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<1:15:21, 138kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<53:47, 193kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<37:46, 275kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<28:47, 359kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<21:12, 487kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<15:04, 684kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<12:56, 794kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<10:06, 1.02MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<07:16, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:31, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:17, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:38, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:38, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:59, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:44, 2.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:59, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:33, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:23, 2.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:41, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:17, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:15, 3.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:35, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:11, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:33, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:11, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:08, 3.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:31, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:10, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:09, 3.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:08, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:28, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:07, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:07, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:27, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:05, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:06, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:25, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:03, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:04, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:23, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:02, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:03, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:00, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:17, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:57, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:57, 3.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:57, 3.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:56, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:41, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:45, 3.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<02:03, 4.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<35:18, 254kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<25:37, 350kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<18:04, 495kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<14:43, 604kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<11:12, 793kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<08:00, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<07:40, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<07:10, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:28, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:13, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:34, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:25, 2.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:24, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:50, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:04, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:43, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:47, 3.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:57, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:38, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:45, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:28, 2.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:36, 3.24MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<01:56, 4.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<33:08, 254kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<24:55, 338kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<17:50, 471kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<13:46, 607kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<10:29, 796kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<07:31, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<07:11, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:43, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:07, 1.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:53, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:16, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:09, 2.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:06, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:42, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:47, 2.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<03:51, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:52, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:09, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:09, 2.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<02:17, 3.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<58:31, 137kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<41:37, 192kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<29:33, 271kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<20:38, 385kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<26:48, 296kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<19:32, 406kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<13:50, 571kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<11:30, 684kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<08:54, 884kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:36, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<04:43, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<06:42, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:17, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:47, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:35, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:00, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:57, 2.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:52, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:16, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:19, 2.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:27, 3.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:08, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:41, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:46, 2.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:40, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:20, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<02:31, 2.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:31, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:14, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:24, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:27, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:02, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:21, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<01:43, 4.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<28:52, 254kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<20:56, 349kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<14:47, 493kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<12:01, 603kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<09:09, 792kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<06:34, 1.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:16, 1.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:07, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:43, 1.92MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:17, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<03:43, 1.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:47, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:36, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:13, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:25, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:20, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:03, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:16, 3.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:14, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:58, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:14, 3.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:11, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:49, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:07, 3.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<01:34, 4.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<26:39, 254kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<20:01, 338kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<14:17, 473kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<10:02, 670kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<11:04, 606kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<08:25, 795kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<06:02, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:46, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:24, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:03, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:54, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:07, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:57, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:37, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:05, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:11, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:15, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:20, 2.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<09:26, 681kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:16, 883kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:14, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<05:07, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:53, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:44, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:39, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<11:47, 534kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<08:54, 706kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:21, 985kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:52, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:23, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:01, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:55, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:59, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:25, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:31, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:10, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:51, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:07, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:20, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:37, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:54, 3.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:10, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:36, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:11, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:26, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:42, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:56, 2.99MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<07:53, 738kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<06:01, 964kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:22, 1.32MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<03:06, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<13:01, 441kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<10:17, 558kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<07:28, 766kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<05:15, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<09:19, 609kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<07:06, 798kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:05, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:50, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:56, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:53, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:19, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:09, 2.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:47, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:30, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:52, 2.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:35, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:55, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:18, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:40, 3.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:44, 929kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:29, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:16, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:29, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:59, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:12, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:46, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:27, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:49, 2.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:29, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:15, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:41, 3.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:22, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:10, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:04<01:38, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:19, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<01:35, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:16, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:05, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:35, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:04, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:32, 3.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:12, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:01, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:30, 3.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:10, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<01:55, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:27, 3.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:07, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:57, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:27, 3.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:06, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:56, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:28, 3.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:55, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:26, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:03, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:21, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:50, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:20, 3.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:29, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:11, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:38, 2.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:08, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:56, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:01, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<01:50, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:23, 3.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:57, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<01:47, 2.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:20, 3.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<00:59, 4.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<49:06, 83.5kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<35:11, 116kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<24:45, 165kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<17:10, 235kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<18:55, 213kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<13:38, 295kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<09:35, 417kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<07:34, 523kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<05:41, 694kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:03, 967kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:44, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:00, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:10, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:25, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:05, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:31, 2.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:57, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:45, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:18, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:47, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:37, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:12, 3.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:30, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:07, 3.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<00:49, 4.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<14:53, 239kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<10:45, 330kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<07:33, 465kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:03, 574kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<04:35, 757kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<03:16, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:04, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:30, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:49, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:03, 1.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:46, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:19, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:41, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:30, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:07, 2.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:32, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:24, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:03, 3.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:28, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:40, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:18, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<00:57, 3.21MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:36, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:26, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:03, 2.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:26, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:37, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:15, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:55, 3.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:44, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:30, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:06, 2.60MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:26, 1.98MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:35, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:14, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:54, 3.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:35, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:24, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:02, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:21, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:13, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:54, 2.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:15, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:25, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<00:48, 3.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<21:56, 118kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<15:35, 166kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<10:51, 236kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<08:05, 312kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<06:10, 408kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<04:25, 567kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<03:03, 802kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<2:23:40, 17.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<1:40:34, 24.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<1:09:42, 34.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<48:38, 49.1kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<34:29, 69.1kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<24:07, 98.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<16:32, 140kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<2:25:01, 16.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<1:41:28, 22.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<1:10:17, 32.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<48:57, 46.0kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<34:40, 64.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<24:14, 92.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<16:55, 129kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<12:02, 181kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<08:22, 256kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:15, 337kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<04:44, 445kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<03:24, 615kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:23, 866kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:24, 847kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:53, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:21, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:24, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:10, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:02, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:55, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:41, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:09, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:55, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:40, 2.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:52, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:53, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:38, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:06, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:14, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:57, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:41, 2.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:56, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:06, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:52, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:37, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:05, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:10, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:53, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:38, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:02, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.79MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:35, 2.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:00, 1.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:05, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:36, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:53, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:42, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:30, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:54, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:55, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:42, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:30, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:40, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:36, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:26, 2.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:38, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:43, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:34, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:24, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:47, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:49, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:38, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:26, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:46, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:34, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:24, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:49, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:50, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:39, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:27, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:37, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:38, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:29, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:20, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:37, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:37, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:27, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:19, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:27, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:28, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:22, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:15, 2.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:35, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:35, 1.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:26, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:26, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:27, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:20, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:13, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:20, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:20, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:12, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:04, 2.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.66MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 213/400000 [00:00<03:07, 2129.23it/s]  0%|          | 925/400000 [00:00<02:28, 2695.89it/s]  0%|          | 1642/400000 [00:00<02:00, 3316.69it/s]  1%|          | 2343/400000 [00:00<01:40, 3938.54it/s]  1%|          | 3003/400000 [00:00<01:28, 4480.05it/s]  1%|          | 3644/400000 [00:00<01:20, 4923.77it/s]  1%|          | 4230/400000 [00:00<01:16, 5171.18it/s]  1%|          | 4946/400000 [00:00<01:10, 5640.93it/s]  1%|▏         | 5645/400000 [00:00<01:05, 5986.65it/s]  2%|▏         | 6350/400000 [00:01<01:02, 6269.56it/s]  2%|▏         | 7059/400000 [00:01<01:00, 6492.70it/s]  2%|▏         | 7805/400000 [00:01<00:58, 6755.30it/s]  2%|▏         | 8537/400000 [00:01<00:56, 6914.64it/s]  2%|▏         | 9252/400000 [00:01<00:55, 6982.14it/s]  2%|▏         | 9962/400000 [00:01<00:56, 6850.56it/s]  3%|▎         | 10682/400000 [00:01<00:56, 6950.33it/s]  3%|▎         | 11390/400000 [00:01<00:55, 6986.97it/s]  3%|▎         | 12104/400000 [00:01<00:55, 7029.89it/s]  3%|▎         | 12811/400000 [00:01<00:55, 7029.87it/s]  3%|▎         | 13517/400000 [00:02<00:55, 6975.81it/s]  4%|▎         | 14217/400000 [00:02<00:55, 6961.48it/s]  4%|▎         | 14922/400000 [00:02<00:55, 6984.52it/s]  4%|▍         | 15635/400000 [00:02<00:54, 7024.55it/s]  4%|▍         | 16339/400000 [00:02<00:55, 6944.37it/s]  4%|▍         | 17105/400000 [00:02<00:53, 7144.51it/s]  4%|▍         | 17831/400000 [00:02<00:53, 7175.97it/s]  5%|▍         | 18560/400000 [00:02<00:52, 7209.45it/s]  5%|▍         | 19282/400000 [00:02<00:52, 7207.31it/s]  5%|▌         | 20004/400000 [00:02<00:53, 7127.48it/s]  5%|▌         | 20718/400000 [00:03<00:55, 6881.80it/s]  5%|▌         | 21420/400000 [00:03<00:54, 6921.74it/s]  6%|▌         | 22144/400000 [00:03<00:53, 7013.76it/s]  6%|▌         | 22847/400000 [00:03<00:54, 6965.43it/s]  6%|▌         | 23545/400000 [00:03<00:54, 6933.55it/s]  6%|▌         | 24297/400000 [00:03<00:52, 7098.94it/s]  6%|▋         | 25009/400000 [00:03<00:53, 7019.21it/s]  6%|▋         | 25734/400000 [00:03<00:52, 7085.24it/s]  7%|▋         | 26459/400000 [00:03<00:52, 7131.32it/s]  7%|▋         | 27176/400000 [00:03<00:52, 7142.58it/s]  7%|▋         | 27891/400000 [00:04<00:52, 7114.20it/s]  7%|▋         | 28611/400000 [00:04<00:52, 7137.87it/s]  7%|▋         | 29326/400000 [00:04<00:52, 7101.13it/s]  8%|▊         | 30053/400000 [00:04<00:51, 7148.71it/s]  8%|▊         | 30776/400000 [00:04<00:51, 7169.49it/s]  8%|▊         | 31494/400000 [00:04<00:51, 7092.61it/s]  8%|▊         | 32246/400000 [00:04<00:50, 7213.50it/s]  8%|▊         | 33000/400000 [00:04<00:50, 7307.80it/s]  8%|▊         | 33732/400000 [00:04<00:50, 7300.55it/s]  9%|▊         | 34474/400000 [00:04<00:49, 7333.52it/s]  9%|▉         | 35208/400000 [00:05<00:51, 7131.50it/s]  9%|▉         | 35923/400000 [00:05<00:51, 7135.00it/s]  9%|▉         | 36638/400000 [00:05<00:53, 6801.68it/s]  9%|▉         | 37371/400000 [00:05<00:52, 6951.96it/s] 10%|▉         | 38070/400000 [00:05<00:52, 6915.75it/s] 10%|▉         | 38799/400000 [00:05<00:51, 7023.08it/s] 10%|▉         | 39535/400000 [00:05<00:50, 7120.00it/s] 10%|█         | 40261/400000 [00:05<00:50, 7160.10it/s] 10%|█         | 40979/400000 [00:05<00:50, 7156.75it/s] 10%|█         | 41696/400000 [00:05<00:50, 7026.19it/s] 11%|█         | 42400/400000 [00:06<00:53, 6650.19it/s] 11%|█         | 43071/400000 [00:06<00:54, 6601.83it/s] 11%|█         | 43784/400000 [00:06<00:52, 6750.69it/s] 11%|█         | 44531/400000 [00:06<00:51, 6949.41it/s] 11%|█▏        | 45231/400000 [00:06<00:50, 6962.67it/s] 11%|█▏        | 45964/400000 [00:06<00:50, 7066.69it/s] 12%|█▏        | 46673/400000 [00:06<00:50, 7053.71it/s] 12%|█▏        | 47411/400000 [00:06<00:49, 7146.36it/s] 12%|█▏        | 48161/400000 [00:06<00:48, 7247.44it/s] 12%|█▏        | 48896/400000 [00:07<00:48, 7276.62it/s] 12%|█▏        | 49625/400000 [00:07<00:48, 7245.65it/s] 13%|█▎        | 50351/400000 [00:07<00:48, 7203.01it/s] 13%|█▎        | 51072/400000 [00:07<00:49, 6994.20it/s] 13%|█▎        | 51808/400000 [00:07<00:49, 7099.54it/s] 13%|█▎        | 52520/400000 [00:07<00:49, 6958.74it/s] 13%|█▎        | 53218/400000 [00:07<00:51, 6742.38it/s] 13%|█▎        | 53953/400000 [00:07<00:50, 6911.91it/s] 14%|█▎        | 54692/400000 [00:07<00:48, 7047.24it/s] 14%|█▍        | 55429/400000 [00:07<00:48, 7139.89it/s] 14%|█▍        | 56185/400000 [00:08<00:47, 7258.52it/s] 14%|█▍        | 56926/400000 [00:08<00:46, 7302.11it/s] 14%|█▍        | 57658/400000 [00:08<00:47, 7209.53it/s] 15%|█▍        | 58381/400000 [00:08<00:47, 7124.03it/s] 15%|█▍        | 59095/400000 [00:08<00:49, 6939.19it/s] 15%|█▍        | 59791/400000 [00:08<00:49, 6845.64it/s] 15%|█▌        | 60478/400000 [00:08<00:49, 6830.17it/s] 15%|█▌        | 61166/400000 [00:08<00:49, 6845.00it/s] 15%|█▌        | 61852/400000 [00:08<00:50, 6759.72it/s] 16%|█▌        | 62529/400000 [00:08<00:52, 6450.27it/s] 16%|█▌        | 63178/400000 [00:09<00:52, 6433.65it/s] 16%|█▌        | 63857/400000 [00:09<00:51, 6536.56it/s] 16%|█▌        | 64591/400000 [00:09<00:49, 6757.16it/s] 16%|█▋        | 65324/400000 [00:09<00:48, 6918.45it/s] 17%|█▋        | 66049/400000 [00:09<00:47, 7012.33it/s] 17%|█▋        | 66753/400000 [00:09<00:47, 7007.65it/s] 17%|█▋        | 67493/400000 [00:09<00:46, 7119.94it/s] 17%|█▋        | 68210/400000 [00:09<00:46, 7133.46it/s] 17%|█▋        | 68953/400000 [00:09<00:45, 7218.19it/s] 17%|█▋        | 69678/400000 [00:09<00:45, 7225.39it/s] 18%|█▊        | 70402/400000 [00:10<00:45, 7179.65it/s] 18%|█▊        | 71124/400000 [00:10<00:45, 7190.47it/s] 18%|█▊        | 71855/400000 [00:10<00:45, 7225.85it/s] 18%|█▊        | 72578/400000 [00:10<00:46, 7080.21it/s] 18%|█▊        | 73299/400000 [00:10<00:45, 7117.41it/s] 19%|█▊        | 74012/400000 [00:10<00:46, 7075.46it/s] 19%|█▊        | 74721/400000 [00:10<00:45, 7072.40it/s] 19%|█▉        | 75429/400000 [00:10<00:46, 7000.11it/s] 19%|█▉        | 76130/400000 [00:10<00:47, 6808.77it/s] 19%|█▉        | 76813/400000 [00:11<00:47, 6754.25it/s] 19%|█▉        | 77490/400000 [00:11<00:48, 6654.68it/s] 20%|█▉        | 78157/400000 [00:11<00:49, 6535.82it/s] 20%|█▉        | 78836/400000 [00:11<00:48, 6609.99it/s] 20%|█▉        | 79499/400000 [00:11<00:48, 6551.93it/s] 20%|██        | 80218/400000 [00:11<00:47, 6730.83it/s] 20%|██        | 80893/400000 [00:11<00:47, 6699.53it/s] 20%|██        | 81588/400000 [00:11<00:47, 6771.59it/s] 21%|██        | 82311/400000 [00:11<00:46, 6902.44it/s] 21%|██        | 83041/400000 [00:11<00:45, 7017.03it/s] 21%|██        | 83788/400000 [00:12<00:44, 7144.92it/s] 21%|██        | 84524/400000 [00:12<00:43, 7204.65it/s] 21%|██▏       | 85246/400000 [00:12<00:43, 7166.28it/s] 21%|██▏       | 85964/400000 [00:12<00:44, 7045.64it/s] 22%|██▏       | 86670/400000 [00:12<00:44, 7022.82it/s] 22%|██▏       | 87384/400000 [00:12<00:44, 7055.89it/s] 22%|██▏       | 88118/400000 [00:12<00:43, 7138.62it/s] 22%|██▏       | 88866/400000 [00:12<00:42, 7237.28it/s] 22%|██▏       | 89591/400000 [00:12<00:42, 7237.01it/s] 23%|██▎       | 90319/400000 [00:12<00:42, 7249.61it/s] 23%|██▎       | 91072/400000 [00:13<00:42, 7331.07it/s] 23%|██▎       | 91806/400000 [00:13<00:42, 7263.82it/s] 23%|██▎       | 92533/400000 [00:13<00:42, 7235.81it/s] 23%|██▎       | 93257/400000 [00:13<00:42, 7204.27it/s] 23%|██▎       | 93978/400000 [00:13<00:43, 7038.79it/s] 24%|██▎       | 94683/400000 [00:13<00:44, 6889.75it/s] 24%|██▍       | 95374/400000 [00:13<00:44, 6795.33it/s] 24%|██▍       | 96099/400000 [00:13<00:43, 6921.67it/s] 24%|██▍       | 96804/400000 [00:13<00:43, 6958.75it/s] 24%|██▍       | 97529/400000 [00:13<00:42, 7042.44it/s] 25%|██▍       | 98245/400000 [00:14<00:42, 7076.22it/s] 25%|██▍       | 98964/400000 [00:14<00:42, 7108.29it/s] 25%|██▍       | 99688/400000 [00:14<00:42, 7146.28it/s] 25%|██▌       | 100432/400000 [00:14<00:41, 7230.53it/s] 25%|██▌       | 101162/400000 [00:14<00:41, 7248.92it/s] 25%|██▌       | 101888/400000 [00:14<00:41, 7201.12it/s] 26%|██▌       | 102609/400000 [00:14<00:42, 7058.72it/s] 26%|██▌       | 103316/400000 [00:14<00:42, 7017.77it/s] 26%|██▌       | 104035/400000 [00:14<00:41, 7066.49it/s] 26%|██▌       | 104798/400000 [00:14<00:40, 7225.14it/s] 26%|██▋       | 105554/400000 [00:15<00:40, 7321.61it/s] 27%|██▋       | 106314/400000 [00:15<00:39, 7401.77it/s] 27%|██▋       | 107056/400000 [00:15<00:39, 7335.20it/s] 27%|██▋       | 107822/400000 [00:15<00:39, 7428.34it/s] 27%|██▋       | 108599/400000 [00:15<00:38, 7526.64it/s] 27%|██▋       | 109362/400000 [00:15<00:38, 7555.75it/s] 28%|██▊       | 110119/400000 [00:15<00:39, 7392.35it/s] 28%|██▊       | 110860/400000 [00:15<00:39, 7356.72it/s] 28%|██▊       | 111597/400000 [00:15<00:39, 7299.83it/s] 28%|██▊       | 112328/400000 [00:16<00:40, 7022.20it/s] 28%|██▊       | 113058/400000 [00:16<00:40, 7102.13it/s] 28%|██▊       | 113771/400000 [00:16<00:40, 7041.50it/s] 29%|██▊       | 114514/400000 [00:16<00:39, 7153.55it/s] 29%|██▉       | 115235/400000 [00:16<00:39, 7169.30it/s] 29%|██▉       | 115954/400000 [00:16<00:39, 7141.62it/s] 29%|██▉       | 116687/400000 [00:16<00:39, 7196.84it/s] 29%|██▉       | 117434/400000 [00:16<00:38, 7275.87it/s] 30%|██▉       | 118220/400000 [00:16<00:37, 7440.54it/s] 30%|██▉       | 118974/400000 [00:16<00:37, 7468.35it/s] 30%|██▉       | 119722/400000 [00:17<00:38, 7356.67it/s] 30%|███       | 120460/400000 [00:17<00:37, 7362.62it/s] 30%|███       | 121198/400000 [00:17<00:38, 7162.40it/s] 30%|███       | 121916/400000 [00:17<00:39, 7063.56it/s] 31%|███       | 122667/400000 [00:17<00:38, 7190.30it/s] 31%|███       | 123420/400000 [00:17<00:37, 7285.07it/s] 31%|███       | 124165/400000 [00:17<00:37, 7332.94it/s] 31%|███       | 124900/400000 [00:17<00:37, 7304.82it/s] 31%|███▏      | 125632/400000 [00:17<00:37, 7292.85it/s] 32%|███▏      | 126378/400000 [00:17<00:37, 7340.65it/s] 32%|███▏      | 127113/400000 [00:18<00:38, 7024.57it/s] 32%|███▏      | 127819/400000 [00:18<00:40, 6664.00it/s] 32%|███▏      | 128492/400000 [00:18<00:43, 6218.21it/s] 32%|███▏      | 129212/400000 [00:18<00:41, 6483.02it/s] 32%|███▏      | 129954/400000 [00:18<00:40, 6738.25it/s] 33%|███▎      | 130702/400000 [00:18<00:38, 6943.23it/s] 33%|███▎      | 131461/400000 [00:18<00:37, 7123.23it/s] 33%|███▎      | 132211/400000 [00:18<00:37, 7230.86it/s] 33%|███▎      | 132948/400000 [00:18<00:36, 7270.86it/s] 33%|███▎      | 133733/400000 [00:18<00:35, 7434.50it/s] 34%|███▎      | 134513/400000 [00:19<00:35, 7539.69it/s] 34%|███▍      | 135270/400000 [00:19<00:36, 7228.17it/s] 34%|███▍      | 135998/400000 [00:19<00:37, 7057.67it/s] 34%|███▍      | 136708/400000 [00:19<00:37, 6987.62it/s] 34%|███▍      | 137410/400000 [00:19<00:38, 6864.49it/s] 35%|███▍      | 138150/400000 [00:19<00:37, 7015.70it/s] 35%|███▍      | 138857/400000 [00:19<00:37, 7029.86it/s] 35%|███▍      | 139581/400000 [00:19<00:36, 7090.08it/s] 35%|███▌      | 140292/400000 [00:19<00:37, 6949.43it/s] 35%|███▌      | 140989/400000 [00:20<00:38, 6789.74it/s] 35%|███▌      | 141704/400000 [00:20<00:37, 6892.96it/s] 36%|███▌      | 142398/400000 [00:20<00:37, 6906.56it/s] 36%|███▌      | 143090/400000 [00:20<00:37, 6793.04it/s] 36%|███▌      | 143819/400000 [00:20<00:36, 6933.77it/s] 36%|███▌      | 144514/400000 [00:20<00:36, 6938.33it/s] 36%|███▋      | 145266/400000 [00:20<00:35, 7101.56it/s] 36%|███▋      | 145984/400000 [00:20<00:35, 7124.04it/s] 37%|███▋      | 146716/400000 [00:20<00:35, 7179.69it/s] 37%|███▋      | 147463/400000 [00:20<00:34, 7262.63it/s] 37%|███▋      | 148201/400000 [00:21<00:34, 7294.72it/s] 37%|███▋      | 148972/400000 [00:21<00:33, 7412.63it/s] 37%|███▋      | 149715/400000 [00:21<00:34, 7336.58it/s] 38%|███▊      | 150457/400000 [00:21<00:33, 7359.92it/s] 38%|███▊      | 151195/400000 [00:21<00:33, 7365.45it/s] 38%|███▊      | 151932/400000 [00:21<00:34, 7226.08it/s] 38%|███▊      | 152656/400000 [00:21<00:35, 6912.37it/s] 38%|███▊      | 153388/400000 [00:21<00:35, 7027.93it/s] 39%|███▊      | 154096/400000 [00:21<00:34, 7041.96it/s] 39%|███▊      | 154825/400000 [00:21<00:34, 7112.94it/s] 39%|███▉      | 155564/400000 [00:22<00:33, 7191.40it/s] 39%|███▉      | 156303/400000 [00:22<00:33, 7248.40it/s] 39%|███▉      | 157029/400000 [00:22<00:33, 7219.45it/s] 39%|███▉      | 157758/400000 [00:22<00:33, 7238.79it/s] 40%|███▉      | 158483/400000 [00:22<00:33, 7193.77it/s] 40%|███▉      | 159214/400000 [00:22<00:33, 7227.91it/s] 40%|███▉      | 159938/400000 [00:22<00:33, 7167.45it/s] 40%|████      | 160673/400000 [00:22<00:33, 7220.09it/s] 40%|████      | 161420/400000 [00:22<00:32, 7292.22it/s] 41%|████      | 162168/400000 [00:22<00:32, 7347.45it/s] 41%|████      | 162923/400000 [00:23<00:32, 7406.09it/s] 41%|████      | 163686/400000 [00:23<00:31, 7469.03it/s] 41%|████      | 164434/400000 [00:23<00:32, 7270.88it/s] 41%|████▏     | 165167/400000 [00:23<00:32, 7288.26it/s] 41%|████▏     | 165897/400000 [00:23<00:33, 7058.35it/s] 42%|████▏     | 166613/400000 [00:23<00:32, 7087.81it/s] 42%|████▏     | 167324/400000 [00:23<00:32, 7085.86it/s] 42%|████▏     | 168034/400000 [00:23<00:33, 7003.19it/s] 42%|████▏     | 168750/400000 [00:23<00:32, 7046.54it/s] 42%|████▏     | 169457/400000 [00:24<00:32, 7051.43it/s] 43%|████▎     | 170198/400000 [00:24<00:32, 7153.69it/s] 43%|████▎     | 170974/400000 [00:24<00:31, 7324.49it/s] 43%|████▎     | 171708/400000 [00:24<00:31, 7237.97it/s] 43%|████▎     | 172434/400000 [00:24<00:31, 7210.94it/s] 43%|████▎     | 173164/400000 [00:24<00:31, 7236.80it/s] 43%|████▎     | 173908/400000 [00:24<00:30, 7295.93it/s] 44%|████▎     | 174639/400000 [00:24<00:31, 7063.44it/s] 44%|████▍     | 175352/400000 [00:24<00:31, 7081.01it/s] 44%|████▍     | 176082/400000 [00:24<00:31, 7144.42it/s] 44%|████▍     | 176832/400000 [00:25<00:30, 7246.43it/s] 44%|████▍     | 177601/400000 [00:25<00:30, 7373.65it/s] 45%|████▍     | 178376/400000 [00:25<00:29, 7482.05it/s] 45%|████▍     | 179126/400000 [00:25<00:29, 7463.47it/s] 45%|████▍     | 179876/400000 [00:25<00:29, 7473.66it/s] 45%|████▌     | 180625/400000 [00:25<00:29, 7462.01it/s] 45%|████▌     | 181372/400000 [00:25<00:29, 7398.20it/s] 46%|████▌     | 182113/400000 [00:25<00:29, 7340.55it/s] 46%|████▌     | 182848/400000 [00:25<00:29, 7287.59it/s] 46%|████▌     | 183587/400000 [00:25<00:29, 7316.19it/s] 46%|████▌     | 184319/400000 [00:26<00:29, 7259.99it/s] 46%|████▋     | 185057/400000 [00:26<00:29, 7293.22it/s] 46%|████▋     | 185787/400000 [00:26<00:29, 7189.74it/s] 47%|████▋     | 186507/400000 [00:26<00:29, 7155.88it/s] 47%|████▋     | 187223/400000 [00:26<00:30, 6899.60it/s] 47%|████▋     | 187969/400000 [00:26<00:30, 7056.92it/s] 47%|████▋     | 188694/400000 [00:26<00:29, 7113.67it/s] 47%|████▋     | 189427/400000 [00:26<00:29, 7176.67it/s] 48%|████▊     | 190147/400000 [00:26<00:29, 7073.44it/s] 48%|████▊     | 190866/400000 [00:26<00:29, 7105.75it/s] 48%|████▊     | 191608/400000 [00:27<00:28, 7196.84it/s] 48%|████▊     | 192358/400000 [00:27<00:28, 7284.23it/s] 48%|████▊     | 193100/400000 [00:27<00:28, 7323.48it/s] 48%|████▊     | 193853/400000 [00:27<00:27, 7381.51it/s] 49%|████▊     | 194599/400000 [00:27<00:27, 7403.29it/s] 49%|████▉     | 195383/400000 [00:27<00:27, 7527.00it/s] 49%|████▉     | 196173/400000 [00:27<00:26, 7632.68it/s] 49%|████▉     | 196938/400000 [00:27<00:26, 7635.39it/s] 49%|████▉     | 197703/400000 [00:27<00:27, 7287.47it/s] 50%|████▉     | 198436/400000 [00:27<00:28, 7113.92it/s] 50%|████▉     | 199151/400000 [00:28<00:29, 6793.91it/s] 50%|████▉     | 199847/400000 [00:28<00:29, 6839.52it/s] 50%|█████     | 200583/400000 [00:28<00:28, 6986.12it/s] 50%|█████     | 201286/400000 [00:28<00:28, 6995.09it/s] 51%|█████     | 202034/400000 [00:28<00:27, 7131.92it/s] 51%|█████     | 202785/400000 [00:28<00:27, 7240.58it/s] 51%|█████     | 203533/400000 [00:28<00:26, 7309.25it/s] 51%|█████     | 204290/400000 [00:28<00:26, 7383.91it/s] 51%|█████▏    | 205030/400000 [00:28<00:26, 7281.36it/s] 51%|█████▏    | 205796/400000 [00:29<00:26, 7390.08it/s] 52%|█████▏    | 206594/400000 [00:29<00:25, 7557.52it/s] 52%|█████▏    | 207352/400000 [00:29<00:25, 7541.16it/s] 52%|█████▏    | 208108/400000 [00:29<00:25, 7514.95it/s] 52%|█████▏    | 208861/400000 [00:29<00:25, 7438.28it/s] 52%|█████▏    | 209622/400000 [00:29<00:25, 7485.64it/s] 53%|█████▎    | 210372/400000 [00:29<00:25, 7439.23it/s] 53%|█████▎    | 211135/400000 [00:29<00:25, 7493.21it/s] 53%|█████▎    | 211907/400000 [00:29<00:24, 7557.11it/s] 53%|█████▎    | 212664/400000 [00:29<00:24, 7518.85it/s] 53%|█████▎    | 213425/400000 [00:30<00:24, 7543.53it/s] 54%|█████▎    | 214180/400000 [00:30<00:24, 7452.48it/s] 54%|█████▎    | 214968/400000 [00:30<00:24, 7574.69it/s] 54%|█████▍    | 215757/400000 [00:30<00:24, 7664.58it/s] 54%|█████▍    | 216525/400000 [00:30<00:24, 7386.93it/s] 54%|█████▍    | 217267/400000 [00:30<00:26, 6986.54it/s] 54%|█████▍    | 217973/400000 [00:30<00:26, 6990.67it/s] 55%|█████▍    | 218677/400000 [00:30<00:26, 6907.97it/s] 55%|█████▍    | 219387/400000 [00:30<00:25, 6963.02it/s] 55%|█████▌    | 220086/400000 [00:30<00:26, 6914.93it/s] 55%|█████▌    | 220819/400000 [00:31<00:25, 7033.58it/s] 55%|█████▌    | 221569/400000 [00:31<00:24, 7166.95it/s] 56%|█████▌    | 222290/400000 [00:31<00:24, 7176.11it/s] 56%|█████▌    | 223015/400000 [00:31<00:24, 7194.52it/s] 56%|█████▌    | 223771/400000 [00:31<00:24, 7297.10it/s] 56%|█████▌    | 224502/400000 [00:31<00:24, 7271.04it/s] 56%|█████▋    | 225230/400000 [00:31<00:25, 6982.26it/s] 56%|█████▋    | 225932/400000 [00:31<00:25, 6940.93it/s] 57%|█████▋    | 226631/400000 [00:31<00:24, 6955.54it/s] 57%|█████▋    | 227329/400000 [00:31<00:25, 6847.70it/s] 57%|█████▋    | 228065/400000 [00:32<00:24, 6991.73it/s] 57%|█████▋    | 228766/400000 [00:32<00:24, 6899.42it/s] 57%|█████▋    | 229487/400000 [00:32<00:24, 6988.67it/s] 58%|█████▊    | 230189/400000 [00:32<00:24, 6994.69it/s] 58%|█████▊    | 230917/400000 [00:32<00:23, 7075.55it/s] 58%|█████▊    | 231626/400000 [00:32<00:24, 6990.68it/s] 58%|█████▊    | 232326/400000 [00:32<00:24, 6807.77it/s] 58%|█████▊    | 233071/400000 [00:32<00:23, 6984.89it/s] 58%|█████▊    | 233801/400000 [00:32<00:23, 7075.63it/s] 59%|█████▊    | 234511/400000 [00:33<00:23, 7039.35it/s] 59%|█████▉    | 235242/400000 [00:33<00:23, 7116.73it/s] 59%|█████▉    | 235955/400000 [00:33<00:23, 7059.30it/s] 59%|█████▉    | 236675/400000 [00:33<00:23, 7098.25it/s] 59%|█████▉    | 237394/400000 [00:33<00:22, 7124.27it/s] 60%|█████▉    | 238140/400000 [00:33<00:22, 7221.05it/s] 60%|█████▉    | 238903/400000 [00:33<00:21, 7338.03it/s] 60%|█████▉    | 239676/400000 [00:33<00:21, 7451.18it/s] 60%|██████    | 240426/400000 [00:33<00:21, 7464.81it/s] 60%|██████    | 241174/400000 [00:33<00:21, 7448.15it/s] 60%|██████    | 241920/400000 [00:34<00:21, 7413.37it/s] 61%|██████    | 242666/400000 [00:34<00:21, 7424.74it/s] 61%|██████    | 243409/400000 [00:34<00:21, 7419.81it/s] 61%|██████    | 244152/400000 [00:34<00:21, 7318.65it/s] 61%|██████    | 244900/400000 [00:34<00:21, 7364.53it/s] 61%|██████▏   | 245637/400000 [00:34<00:21, 7305.16it/s] 62%|██████▏   | 246416/400000 [00:34<00:20, 7442.68it/s] 62%|██████▏   | 247190/400000 [00:34<00:20, 7529.12it/s] 62%|██████▏   | 247946/400000 [00:34<00:20, 7537.07it/s] 62%|██████▏   | 248701/400000 [00:34<00:20, 7450.10it/s] 62%|██████▏   | 249447/400000 [00:35<00:21, 7087.49it/s] 63%|██████▎   | 250160/400000 [00:35<00:21, 7057.12it/s] 63%|██████▎   | 250869/400000 [00:35<00:21, 7022.24it/s] 63%|██████▎   | 251574/400000 [00:35<00:21, 7024.53it/s] 63%|██████▎   | 252278/400000 [00:35<00:21, 7001.04it/s] 63%|██████▎   | 252980/400000 [00:35<00:21, 6894.73it/s] 63%|██████▎   | 253741/400000 [00:35<00:20, 7094.14it/s] 64%|██████▎   | 254514/400000 [00:35<00:20, 7271.62it/s] 64%|██████▍   | 255295/400000 [00:35<00:19, 7422.87it/s] 64%|██████▍   | 256040/400000 [00:35<00:19, 7381.95it/s] 64%|██████▍   | 256781/400000 [00:36<00:19, 7264.20it/s] 64%|██████▍   | 257510/400000 [00:36<00:19, 7215.28it/s] 65%|██████▍   | 258233/400000 [00:36<00:20, 6777.75it/s] 65%|██████▍   | 258918/400000 [00:36<00:20, 6763.61it/s] 65%|██████▍   | 259599/400000 [00:36<00:20, 6721.88it/s] 65%|██████▌   | 260275/400000 [00:36<00:20, 6727.75it/s] 65%|██████▌   | 261022/400000 [00:36<00:20, 6934.17it/s] 65%|██████▌   | 261719/400000 [00:36<00:20, 6702.09it/s] 66%|██████▌   | 262455/400000 [00:36<00:19, 6885.09it/s] 66%|██████▌   | 263181/400000 [00:37<00:19, 6990.88it/s] 66%|██████▌   | 263912/400000 [00:37<00:19, 7083.06it/s] 66%|██████▌   | 264623/400000 [00:37<00:19, 6972.63it/s] 66%|██████▋   | 265323/400000 [00:37<00:19, 6904.91it/s] 67%|██████▋   | 266016/400000 [00:37<00:20, 6617.37it/s] 67%|██████▋   | 266682/400000 [00:37<00:20, 6566.25it/s] 67%|██████▋   | 267392/400000 [00:37<00:19, 6716.88it/s] 67%|██████▋   | 268086/400000 [00:37<00:19, 6781.42it/s] 67%|██████▋   | 268773/400000 [00:37<00:19, 6807.56it/s] 67%|██████▋   | 269456/400000 [00:37<00:19, 6692.81it/s] 68%|██████▊   | 270144/400000 [00:38<00:19, 6746.36it/s] 68%|██████▊   | 270858/400000 [00:38<00:18, 6853.43it/s] 68%|██████▊   | 271607/400000 [00:38<00:18, 7030.36it/s] 68%|██████▊   | 272349/400000 [00:38<00:17, 7141.39it/s] 68%|██████▊   | 273078/400000 [00:38<00:17, 7184.83it/s] 68%|██████▊   | 273798/400000 [00:38<00:17, 7150.51it/s] 69%|██████▊   | 274515/400000 [00:38<00:17, 7112.15it/s] 69%|██████▉   | 275267/400000 [00:38<00:17, 7227.04it/s] 69%|██████▉   | 276003/400000 [00:38<00:17, 7264.75it/s] 69%|██████▉   | 276731/400000 [00:38<00:17, 7250.93it/s] 69%|██████▉   | 277469/400000 [00:39<00:16, 7286.65it/s] 70%|██████▉   | 278199/400000 [00:39<00:16, 7272.64it/s] 70%|██████▉   | 278933/400000 [00:39<00:16, 7290.71it/s] 70%|██████▉   | 279691/400000 [00:39<00:16, 7375.11it/s] 70%|███████   | 280451/400000 [00:39<00:16, 7439.82it/s] 70%|███████   | 281196/400000 [00:39<00:15, 7438.28it/s] 70%|███████   | 281967/400000 [00:39<00:15, 7517.12it/s] 71%|███████   | 282720/400000 [00:39<00:15, 7495.75it/s] 71%|███████   | 283470/400000 [00:39<00:15, 7484.43it/s] 71%|███████   | 284219/400000 [00:39<00:15, 7454.06it/s] 71%|███████   | 284965/400000 [00:40<00:15, 7237.94it/s] 71%|███████▏  | 285692/400000 [00:40<00:15, 7245.85it/s] 72%|███████▏  | 286431/400000 [00:40<00:15, 7285.87it/s] 72%|███████▏  | 287161/400000 [00:40<00:15, 7245.69it/s] 72%|███████▏  | 287894/400000 [00:40<00:15, 7269.17it/s] 72%|███████▏  | 288629/400000 [00:40<00:15, 7292.41it/s] 72%|███████▏  | 289359/400000 [00:40<00:15, 7278.25it/s] 73%|███████▎  | 290102/400000 [00:40<00:15, 7320.57it/s] 73%|███████▎  | 290847/400000 [00:40<00:14, 7357.75it/s] 73%|███████▎  | 291583/400000 [00:40<00:14, 7297.68it/s] 73%|███████▎  | 292316/400000 [00:41<00:14, 7304.73it/s] 73%|███████▎  | 293065/400000 [00:41<00:14, 7358.43it/s] 73%|███████▎  | 293802/400000 [00:41<00:14, 7173.67it/s] 74%|███████▎  | 294521/400000 [00:41<00:14, 7121.69it/s] 74%|███████▍  | 295237/400000 [00:41<00:14, 7131.43it/s] 74%|███████▍  | 295951/400000 [00:41<00:14, 7108.39it/s] 74%|███████▍  | 296663/400000 [00:41<00:14, 6986.54it/s] 74%|███████▍  | 297363/400000 [00:41<00:14, 6982.94it/s] 75%|███████▍  | 298085/400000 [00:41<00:14, 7050.47it/s] 75%|███████▍  | 298823/400000 [00:41<00:14, 7145.33it/s] 75%|███████▍  | 299570/400000 [00:42<00:13, 7237.87it/s] 75%|███████▌  | 300295/400000 [00:42<00:13, 7236.19it/s] 75%|███████▌  | 301044/400000 [00:42<00:13, 7309.10it/s] 75%|███████▌  | 301817/400000 [00:42<00:13, 7428.67it/s] 76%|███████▌  | 302561/400000 [00:42<00:13, 7180.17it/s] 76%|███████▌  | 303282/400000 [00:42<00:13, 7188.10it/s] 76%|███████▌  | 304003/400000 [00:42<00:13, 7115.77it/s] 76%|███████▌  | 304716/400000 [00:42<00:13, 6975.17it/s] 76%|███████▋  | 305416/400000 [00:42<00:13, 6910.30it/s] 77%|███████▋  | 306119/400000 [00:43<00:13, 6944.82it/s] 77%|███████▋  | 306833/400000 [00:43<00:13, 6998.63it/s] 77%|███████▋  | 307563/400000 [00:43<00:13, 7084.53it/s] 77%|███████▋  | 308303/400000 [00:43<00:12, 7176.10it/s] 77%|███████▋  | 309090/400000 [00:43<00:12, 7370.90it/s] 77%|███████▋  | 309856/400000 [00:43<00:12, 7454.19it/s] 78%|███████▊  | 310603/400000 [00:43<00:12, 7351.22it/s] 78%|███████▊  | 311340/400000 [00:43<00:12, 7078.50it/s] 78%|███████▊  | 312058/400000 [00:43<00:12, 7105.91it/s] 78%|███████▊  | 312779/400000 [00:43<00:12, 7136.82it/s] 78%|███████▊  | 313510/400000 [00:44<00:12, 7185.61it/s] 79%|███████▊  | 314239/400000 [00:44<00:11, 7216.54it/s] 79%|███████▊  | 314978/400000 [00:44<00:11, 7266.88it/s] 79%|███████▉  | 315706/400000 [00:44<00:11, 7251.89it/s] 79%|███████▉  | 316432/400000 [00:44<00:11, 7175.73it/s] 79%|███████▉  | 317153/400000 [00:44<00:11, 7185.16it/s] 79%|███████▉  | 317872/400000 [00:44<00:11, 7142.97it/s] 80%|███████▉  | 318598/400000 [00:44<00:11, 7176.63it/s] 80%|███████▉  | 319316/400000 [00:44<00:11, 7153.62it/s] 80%|████████  | 320042/400000 [00:44<00:11, 7182.19it/s] 80%|████████  | 320782/400000 [00:45<00:10, 7245.83it/s] 80%|████████  | 321507/400000 [00:45<00:10, 7211.98it/s] 81%|████████  | 322235/400000 [00:45<00:10, 7231.04it/s] 81%|████████  | 322959/400000 [00:45<00:10, 7095.63it/s] 81%|████████  | 323683/400000 [00:45<00:10, 7136.41it/s] 81%|████████  | 324398/400000 [00:45<00:10, 7085.58it/s] 81%|████████▏ | 325108/400000 [00:45<00:10, 7045.15it/s] 81%|████████▏ | 325813/400000 [00:45<00:10, 7036.07it/s] 82%|████████▏ | 326517/400000 [00:45<00:10, 6911.61it/s] 82%|████████▏ | 327209/400000 [00:45<00:10, 6786.94it/s] 82%|████████▏ | 327911/400000 [00:46<00:10, 6852.37it/s] 82%|████████▏ | 328598/400000 [00:46<00:10, 6762.19it/s] 82%|████████▏ | 329289/400000 [00:46<00:10, 6805.22it/s] 82%|████████▏ | 329972/400000 [00:46<00:10, 6812.63it/s] 83%|████████▎ | 330654/400000 [00:46<00:10, 6785.39it/s] 83%|████████▎ | 331387/400000 [00:46<00:09, 6939.92it/s] 83%|████████▎ | 332112/400000 [00:46<00:09, 7027.07it/s] 83%|████████▎ | 332816/400000 [00:46<00:09, 6969.58it/s] 83%|████████▎ | 333514/400000 [00:46<00:09, 6881.66it/s] 84%|████████▎ | 334204/400000 [00:47<00:10, 6419.61it/s] 84%|████████▎ | 334853/400000 [00:47<00:10, 6020.20it/s] 84%|████████▍ | 335480/400000 [00:47<00:10, 6090.94it/s] 84%|████████▍ | 336169/400000 [00:47<00:10, 6308.99it/s] 84%|████████▍ | 336840/400000 [00:47<00:09, 6422.30it/s] 84%|████████▍ | 337488/400000 [00:47<00:10, 6148.63it/s] 85%|████████▍ | 338188/400000 [00:47<00:09, 6379.74it/s] 85%|████████▍ | 338893/400000 [00:47<00:09, 6566.37it/s] 85%|████████▍ | 339585/400000 [00:47<00:09, 6667.72it/s] 85%|████████▌ | 340276/400000 [00:47<00:08, 6738.49it/s] 85%|████████▌ | 340976/400000 [00:48<00:08, 6813.47it/s] 85%|████████▌ | 341660/400000 [00:48<00:08, 6799.56it/s] 86%|████████▌ | 342342/400000 [00:48<00:08, 6599.45it/s] 86%|████████▌ | 343005/400000 [00:48<00:08, 6562.58it/s] 86%|████████▌ | 343664/400000 [00:48<00:08, 6552.14it/s] 86%|████████▌ | 344321/400000 [00:48<00:08, 6473.64it/s] 86%|████████▋ | 345057/400000 [00:48<00:08, 6716.05it/s] 86%|████████▋ | 345789/400000 [00:48<00:07, 6884.00it/s] 87%|████████▋ | 346481/400000 [00:48<00:07, 6849.29it/s] 87%|████████▋ | 347169/400000 [00:48<00:07, 6779.82it/s] 87%|████████▋ | 347849/400000 [00:49<00:07, 6753.95it/s] 87%|████████▋ | 348526/400000 [00:49<00:07, 6585.30it/s] 87%|████████▋ | 349187/400000 [00:49<00:07, 6523.11it/s] 87%|████████▋ | 349841/400000 [00:49<00:07, 6474.74it/s] 88%|████████▊ | 350574/400000 [00:49<00:07, 6708.01it/s] 88%|████████▊ | 351277/400000 [00:49<00:07, 6801.35it/s] 88%|████████▊ | 351999/400000 [00:49<00:06, 6920.15it/s] 88%|████████▊ | 352694/400000 [00:49<00:06, 6799.47it/s] 88%|████████▊ | 353382/400000 [00:49<00:06, 6822.59it/s] 89%|████████▊ | 354066/400000 [00:50<00:06, 6669.28it/s] 89%|████████▊ | 354775/400000 [00:50<00:06, 6788.34it/s] 89%|████████▉ | 355456/400000 [00:50<00:06, 6625.47it/s] 89%|████████▉ | 356121/400000 [00:50<00:06, 6591.59it/s] 89%|████████▉ | 356820/400000 [00:50<00:06, 6705.64it/s] 89%|████████▉ | 357511/400000 [00:50<00:06, 6763.53it/s] 90%|████████▉ | 358251/400000 [00:50<00:06, 6941.04it/s] 90%|████████▉ | 359000/400000 [00:50<00:05, 7095.13it/s] 90%|████████▉ | 359712/400000 [00:50<00:05, 7030.60it/s] 90%|█████████ | 360436/400000 [00:50<00:05, 7090.91it/s] 90%|█████████ | 361196/400000 [00:51<00:05, 7235.36it/s] 90%|█████████ | 361974/400000 [00:51<00:05, 7389.02it/s] 91%|█████████ | 362758/400000 [00:51<00:04, 7516.79it/s] 91%|█████████ | 363512/400000 [00:51<00:04, 7336.59it/s] 91%|█████████ | 364248/400000 [00:51<00:05, 6709.10it/s] 91%|█████████ | 364931/400000 [00:51<00:05, 6643.58it/s] 91%|█████████▏| 365683/400000 [00:51<00:04, 6883.86it/s] 92%|█████████▏| 366380/400000 [00:51<00:05, 6676.99it/s] 92%|█████████▏| 367073/400000 [00:51<00:04, 6749.26it/s] 92%|█████████▏| 367754/400000 [00:51<00:05, 6375.30it/s] 92%|█████████▏| 368400/400000 [00:52<00:04, 6396.49it/s] 92%|█████████▏| 369107/400000 [00:52<00:04, 6583.79it/s] 92%|█████████▏| 369804/400000 [00:52<00:04, 6694.73it/s] 93%|█████████▎| 370543/400000 [00:52<00:04, 6886.79it/s] 93%|█████████▎| 371322/400000 [00:52<00:04, 7133.79it/s] 93%|█████████▎| 372071/400000 [00:52<00:03, 7236.66it/s] 93%|█████████▎| 372804/400000 [00:52<00:03, 7263.61it/s] 93%|█████████▎| 373534/400000 [00:52<00:03, 7253.53it/s] 94%|█████████▎| 374262/400000 [00:52<00:03, 7223.62it/s] 94%|█████████▍| 375003/400000 [00:53<00:03, 7278.25it/s] 94%|█████████▍| 375744/400000 [00:53<00:03, 7314.64it/s] 94%|█████████▍| 376477/400000 [00:53<00:03, 7231.53it/s] 94%|█████████▍| 377201/400000 [00:53<00:03, 7166.04it/s] 94%|█████████▍| 377942/400000 [00:53<00:03, 7237.04it/s] 95%|█████████▍| 378690/400000 [00:53<00:02, 7308.00it/s] 95%|█████████▍| 379433/400000 [00:53<00:02, 7342.56it/s] 95%|█████████▌| 380168/400000 [00:53<00:02, 7207.79it/s] 95%|█████████▌| 380890/400000 [00:53<00:02, 7163.41it/s] 95%|█████████▌| 381634/400000 [00:53<00:02, 7243.75it/s] 96%|█████████▌| 382372/400000 [00:54<00:02, 7282.43it/s] 96%|█████████▌| 383109/400000 [00:54<00:02, 7307.16it/s] 96%|█████████▌| 383841/400000 [00:54<00:02, 7304.99it/s] 96%|█████████▌| 384572/400000 [00:54<00:02, 7244.72it/s] 96%|█████████▋| 385297/400000 [00:54<00:02, 7213.19it/s] 97%|█████████▋| 386038/400000 [00:54<00:01, 7266.98it/s] 97%|█████████▋| 386775/400000 [00:54<00:01, 7296.39it/s] 97%|█████████▋| 387551/400000 [00:54<00:01, 7428.50it/s] 97%|█████████▋| 388295/400000 [00:54<00:01, 7382.25it/s] 97%|█████████▋| 389034/400000 [00:54<00:01, 7250.09it/s] 97%|█████████▋| 389760/400000 [00:55<00:01, 7168.86it/s] 98%|█████████▊| 390488/400000 [00:55<00:01, 7199.87it/s] 98%|█████████▊| 391209/400000 [00:55<00:01, 7201.79it/s] 98%|█████████▊| 391930/400000 [00:55<00:01, 7038.17it/s] 98%|█████████▊| 392635/400000 [00:55<00:01, 6896.63it/s] 98%|█████████▊| 393336/400000 [00:55<00:00, 6928.87it/s] 99%|█████████▊| 394082/400000 [00:55<00:00, 7078.03it/s] 99%|█████████▊| 394833/400000 [00:55<00:00, 7201.80it/s] 99%|█████████▉| 395578/400000 [00:55<00:00, 7273.05it/s] 99%|█████████▉| 396318/400000 [00:55<00:00, 7308.13it/s] 99%|█████████▉| 397050/400000 [00:56<00:00, 7296.56it/s] 99%|█████████▉| 397781/400000 [00:56<00:00, 7176.55it/s]100%|█████████▉| 398500/400000 [00:56<00:00, 6936.46it/s]100%|█████████▉| 399197/400000 [00:56<00:00, 6915.95it/s]100%|█████████▉| 399913/400000 [00:56<00:00, 6986.25it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7081.79it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdf640460f0>, <torchtext.data.dataset.TabularDataset object at 0x7fdf64046240>, <torchtext.vocab.Vocab object at 0x7fdf64046160>), {}) 

