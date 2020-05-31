
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f47d5c39378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f47d5c39378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f4847c880d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f4847c880d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f47df41d9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f47df41d9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f47f556c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f47f556c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f47f556c8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3547136/9912422 [00:00<00:00, 34281595.33it/s]9920512it [00:00, 33031862.93it/s]                             
0it [00:00, ?it/s]32768it [00:00, 328162.27it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154035.68it/s]1654784it [00:00, 10677883.75it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 76639.22it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f47d7f83ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f47d7f83fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f47f556c510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f47f556c510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f47f556c510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:17,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:17,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:12, 10.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 14.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 28.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 34.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 34.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 40.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 40.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:01, 45.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 49.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 49.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 53.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 56.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 59.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 60.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 62.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 63.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 62.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.92 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 62.92 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.90 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ url]
0 examples [00:00, ? examples/s]2020-05-31 00:10:00.167338: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-31 00:10:00.191352: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-31 00:10:00.191577: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a21a3d4d70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-31 00:10:00.191600: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
29 examples [00:00, 289.92 examples/s]120 examples [00:00, 364.10 examples/s]213 examples [00:00, 445.00 examples/s]303 examples [00:00, 523.88 examples/s]390 examples [00:00, 594.35 examples/s]474 examples [00:00, 648.50 examples/s]556 examples [00:00, 691.26 examples/s]647 examples [00:00, 742.96 examples/s]737 examples [00:00, 783.69 examples/s]823 examples [00:01, 805.07 examples/s]907 examples [00:01, 812.69 examples/s]994 examples [00:01, 827.21 examples/s]1080 examples [00:01, 836.35 examples/s]1166 examples [00:01, 840.80 examples/s]1258 examples [00:01, 862.70 examples/s]1348 examples [00:01, 873.20 examples/s]1439 examples [00:01, 881.28 examples/s]1533 examples [00:01, 897.64 examples/s]1627 examples [00:01, 909.85 examples/s]1721 examples [00:02, 917.16 examples/s]1814 examples [00:02, 920.45 examples/s]1907 examples [00:02, 873.88 examples/s]1996 examples [00:02, 877.57 examples/s]2089 examples [00:02, 891.49 examples/s]2179 examples [00:02, 813.97 examples/s]2262 examples [00:02, 815.50 examples/s]2345 examples [00:02, 785.78 examples/s]2428 examples [00:02, 798.51 examples/s]2511 examples [00:02, 804.94 examples/s]2593 examples [00:03, 806.40 examples/s]2682 examples [00:03, 829.54 examples/s]2775 examples [00:03, 854.97 examples/s]2871 examples [00:03, 880.08 examples/s]2967 examples [00:03, 900.80 examples/s]3061 examples [00:03, 910.51 examples/s]3156 examples [00:03, 921.69 examples/s]3249 examples [00:03, 905.32 examples/s]3340 examples [00:03, 875.84 examples/s]3429 examples [00:04, 880.03 examples/s]3519 examples [00:04, 884.01 examples/s]3611 examples [00:04, 892.73 examples/s]3704 examples [00:04, 901.16 examples/s]3796 examples [00:04, 906.28 examples/s]3889 examples [00:04, 911.16 examples/s]3981 examples [00:04, 908.35 examples/s]4077 examples [00:04, 921.02 examples/s]4171 examples [00:04, 926.49 examples/s]4264 examples [00:04, 924.47 examples/s]4357 examples [00:05, 913.67 examples/s]4449 examples [00:05, 895.31 examples/s]4539 examples [00:05, 888.49 examples/s]4630 examples [00:05, 893.59 examples/s]4720 examples [00:05, 859.70 examples/s]4808 examples [00:05, 864.01 examples/s]4895 examples [00:05, 853.62 examples/s]4986 examples [00:05, 868.83 examples/s]5074 examples [00:05, 808.82 examples/s]5166 examples [00:05, 838.41 examples/s]5256 examples [00:06, 855.08 examples/s]5349 examples [00:06, 875.93 examples/s]5438 examples [00:06, 879.70 examples/s]5527 examples [00:06, 880.81 examples/s]5619 examples [00:06, 890.57 examples/s]5712 examples [00:06, 901.82 examples/s]5804 examples [00:06, 905.62 examples/s]5895 examples [00:06, 903.87 examples/s]5986 examples [00:06, 897.33 examples/s]6076 examples [00:06, 856.24 examples/s]6166 examples [00:07, 867.34 examples/s]6256 examples [00:07, 876.76 examples/s]6350 examples [00:07, 893.53 examples/s]6442 examples [00:07, 900.03 examples/s]6533 examples [00:07, 890.63 examples/s]6623 examples [00:07, 850.10 examples/s]6715 examples [00:07, 867.65 examples/s]6807 examples [00:07, 881.33 examples/s]6896 examples [00:07, 850.91 examples/s]6989 examples [00:08, 872.34 examples/s]7080 examples [00:08, 880.99 examples/s]7175 examples [00:08, 898.54 examples/s]7269 examples [00:08, 909.53 examples/s]7364 examples [00:08, 919.65 examples/s]7457 examples [00:08, 919.57 examples/s]7550 examples [00:08, 911.73 examples/s]7644 examples [00:08, 918.76 examples/s]7738 examples [00:08, 924.23 examples/s]7831 examples [00:08, 901.69 examples/s]7925 examples [00:09, 912.81 examples/s]8017 examples [00:09, 909.19 examples/s]8109 examples [00:09, 893.76 examples/s]8204 examples [00:09, 909.34 examples/s]8299 examples [00:09, 921.07 examples/s]8395 examples [00:09, 930.26 examples/s]8489 examples [00:09, 927.93 examples/s]8583 examples [00:09, 931.45 examples/s]8677 examples [00:09, 924.08 examples/s]8770 examples [00:09, 906.92 examples/s]8864 examples [00:10, 914.22 examples/s]8956 examples [00:10, 913.27 examples/s]9048 examples [00:10, 912.26 examples/s]9140 examples [00:10, 914.20 examples/s]9232 examples [00:10, 893.44 examples/s]9322 examples [00:10, 889.27 examples/s]9415 examples [00:10, 899.91 examples/s]9506 examples [00:10, 902.56 examples/s]9597 examples [00:10, 860.88 examples/s]9689 examples [00:11, 877.50 examples/s]9781 examples [00:11, 888.69 examples/s]9871 examples [00:11, 864.44 examples/s]9958 examples [00:11, 862.94 examples/s]10045 examples [00:11, 822.67 examples/s]10136 examples [00:11, 846.82 examples/s]10227 examples [00:11, 864.54 examples/s]10319 examples [00:11, 879.40 examples/s]10410 examples [00:11, 886.24 examples/s]10499 examples [00:11, 882.91 examples/s]10588 examples [00:12, 861.33 examples/s]10675 examples [00:12, 856.22 examples/s]10761 examples [00:12, 849.82 examples/s]10847 examples [00:12, 831.44 examples/s]10931 examples [00:12, 832.32 examples/s]11025 examples [00:12, 861.56 examples/s]11120 examples [00:12, 885.02 examples/s]11214 examples [00:12, 898.84 examples/s]11310 examples [00:12, 914.12 examples/s]11403 examples [00:12, 917.89 examples/s]11496 examples [00:13, 892.91 examples/s]11588 examples [00:13, 899.27 examples/s]11679 examples [00:13, 892.18 examples/s]11770 examples [00:13, 896.31 examples/s]11861 examples [00:13, 899.88 examples/s]11955 examples [00:13, 910.41 examples/s]12049 examples [00:13, 916.49 examples/s]12141 examples [00:13, 917.21 examples/s]12234 examples [00:13, 918.19 examples/s]12328 examples [00:13, 924.48 examples/s]12421 examples [00:14, 899.07 examples/s]12512 examples [00:14, 851.08 examples/s]12598 examples [00:14, 846.96 examples/s]12684 examples [00:14, 844.78 examples/s]12777 examples [00:14, 867.79 examples/s]12870 examples [00:14, 885.18 examples/s]12964 examples [00:14, 899.47 examples/s]13055 examples [00:14, 902.12 examples/s]13146 examples [00:14, 903.13 examples/s]13237 examples [00:15, 884.36 examples/s]13328 examples [00:15, 890.54 examples/s]13420 examples [00:15, 899.15 examples/s]13514 examples [00:15, 908.38 examples/s]13610 examples [00:15, 922.65 examples/s]13705 examples [00:15, 929.94 examples/s]13799 examples [00:15, 910.29 examples/s]13891 examples [00:15, 885.63 examples/s]13980 examples [00:15, 882.41 examples/s]14070 examples [00:15, 885.77 examples/s]14161 examples [00:16, 889.50 examples/s]14254 examples [00:16, 898.70 examples/s]14344 examples [00:16, 894.57 examples/s]14437 examples [00:16, 903.70 examples/s]14528 examples [00:16, 902.29 examples/s]14622 examples [00:16, 913.11 examples/s]14715 examples [00:16, 915.98 examples/s]14808 examples [00:16, 919.12 examples/s]14902 examples [00:16, 923.74 examples/s]14995 examples [00:16, 902.75 examples/s]15088 examples [00:17, 910.09 examples/s]15180 examples [00:17, 902.75 examples/s]15272 examples [00:17, 905.33 examples/s]15363 examples [00:17, 901.97 examples/s]15454 examples [00:17, 891.60 examples/s]15550 examples [00:17, 910.13 examples/s]15645 examples [00:17, 919.73 examples/s]15738 examples [00:17, 915.52 examples/s]15830 examples [00:17, 916.59 examples/s]15922 examples [00:17, 916.73 examples/s]16014 examples [00:18, 872.95 examples/s]16102 examples [00:18, 871.56 examples/s]16191 examples [00:18, 875.97 examples/s]16281 examples [00:18, 880.80 examples/s]16373 examples [00:18, 891.51 examples/s]16463 examples [00:18, 886.86 examples/s]16553 examples [00:18, 888.95 examples/s]16644 examples [00:18, 894.43 examples/s]16736 examples [00:18, 899.37 examples/s]16826 examples [00:19, 855.82 examples/s]16915 examples [00:19, 864.93 examples/s]17002 examples [00:19, 862.26 examples/s]17093 examples [00:19, 873.42 examples/s]17186 examples [00:19, 888.22 examples/s]17279 examples [00:19, 899.93 examples/s]17372 examples [00:19, 907.39 examples/s]17463 examples [00:19, 898.66 examples/s]17553 examples [00:19, 854.41 examples/s]17639 examples [00:19, 823.83 examples/s]17729 examples [00:20, 844.40 examples/s]17814 examples [00:20, 829.47 examples/s]17900 examples [00:20, 838.10 examples/s]17985 examples [00:20, 819.25 examples/s]18070 examples [00:20, 827.83 examples/s]18156 examples [00:20, 836.97 examples/s]18240 examples [00:20, 835.62 examples/s]18324 examples [00:20, 831.94 examples/s]18408 examples [00:20, 804.31 examples/s]18491 examples [00:20, 811.63 examples/s]18573 examples [00:21, 809.43 examples/s]18660 examples [00:21, 825.06 examples/s]18747 examples [00:21, 835.26 examples/s]18831 examples [00:21, 831.23 examples/s]18921 examples [00:21, 849.66 examples/s]19007 examples [00:21, 841.21 examples/s]19092 examples [00:21, 839.36 examples/s]19179 examples [00:21, 846.73 examples/s]19264 examples [00:21, 820.77 examples/s]19347 examples [00:22, 809.88 examples/s]19429 examples [00:22, 807.97 examples/s]19510 examples [00:22, 790.92 examples/s]19598 examples [00:22, 814.07 examples/s]19687 examples [00:22, 834.94 examples/s]19778 examples [00:22, 856.11 examples/s]19866 examples [00:22, 862.60 examples/s]19953 examples [00:22, 860.77 examples/s]20040 examples [00:22, 805.41 examples/s]20124 examples [00:22, 813.18 examples/s]20216 examples [00:23, 841.80 examples/s]20308 examples [00:23, 862.80 examples/s]20398 examples [00:23, 872.34 examples/s]20491 examples [00:23, 885.93 examples/s]20580 examples [00:23, 883.80 examples/s]20674 examples [00:23, 898.92 examples/s]20765 examples [00:23, 900.72 examples/s]20858 examples [00:23, 907.82 examples/s]20949 examples [00:23, 855.83 examples/s]21039 examples [00:23, 867.79 examples/s]21127 examples [00:24, 864.74 examples/s]21214 examples [00:24, 846.26 examples/s]21301 examples [00:24, 850.84 examples/s]21387 examples [00:24, 846.70 examples/s]21472 examples [00:24, 843.80 examples/s]21558 examples [00:24, 846.16 examples/s]21643 examples [00:24, 838.15 examples/s]21736 examples [00:24, 862.55 examples/s]21823 examples [00:24, 855.77 examples/s]21909 examples [00:25, 850.71 examples/s]21995 examples [00:25, 848.03 examples/s]22080 examples [00:25, 831.14 examples/s]22164 examples [00:25, 814.75 examples/s]22256 examples [00:25, 842.40 examples/s]22345 examples [00:25, 854.56 examples/s]22437 examples [00:25, 870.47 examples/s]22529 examples [00:25, 883.62 examples/s]22618 examples [00:25, 880.27 examples/s]22707 examples [00:25, 861.15 examples/s]22795 examples [00:26, 865.88 examples/s]22882 examples [00:26, 863.25 examples/s]22969 examples [00:26, 861.17 examples/s]23056 examples [00:26, 843.51 examples/s]23141 examples [00:26, 832.29 examples/s]23227 examples [00:26, 839.65 examples/s]23312 examples [00:26, 841.56 examples/s]23397 examples [00:26, 843.60 examples/s]23482 examples [00:26, 839.80 examples/s]23568 examples [00:26, 845.14 examples/s]23660 examples [00:27, 865.85 examples/s]23747 examples [00:27, 835.06 examples/s]23831 examples [00:27, 810.49 examples/s]23920 examples [00:27, 831.55 examples/s]24008 examples [00:27, 843.05 examples/s]24095 examples [00:27, 849.84 examples/s]24181 examples [00:27, 845.07 examples/s]24270 examples [00:27, 856.27 examples/s]24358 examples [00:27, 854.33 examples/s]24444 examples [00:28, 843.92 examples/s]24530 examples [00:28, 848.38 examples/s]24619 examples [00:28, 860.38 examples/s]24709 examples [00:28, 870.59 examples/s]24800 examples [00:28, 880.26 examples/s]24889 examples [00:28, 880.71 examples/s]24978 examples [00:28, 866.74 examples/s]25065 examples [00:28, 856.34 examples/s]25151 examples [00:28, 854.78 examples/s]25237 examples [00:28, 839.77 examples/s]25322 examples [00:29, 824.96 examples/s]25406 examples [00:29, 827.35 examples/s]25491 examples [00:29, 832.93 examples/s]25576 examples [00:29, 835.64 examples/s]25669 examples [00:29, 861.18 examples/s]25760 examples [00:29, 872.75 examples/s]25851 examples [00:29, 882.72 examples/s]25942 examples [00:29, 888.86 examples/s]26032 examples [00:29, 873.18 examples/s]26120 examples [00:29, 851.13 examples/s]26206 examples [00:30, 850.95 examples/s]26295 examples [00:30, 859.87 examples/s]26382 examples [00:30, 851.90 examples/s]26468 examples [00:30, 816.97 examples/s]26551 examples [00:30, 819.07 examples/s]26635 examples [00:30, 824.45 examples/s]26722 examples [00:30, 836.25 examples/s]26814 examples [00:30, 857.29 examples/s]26907 examples [00:30, 875.62 examples/s]26995 examples [00:30, 856.61 examples/s]27081 examples [00:31, 850.78 examples/s]27167 examples [00:31, 850.29 examples/s]27255 examples [00:31, 856.86 examples/s]27344 examples [00:31, 865.04 examples/s]27431 examples [00:31, 841.67 examples/s]27520 examples [00:31, 855.36 examples/s]27606 examples [00:31, 845.14 examples/s]27692 examples [00:31, 848.52 examples/s]27777 examples [00:31, 844.42 examples/s]27862 examples [00:32, 827.79 examples/s]27945 examples [00:32, 824.01 examples/s]28034 examples [00:32, 840.57 examples/s]28120 examples [00:32, 845.67 examples/s]28213 examples [00:32, 867.81 examples/s]28305 examples [00:32, 882.50 examples/s]28398 examples [00:32, 896.21 examples/s]28491 examples [00:32, 904.53 examples/s]28585 examples [00:32, 914.18 examples/s]28678 examples [00:32, 916.56 examples/s]28770 examples [00:33, 910.42 examples/s]28862 examples [00:33, 891.89 examples/s]28952 examples [00:33, 881.84 examples/s]29041 examples [00:33, 873.24 examples/s]29129 examples [00:33, 844.23 examples/s]29218 examples [00:33, 857.12 examples/s]29309 examples [00:33, 869.95 examples/s]29403 examples [00:33, 889.01 examples/s]29498 examples [00:33, 904.54 examples/s]29589 examples [00:33, 889.27 examples/s]29679 examples [00:34, 891.39 examples/s]29771 examples [00:34, 899.15 examples/s]29864 examples [00:34, 905.71 examples/s]29955 examples [00:34, 877.40 examples/s]30044 examples [00:34, 838.68 examples/s]30134 examples [00:34, 856.13 examples/s]30229 examples [00:34, 878.97 examples/s]30322 examples [00:34, 892.50 examples/s]30413 examples [00:34, 896.73 examples/s]30503 examples [00:35, 887.80 examples/s]30592 examples [00:35, 874.30 examples/s]30680 examples [00:35, 864.75 examples/s]30770 examples [00:35, 873.71 examples/s]30858 examples [00:35, 863.44 examples/s]30951 examples [00:35, 880.64 examples/s]31040 examples [00:35, 881.60 examples/s]31129 examples [00:35, 873.19 examples/s]31217 examples [00:35, 862.89 examples/s]31304 examples [00:35, 846.37 examples/s]31394 examples [00:36, 859.83 examples/s]31484 examples [00:36, 869.76 examples/s]31577 examples [00:36, 886.38 examples/s]31670 examples [00:36, 893.93 examples/s]31760 examples [00:36, 875.85 examples/s]31849 examples [00:36, 879.49 examples/s]31938 examples [00:36, 882.59 examples/s]32027 examples [00:36, 881.77 examples/s]32120 examples [00:36, 894.93 examples/s]32210 examples [00:36, 868.63 examples/s]32298 examples [00:37, 860.21 examples/s]32385 examples [00:37, 862.73 examples/s]32472 examples [00:37, 862.49 examples/s]32561 examples [00:37, 868.92 examples/s]32656 examples [00:37, 890.89 examples/s]32752 examples [00:37, 909.66 examples/s]32844 examples [00:37, 884.57 examples/s]32937 examples [00:37, 896.57 examples/s]33029 examples [00:37, 903.33 examples/s]33121 examples [00:37, 908.23 examples/s]33214 examples [00:38, 913.04 examples/s]33306 examples [00:38, 911.91 examples/s]33398 examples [00:38, 913.54 examples/s]33493 examples [00:38, 921.46 examples/s]33587 examples [00:38, 926.55 examples/s]33680 examples [00:38, 927.33 examples/s]33773 examples [00:38, 912.84 examples/s]33865 examples [00:38, 914.57 examples/s]33957 examples [00:38, 914.40 examples/s]34051 examples [00:38, 920.84 examples/s]34144 examples [00:39, 919.06 examples/s]34237 examples [00:39, 921.24 examples/s]34330 examples [00:39, 920.82 examples/s]34423 examples [00:39, 917.75 examples/s]34515 examples [00:39, 878.11 examples/s]34604 examples [00:39, 863.28 examples/s]34695 examples [00:39, 876.34 examples/s]34786 examples [00:39, 885.39 examples/s]34875 examples [00:39, 857.74 examples/s]34963 examples [00:40, 862.31 examples/s]35053 examples [00:40, 872.78 examples/s]35142 examples [00:40, 877.20 examples/s]35234 examples [00:40, 887.92 examples/s]35326 examples [00:40, 895.52 examples/s]35418 examples [00:40, 900.69 examples/s]35511 examples [00:40, 907.68 examples/s]35602 examples [00:40, 907.55 examples/s]35696 examples [00:40, 913.60 examples/s]35789 examples [00:40, 916.66 examples/s]35882 examples [00:41, 919.43 examples/s]35976 examples [00:41, 921.58 examples/s]36069 examples [00:41, 920.39 examples/s]36162 examples [00:41, 911.19 examples/s]36254 examples [00:41, 897.34 examples/s]36344 examples [00:41, 885.07 examples/s]36433 examples [00:41, 882.39 examples/s]36523 examples [00:41, 886.36 examples/s]36612 examples [00:41, 885.32 examples/s]36701 examples [00:41, 871.81 examples/s]36789 examples [00:42, 866.82 examples/s]36876 examples [00:42, 861.04 examples/s]36967 examples [00:42, 874.43 examples/s]37059 examples [00:42, 887.07 examples/s]37153 examples [00:42, 900.30 examples/s]37244 examples [00:42, 883.89 examples/s]37334 examples [00:42, 888.61 examples/s]37426 examples [00:42, 897.20 examples/s]37520 examples [00:42, 908.22 examples/s]37613 examples [00:42, 913.36 examples/s]37708 examples [00:43, 922.47 examples/s]37801 examples [00:43, 915.78 examples/s]37893 examples [00:43, 904.32 examples/s]37987 examples [00:43, 910.66 examples/s]38079 examples [00:43, 889.18 examples/s]38169 examples [00:43, 869.63 examples/s]38257 examples [00:43, 852.47 examples/s]38343 examples [00:43, 830.53 examples/s]38427 examples [00:43, 827.51 examples/s]38514 examples [00:44, 837.35 examples/s]38599 examples [00:44, 840.96 examples/s]38691 examples [00:44, 862.51 examples/s]38784 examples [00:44, 881.38 examples/s]38879 examples [00:44, 900.67 examples/s]38972 examples [00:44, 907.28 examples/s]39066 examples [00:44, 915.04 examples/s]39158 examples [00:44, 875.42 examples/s]39247 examples [00:44, 862.36 examples/s]39334 examples [00:44, 850.15 examples/s]39420 examples [00:45, 830.87 examples/s]39504 examples [00:45, 820.57 examples/s]39589 examples [00:45, 828.99 examples/s]39674 examples [00:45, 834.97 examples/s]39759 examples [00:45, 838.60 examples/s]39846 examples [00:45, 845.90 examples/s]39931 examples [00:45, 817.84 examples/s]40014 examples [00:45, 782.08 examples/s]40108 examples [00:45, 821.40 examples/s]40203 examples [00:46, 854.38 examples/s]40298 examples [00:46, 878.96 examples/s]40387 examples [00:46, 873.50 examples/s]40475 examples [00:46, 863.55 examples/s]40562 examples [00:46, 864.06 examples/s]40649 examples [00:46, 858.44 examples/s]40736 examples [00:46, 841.41 examples/s]40824 examples [00:46, 851.58 examples/s]40911 examples [00:46, 856.19 examples/s]41002 examples [00:46, 871.54 examples/s]41096 examples [00:47, 890.81 examples/s]41191 examples [00:47, 907.24 examples/s]41284 examples [00:47, 911.80 examples/s]41376 examples [00:47, 908.89 examples/s]41468 examples [00:47, 884.56 examples/s]41557 examples [00:47, 806.89 examples/s]41646 examples [00:47, 828.39 examples/s]41732 examples [00:47, 835.07 examples/s]41824 examples [00:47, 856.26 examples/s]41911 examples [00:47, 857.30 examples/s]42003 examples [00:48, 874.68 examples/s]42096 examples [00:48, 888.67 examples/s]42188 examples [00:48, 897.75 examples/s]42279 examples [00:48, 867.02 examples/s]42367 examples [00:48, 866.03 examples/s]42461 examples [00:48, 885.10 examples/s]42550 examples [00:48, 873.24 examples/s]42639 examples [00:48, 876.15 examples/s]42733 examples [00:48, 892.15 examples/s]42826 examples [00:48, 902.78 examples/s]42922 examples [00:49, 917.62 examples/s]43014 examples [00:49, 874.30 examples/s]43102 examples [00:49, 861.38 examples/s]43189 examples [00:49, 856.50 examples/s]43275 examples [00:49, 850.07 examples/s]43362 examples [00:49, 855.51 examples/s]43454 examples [00:49, 873.62 examples/s]43543 examples [00:49, 877.37 examples/s]43636 examples [00:49, 891.79 examples/s]43726 examples [00:50, 860.61 examples/s]43813 examples [00:50, 827.24 examples/s]43903 examples [00:50, 847.00 examples/s]43992 examples [00:50, 857.42 examples/s]44079 examples [00:50, 846.37 examples/s]44164 examples [00:50, 845.59 examples/s]44250 examples [00:50, 848.86 examples/s]44339 examples [00:50, 858.99 examples/s]44430 examples [00:50, 871.81 examples/s]44521 examples [00:50, 882.05 examples/s]44610 examples [00:51, 860.91 examples/s]44697 examples [00:51, 850.42 examples/s]44790 examples [00:51, 870.44 examples/s]44878 examples [00:51, 831.39 examples/s]44962 examples [00:51, 824.25 examples/s]45047 examples [00:51, 829.11 examples/s]45131 examples [00:51, 827.06 examples/s]45214 examples [00:51, 822.17 examples/s]45297 examples [00:51, 811.56 examples/s]45381 examples [00:52, 819.27 examples/s]45467 examples [00:52, 830.83 examples/s]45559 examples [00:52, 855.45 examples/s]45649 examples [00:52, 867.29 examples/s]45738 examples [00:52, 873.50 examples/s]45832 examples [00:52, 891.69 examples/s]45922 examples [00:52, 841.71 examples/s]46009 examples [00:52, 849.34 examples/s]46095 examples [00:52, 829.17 examples/s]46182 examples [00:52, 838.97 examples/s]46273 examples [00:53, 857.01 examples/s]46368 examples [00:53, 880.91 examples/s]46457 examples [00:53, 868.14 examples/s]46549 examples [00:53, 881.26 examples/s]46638 examples [00:53, 882.13 examples/s]46732 examples [00:53, 897.44 examples/s]46823 examples [00:53, 898.39 examples/s]46913 examples [00:53, 898.61 examples/s]47006 examples [00:53, 907.03 examples/s]47097 examples [00:53, 899.61 examples/s]47188 examples [00:54, 901.92 examples/s]47279 examples [00:54, 902.14 examples/s]47370 examples [00:54, 896.78 examples/s]47461 examples [00:54, 898.12 examples/s]47551 examples [00:54, 893.56 examples/s]47643 examples [00:54, 898.79 examples/s]47734 examples [00:54, 901.76 examples/s]47825 examples [00:54, 887.90 examples/s]47914 examples [00:54, 825.99 examples/s]47999 examples [00:54, 832.83 examples/s]48088 examples [00:55, 846.76 examples/s]48178 examples [00:55, 859.48 examples/s]48271 examples [00:55, 877.03 examples/s]48360 examples [00:55, 850.15 examples/s]48452 examples [00:55, 867.53 examples/s]48540 examples [00:55, 868.88 examples/s]48628 examples [00:55, 853.97 examples/s]48714 examples [00:55, 832.14 examples/s]48800 examples [00:55, 839.02 examples/s]48891 examples [00:56, 857.60 examples/s]48980 examples [00:56, 866.18 examples/s]49073 examples [00:56, 883.75 examples/s]49164 examples [00:56, 891.13 examples/s]49254 examples [00:56, 887.11 examples/s]49343 examples [00:56, 845.82 examples/s]49430 examples [00:56, 850.89 examples/s]49523 examples [00:56, 870.88 examples/s]49614 examples [00:56, 881.12 examples/s]49705 examples [00:56, 887.92 examples/s]49797 examples [00:57, 894.78 examples/s]49887 examples [00:57, 893.66 examples/s]49977 examples [00:57, 888.12 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6008/50000 [00:00<00:00, 60078.62 examples/s] 33%|███▎      | 16485/50000 [00:00<00:00, 68893.65 examples/s] 54%|█████▎    | 26849/50000 [00:00<00:00, 76597.31 examples/s] 72%|███████▏  | 36179/50000 [00:00<00:00, 80942.86 examples/s] 93%|█████████▎| 46266/50000 [00:00<00:00, 86040.41 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 753.49 examples/s]169 examples [00:00, 797.69 examples/s]256 examples [00:00, 816.81 examples/s]335 examples [00:00, 808.31 examples/s]423 examples [00:00, 827.30 examples/s]512 examples [00:00, 843.00 examples/s]609 examples [00:00, 875.58 examples/s]697 examples [00:00, 679.68 examples/s]787 examples [00:01, 733.12 examples/s]880 examples [00:01, 781.67 examples/s]973 examples [00:01, 820.30 examples/s]1066 examples [00:01, 848.35 examples/s]1159 examples [00:01, 868.83 examples/s]1252 examples [00:01, 885.51 examples/s]1342 examples [00:01, 887.57 examples/s]1436 examples [00:01, 902.40 examples/s]1528 examples [00:01, 907.55 examples/s]1621 examples [00:01, 913.28 examples/s]1713 examples [00:02, 841.26 examples/s]1799 examples [00:02, 838.45 examples/s]1889 examples [00:02, 854.53 examples/s]1977 examples [00:02, 861.23 examples/s]2064 examples [00:02, 859.94 examples/s]2151 examples [00:02, 861.29 examples/s]2239 examples [00:02, 866.61 examples/s]2334 examples [00:02, 888.54 examples/s]2429 examples [00:02, 905.96 examples/s]2521 examples [00:02, 908.94 examples/s]2613 examples [00:03, 890.79 examples/s]2703 examples [00:03, 886.53 examples/s]2795 examples [00:03, 895.34 examples/s]2886 examples [00:03, 899.67 examples/s]2977 examples [00:03, 902.09 examples/s]3068 examples [00:03, 881.30 examples/s]3157 examples [00:03, 853.30 examples/s]3250 examples [00:03, 873.87 examples/s]3342 examples [00:03, 884.95 examples/s]3431 examples [00:03, 872.19 examples/s]3522 examples [00:04, 881.46 examples/s]3614 examples [00:04, 891.15 examples/s]3704 examples [00:04, 881.78 examples/s]3793 examples [00:04, 879.55 examples/s]3882 examples [00:04, 881.19 examples/s]3971 examples [00:04, 868.64 examples/s]4060 examples [00:04, 874.12 examples/s]4151 examples [00:04, 884.08 examples/s]4240 examples [00:04, 884.08 examples/s]4333 examples [00:04, 895.31 examples/s]4423 examples [00:05, 890.18 examples/s]4513 examples [00:05, 873.12 examples/s]4601 examples [00:05, 864.52 examples/s]4693 examples [00:05, 877.86 examples/s]4786 examples [00:05, 892.68 examples/s]4878 examples [00:05, 899.76 examples/s]4969 examples [00:05, 900.28 examples/s]5063 examples [00:05, 911.49 examples/s]5158 examples [00:05, 920.16 examples/s]5251 examples [00:06, 920.78 examples/s]5344 examples [00:06, 891.52 examples/s]5434 examples [00:06, 852.93 examples/s]5526 examples [00:06, 869.89 examples/s]5614 examples [00:06, 859.46 examples/s]5709 examples [00:06, 883.90 examples/s]5803 examples [00:06, 899.58 examples/s]5896 examples [00:06, 908.11 examples/s]5988 examples [00:06, 887.92 examples/s]6079 examples [00:06, 893.07 examples/s]6169 examples [00:07, 884.94 examples/s]6261 examples [00:07, 893.07 examples/s]6351 examples [00:07, 892.03 examples/s]6441 examples [00:07, 888.74 examples/s]6531 examples [00:07, 888.95 examples/s]6620 examples [00:07, 865.95 examples/s]6709 examples [00:07, 872.67 examples/s]6801 examples [00:07, 885.50 examples/s]6890 examples [00:07, 868.00 examples/s]6978 examples [00:07, 869.82 examples/s]7069 examples [00:08, 880.91 examples/s]7158 examples [00:08, 874.74 examples/s]7246 examples [00:08, 868.77 examples/s]7333 examples [00:08, 837.29 examples/s]7418 examples [00:08, 793.95 examples/s]7507 examples [00:08, 819.96 examples/s]7595 examples [00:08, 836.08 examples/s]7689 examples [00:08, 864.01 examples/s]7784 examples [00:08, 885.65 examples/s]7878 examples [00:09, 900.30 examples/s]7971 examples [00:09, 908.85 examples/s]8063 examples [00:09, 910.83 examples/s]8155 examples [00:09, 878.23 examples/s]8247 examples [00:09, 887.75 examples/s]8340 examples [00:09, 898.94 examples/s]8433 examples [00:09, 907.13 examples/s]8524 examples [00:09, 907.75 examples/s]8618 examples [00:09, 910.05 examples/s]8710 examples [00:09, 902.81 examples/s]8801 examples [00:10, 901.35 examples/s]8892 examples [00:10, 864.01 examples/s]8979 examples [00:10, 856.14 examples/s]9065 examples [00:10, 856.16 examples/s]9151 examples [00:10, 854.74 examples/s]9237 examples [00:10, 854.92 examples/s]9329 examples [00:10, 871.54 examples/s]9420 examples [00:10, 882.03 examples/s]9513 examples [00:10, 894.36 examples/s]9606 examples [00:10, 903.41 examples/s]9697 examples [00:11, 899.15 examples/s]9788 examples [00:11, 899.25 examples/s]9878 examples [00:11, 885.54 examples/s]9967 examples [00:11, 861.19 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW0AXNK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW0AXNK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f47f556c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f47f556c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f47f556c8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4861a0fda0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f47deb226a0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f47f556c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f47f556c8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f47f556c8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f47d629a6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f47d8bde2e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f47cf03a6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f47cf03a6a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f47cf03a6a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f47cf03a7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f47cf03a7b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f47cf03a7b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=43e2c8d11e56c015eea6cb36d26481fa16b03917e0bd66e09ca6dd82649b7b04
  Stored in directory: /tmp/pip-ephem-wheel-cache-fifmqygn/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<136:44:23, 1.75kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<95:56:51, 2.50kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<67:12:19, 3.56kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<47:01:14, 5.09kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<32:48:47, 7.27kB/s].vector_cache/glove.6B.zip:   1%|          | 9.59M/862M [00:05<22:48:42, 10.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:05<15:51:41, 14.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:05<11:01:48, 21.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:05<7:40:13, 30.3kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:05<5:19:56, 43.2kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:05<3:42:26, 61.7kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:06<2:35:20, 88.1kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:06<1:48:12, 126kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:06<1:15:25, 179kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:06<53:55, 250kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:08<39:28, 340kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:08<31:27, 427kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:08<22:55, 585kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:09<16:14, 824kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:10<17:05, 782kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:10<13:20, 1.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:10<09:39, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:12<09:50, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:12<08:02, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:12<05:57, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:14<07:17, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:14<07:47, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:14<06:06, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:16<06:24, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:16<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:16<04:24, 2.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:18<06:08, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:18<06:57, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:18<05:31, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:20<05:57, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:20<05:31, 2.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:20<04:11, 3.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:22<05:57, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:22<06:53, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:22<05:23, 2.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:22<03:55, 3.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:24<11:30, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:24<09:23, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:24<06:52, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:26<07:49, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<06:46, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:26<05:03, 2.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:28<06:33, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:28<07:11, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:28<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<04:02, 3.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<10:38, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<08:45, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<06:26, 1.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<07:27, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<07:47, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<05:59, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<04:20, 2.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<12:25, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<09:57, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<07:16, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<06:51, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<05:06, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<06:30, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<07:03, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<05:34, 2.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<05:53, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<05:26, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<04:07, 2.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<05:44, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<06:31, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<05:10, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<03:44, 3.27MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<38:16, 319kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<28:03, 434kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<19:52, 612kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<16:37, 729kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<14:12, 854kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<10:34, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<07:30, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<33:10, 364kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<24:30, 492kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<17:26, 690kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<14:52, 806kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<12:56, 926kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<09:36, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<06:49, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<18:54, 631kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<14:30, 821kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<10:25, 1.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<09:56, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<09:22, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<07:10, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<05:09, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<42:47, 275kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<32:26, 363kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<23:18, 505kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<16:23, 714kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<48:44, 240kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<35:20, 331kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<24:56, 468kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<20:03, 581kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<16:31, 704kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<12:10, 955kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<08:37, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<32:45, 353kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<24:09, 479kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<17:10, 672kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<14:35, 789kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<12:38, 910kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<09:21, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<06:39, 1.72MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<13:13, 865kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<10:27, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<07:34, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<07:51, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<07:54, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<06:03, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<04:20, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<13:28, 839kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<10:36, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<07:53, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<07:53, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<06:02, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<04:20, 2.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<13:00, 858kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<10:17, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:13<07:26, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<07:43, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<07:45, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<04:19, 2.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<29:28, 374kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<21:47, 505kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:17<15:28, 710kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<13:16, 825kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<11:38, 941kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<08:43, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<06:13, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<37:04, 294kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<27:05, 402kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<19:09, 566kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<15:49, 684kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<12:12, 885kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<08:45, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<08:33, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<08:16, 1.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<06:21, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<04:33, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<28:55, 369kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<21:21, 500kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<15:09, 702kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<12:59, 817kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<11:20, 935kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<08:24, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<05:59, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<10:21, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<08:22, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<06:07, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<06:38, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<06:51, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<05:15, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:33<03:49, 2.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<07:10, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<06:07, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<04:33, 2.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:30, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<06:04, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:42, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<03:25, 3.00MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<06:59, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<04:27, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<05:24, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<04:35, 2.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<03:20, 3.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<07:08, 1.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<06:03, 1.67MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<04:30, 2.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<05:24, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<04:35, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<03:19, 3.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<07:57, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<06:35, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<04:52, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<05:41, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<06:03, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<04:40, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<03:24, 2.90MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<06:54, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:51<05:51, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<04:18, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:15, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:44, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<04:32, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<03:17, 2.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<31:29, 309kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<23:03, 421kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<16:20, 592kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<13:36, 708kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<10:32, 914kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<07:36, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<07:28, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<07:09, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:59<05:28, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<03:55, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<44:52, 212kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<32:13, 295kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<22:46, 416kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<16:00, 590kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<21:46, 433kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<17:13, 548kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<12:32, 751kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<08:51, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<34:21, 273kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<25:01, 374kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<17:41, 528kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<14:27, 644kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<11:59, 775kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<08:48, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<06:18, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<07:33, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<06:14, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<04:36, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<05:19, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<05:39, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<03:13, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<29:31, 308kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<21:36, 421kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<15:17, 593kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<12:44, 709kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<10:48, 834kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<07:59, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<05:40, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<11:30, 779kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<09:00, 993kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<06:29, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<06:30, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<06:26, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<04:54, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<03:30, 2.51MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<11:55, 740kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<09:16, 950kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<06:42, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<06:39, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<06:25, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<04:56, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<03:32, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<48:05, 181kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<34:32, 251kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<24:19, 355kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<18:56, 455kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<15:01, 573kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<10:56, 786kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<07:43, 1.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<1:15:04, 114kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<53:24, 160kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<37:28, 227kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<28:02, 302kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<21:24, 396kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<15:20, 551kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<10:46, 781kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<17:40, 476kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<13:14, 634kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<09:27, 885kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<08:28, 983kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<06:48, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<04:58, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<05:20, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<05:24, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<04:07, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<02:59, 2.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<05:48, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<04:55, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<03:39, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<04:22, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<04:47, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<03:42, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:40<02:42, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<05:15, 1.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<04:30, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<03:19, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<04:10, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<04:31, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<03:34, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<02:35, 3.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<58:30, 135kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<41:45, 190kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<29:19, 269kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<22:12, 354kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<17:12, 456kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<12:23, 633kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<08:42, 895kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<12:08, 641kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<09:18, 835kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:50<06:40, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<06:24, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<05:15, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:52<03:50, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:28, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<04:44, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:39, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<02:42, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<04:02, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<03:37, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<02:42, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<04:06, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<03:11, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<02:24, 3.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<03:33, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<03:15, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<02:28, 3.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<03:03, 2.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<02:34, 2.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<01:53, 3.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:43, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:04, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<03:02, 2.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<03:45, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<04:09, 1.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:14, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<02:23, 3.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<03:53, 1.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<03:28, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<02:35, 2.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:25, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:53, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:04, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<02:12, 3.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<18:03, 389kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<13:22, 525kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<09:29, 737kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<08:10, 851kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<07:11, 966kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<05:23, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<03:49, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<35:56, 192kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<25:51, 266kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<18:11, 377kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<14:13, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<11:23, 598kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<08:19, 817kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<05:52, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<25:31, 264kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<18:32, 364kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<13:06, 512kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:21<10:40, 626kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<08:09, 817kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<05:51, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<05:33, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<05:17, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<04:02, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:24<02:52, 2.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<33:32, 195kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<24:09, 271kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<17:00, 383kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<13:19, 486kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<10:41, 605kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<07:45, 833kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<05:28, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<07:37, 841kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<06:00, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<04:19, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<04:27, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:45, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<02:46, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<03:24, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<03:02, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<02:16, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:01, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:25, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<02:40, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<01:56, 3.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<04:39, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<02:51, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<03:21, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<03:38, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:01, 2.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<11:48, 507kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<08:53, 673kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<06:21, 938kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<05:45, 1.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<04:37, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:43, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:12, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<02:22, 2.45MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<02:57, 1.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:17, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<02:36, 2.22MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<01:52, 3.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<18:31, 309kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<13:33, 421kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<09:35, 593kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<07:56, 711kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<06:45, 836kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<04:58, 1.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<03:31, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<06:09, 905kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<04:54, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<03:32, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<03:42, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<03:46, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:55<02:55, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<02:05, 2.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<40:24, 135kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<28:49, 188kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<20:13, 267kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<15:16, 352kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<11:50, 453kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:59<08:30, 629kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<05:59, 889kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<06:58, 760kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<05:26, 972kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<03:56, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<03:54, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<02:57, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<02:07, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<13:45, 375kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<10:10, 507kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<07:12, 712kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<06:09, 827kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<05:24, 942kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<04:02, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:07<02:52, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<13:57, 360kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<10:17, 488kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<07:17, 684kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<06:11, 801kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<05:19, 929kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<03:57, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<02:49, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<03:54, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<03:14, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<02:23, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:46, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:55, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<02:14, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<01:38, 2.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<03:03, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:38, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<01:57, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<02:25, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:19<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<01:37, 2.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:21<02:12, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:21<02:01, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<01:31, 3.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<01:27, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<02:03, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<01:49, 2.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<01:23, 3.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<01:02, 4.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<11:14, 392kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<08:45, 503kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:27<06:18, 697kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<04:25, 982kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<05:19, 814kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:29<04:06, 1.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<02:59, 1.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<02:08, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<06:15, 681kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:31<05:17, 807kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<03:52, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<02:44, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<05:37, 748kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<04:22, 959kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<03:09, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:08, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:35<03:04, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<02:21, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<10:33, 385kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<07:48, 519kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<05:32, 728kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<04:45, 839kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<04:10, 956kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:39<03:07, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:12, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<11:26, 343kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<08:24, 466kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<05:56, 656kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:59, 772kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:18, 895kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<02:15, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<03:22, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<02:45, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<02:01, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:15, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:19, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:47, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:17, 2.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:38, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<02:14, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<01:38, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:07, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:39, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<01:11, 2.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<25:54, 136kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<18:28, 190kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<12:55, 270kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<09:43, 354kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<07:30, 458kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<05:24, 633kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<03:46, 894kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<30:14, 112kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<21:29, 157kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<15:01, 223kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<11:08, 297kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<08:29, 389kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<06:03, 542kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<04:14, 767kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<04:34, 707kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:00<03:32, 912kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<02:32, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:29, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:04, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<01:31, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:46, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:04<01:33, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<01:09, 2.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:29, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:41, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:20, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<00:57, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<09:35, 309kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<07:00, 422kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<04:55, 594kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<04:03, 712kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<03:27, 834kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:10<02:33, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:48, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<09:43, 290kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<07:05, 397kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<04:59, 559kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<04:05, 674kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<03:24, 806kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<02:31, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<01:49, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<01:52, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<01:37, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<01:11, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:22, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:31, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:12, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<00:51, 2.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<06:43, 379kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<04:55, 517kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<03:30, 719kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<02:27, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<07:11, 345kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<05:32, 447kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<03:58, 619kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<02:45, 876kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<04:53, 494kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<03:39, 656kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<02:35, 918kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<02:19, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<02:07, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<01:49, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<01:20, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<01:02, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:53, 2.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<00:43, 3.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<01:59, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<02:36, 870kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<02:07, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<01:13, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<00:45, 2.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<02:21, 932kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<02:43, 806kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<02:10, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:37, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<01:12, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<00:56, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<00:44, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<00:37, 3.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<30:39, 69.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<22:23, 95.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<15:53, 134kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<11:08, 190kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<07:50, 268kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:32<05:31, 377kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<04:42, 439kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:34<04:12, 491kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<03:10, 650kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<02:17, 891kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<01:39, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<01:16, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<00:57, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<03:24, 587kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:36<03:15, 612kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<02:29, 800kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:36<01:49, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<01:20, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:36<01:00, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<01:38, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<02:00, 963kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<01:35, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<01:11, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<00:53, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<00:41, 2.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:30, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<01:52, 987kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<01:30, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<01:07, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:51, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<00:39, 2.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:30, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<01:46, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<01:26, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<01:04, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<00:49, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<00:37, 2.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<00:30, 3.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:47, 962kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:56, 883kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:44<01:32, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<01:08, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:51, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<00:39, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<00:31, 3.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<02:26, 676kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<02:22, 695kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:46<01:49, 898kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<01:20, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:46<00:58, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:45, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:34, 2.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<03:36, 440kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<03:09, 500kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<02:21, 666kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<01:42, 912kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<01:14, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<00:55, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:42, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<04:06, 369kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<03:28, 435kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<02:34, 584kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<01:51, 805kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<01:20, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:58, 1.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:36, 895kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<01:42, 842kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:52<01:20, 1.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<00:59, 1.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<00:33, 2.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<01:24, 978kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<01:30, 914kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<01:09, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:52, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:39, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<00:29, 2.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:23, 936kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:27, 892kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:08, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:50, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:37, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<00:28, 2.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:22, 3.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<58:32, 21.2kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<41:23, 29.9kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<28:54, 42.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<19:59, 60.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<13:47, 86.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<09:31, 123kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<11:19, 103kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<08:19, 140kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<05:53, 197kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<04:05, 279kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<02:50, 393kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<02:22, 465kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<02:25, 454kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<01:52, 585kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<01:20, 803kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<00:57, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:41, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<01:04, 956kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<01:04, 965kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:49, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:35, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:25, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:52, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<01:08, 848kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:55, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:40, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:29, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:38, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:55, 959kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:46, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:33, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:23, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:35, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:49, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:40, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:28, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:20, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:35, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:46, 964kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:37, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:27, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:19, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:36, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:42, 970kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:33, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:24, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:16, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:44, 838kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:44, 825kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:34, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:16, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<01:27, 377kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<01:13, 448kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:53, 605kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:36, 843kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<00:30, 954kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:30, 926kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:23, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<00:16, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:16, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:19, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:14, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<00:10, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:12, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:14, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<00:07, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:09, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:11, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:09, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:25<00:06, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:06, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:05, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:03, 2.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:05, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:05, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:03, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:01, 2.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:02, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:02, 1.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:01, 1.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 2.44MB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<110:58:09,  1.00it/s]  0%|          | 605/400000 [00:01<77:34:00,  1.43it/s]  0%|          | 1227/400000 [00:01<54:13:03,  2.04it/s]  0%|          | 1840/400000 [00:01<37:53:57,  2.92it/s]  1%|          | 2411/400000 [00:01<26:29:50,  4.17it/s]  1%|          | 3085/400000 [00:01<18:31:17,  5.95it/s]  1%|          | 3678/400000 [00:01<12:57:04,  8.50it/s]  1%|          | 4390/400000 [00:01<9:03:15, 12.14it/s]   1%|▏         | 5076/400000 [00:01<6:19:54, 17.33it/s]  1%|▏         | 5723/400000 [00:01<4:25:48, 24.72it/s]  2%|▏         | 6361/400000 [00:01<3:06:04, 35.26it/s]  2%|▏         | 6993/400000 [00:02<2:10:21, 50.24it/s]  2%|▏         | 7624/400000 [00:02<1:31:25, 71.53it/s]  2%|▏         | 8264/400000 [00:02<1:04:11, 101.70it/s]  2%|▏         | 8902/400000 [00:02<45:10, 144.30it/s]    2%|▏         | 9538/400000 [00:02<31:52, 204.16it/s]  3%|▎         | 10195/400000 [00:02<22:34, 287.82it/s]  3%|▎         | 10924/400000 [00:02<16:02, 404.32it/s]  3%|▎         | 11591/400000 [00:02<11:30, 562.90it/s]  3%|▎         | 12320/400000 [00:02<08:18, 778.36it/s]  3%|▎         | 13024/400000 [00:03<06:04, 1061.62it/s]  3%|▎         | 13715/400000 [00:03<04:32, 1416.97it/s]  4%|▎         | 14393/400000 [00:03<03:28, 1852.14it/s]  4%|▍         | 15064/400000 [00:03<02:43, 2352.01it/s]  4%|▍         | 15745/400000 [00:03<02:11, 2926.41it/s]  4%|▍         | 16460/400000 [00:03<01:47, 3556.42it/s]  4%|▍         | 17141/400000 [00:03<01:33, 4111.23it/s]  4%|▍         | 17812/400000 [00:03<01:22, 4643.38it/s]  5%|▍         | 18481/400000 [00:03<01:15, 5036.22it/s]  5%|▍         | 19200/400000 [00:03<01:08, 5531.65it/s]  5%|▍         | 19889/400000 [00:04<01:04, 5877.63it/s]  5%|▌         | 20573/400000 [00:04<01:01, 6135.67it/s]  5%|▌         | 21254/400000 [00:04<01:00, 6254.63it/s]  5%|▌         | 21927/400000 [00:04<01:00, 6299.16it/s]  6%|▌         | 22632/400000 [00:04<00:57, 6506.91it/s]  6%|▌         | 23317/400000 [00:04<00:57, 6605.68it/s]  6%|▌         | 23996/400000 [00:04<00:57, 6552.62it/s]  6%|▌         | 24664/400000 [00:04<00:58, 6444.59it/s]  6%|▋         | 25318/400000 [00:04<00:58, 6455.64it/s]  7%|▋         | 26035/400000 [00:04<00:56, 6652.60it/s]  7%|▋         | 26763/400000 [00:05<00:54, 6827.27it/s]  7%|▋         | 27481/400000 [00:05<00:53, 6928.34it/s]  7%|▋         | 28210/400000 [00:05<00:52, 7032.17it/s]  7%|▋         | 28917/400000 [00:05<00:53, 6907.11it/s]  7%|▋         | 29626/400000 [00:05<00:53, 6960.09it/s]  8%|▊         | 30325/400000 [00:05<00:54, 6837.27it/s]  8%|▊         | 31011/400000 [00:05<00:54, 6742.71it/s]  8%|▊         | 31687/400000 [00:05<00:55, 6626.04it/s]  8%|▊         | 32384/400000 [00:05<00:54, 6723.11it/s]  8%|▊         | 33058/400000 [00:05<00:54, 6695.35it/s]  8%|▊         | 33785/400000 [00:06<00:53, 6856.30it/s]  9%|▊         | 34505/400000 [00:06<00:52, 6954.25it/s]  9%|▉         | 35230/400000 [00:06<00:51, 7038.95it/s]  9%|▉         | 35954/400000 [00:06<00:51, 7097.15it/s]  9%|▉         | 36665/400000 [00:06<00:51, 7056.04it/s]  9%|▉         | 37372/400000 [00:06<00:53, 6774.60it/s] 10%|▉         | 38053/400000 [00:06<00:53, 6717.28it/s] 10%|▉         | 38727/400000 [00:06<00:53, 6700.86it/s] 10%|▉         | 39399/400000 [00:06<00:54, 6607.05it/s] 10%|█         | 40072/400000 [00:07<00:54, 6641.42it/s] 10%|█         | 40799/400000 [00:07<00:52, 6817.75it/s] 10%|█         | 41513/400000 [00:07<00:51, 6906.55it/s] 11%|█         | 42206/400000 [00:07<00:51, 6912.85it/s] 11%|█         | 42918/400000 [00:07<00:51, 6971.93it/s] 11%|█         | 43617/400000 [00:07<00:51, 6905.88it/s] 11%|█         | 44350/400000 [00:07<00:50, 7026.12it/s] 11%|█▏        | 45079/400000 [00:07<00:49, 7101.07it/s] 11%|█▏        | 45791/400000 [00:07<00:50, 7070.01it/s] 12%|█▏        | 46499/400000 [00:07<00:50, 6976.42it/s] 12%|█▏        | 47208/400000 [00:08<00:50, 6981.15it/s] 12%|█▏        | 47907/400000 [00:08<00:50, 6963.42it/s] 12%|█▏        | 48626/400000 [00:08<00:49, 7029.35it/s] 12%|█▏        | 49330/400000 [00:08<00:50, 6921.10it/s] 13%|█▎        | 50029/400000 [00:08<00:50, 6940.61it/s] 13%|█▎        | 50724/400000 [00:08<00:51, 6816.16it/s] 13%|█▎        | 51454/400000 [00:08<00:50, 6953.36it/s] 13%|█▎        | 52183/400000 [00:08<00:49, 7049.52it/s] 13%|█▎        | 52901/400000 [00:08<00:48, 7085.64it/s] 13%|█▎        | 53624/400000 [00:08<00:48, 7127.28it/s] 14%|█▎        | 54354/400000 [00:09<00:48, 7176.43it/s] 14%|█▍        | 55073/400000 [00:09<00:49, 7021.71it/s] 14%|█▍        | 55802/400000 [00:09<00:48, 7099.51it/s] 14%|█▍        | 56513/400000 [00:09<00:48, 7056.19it/s] 14%|█▍        | 57220/400000 [00:09<00:49, 6867.61it/s] 14%|█▍        | 57909/400000 [00:09<00:50, 6719.38it/s] 15%|█▍        | 58628/400000 [00:09<00:49, 6837.67it/s] 15%|█▍        | 59314/400000 [00:09<00:50, 6711.30it/s] 15%|█▌        | 60032/400000 [00:09<00:49, 6842.95it/s] 15%|█▌        | 60732/400000 [00:09<00:49, 6887.70it/s] 15%|█▌        | 61423/400000 [00:10<00:49, 6873.50it/s] 16%|█▌        | 62133/400000 [00:10<00:48, 6939.41it/s] 16%|█▌        | 62850/400000 [00:10<00:48, 7006.59it/s] 16%|█▌        | 63552/400000 [00:10<00:48, 6955.31it/s] 16%|█▌        | 64272/400000 [00:10<00:47, 7025.89it/s] 16%|█▌        | 64976/400000 [00:10<00:48, 6941.95it/s] 16%|█▋        | 65676/400000 [00:10<00:48, 6958.59it/s] 17%|█▋        | 66399/400000 [00:10<00:47, 7036.18it/s] 17%|█▋        | 67129/400000 [00:10<00:46, 7112.66it/s] 17%|█▋        | 67841/400000 [00:10<00:47, 7022.94it/s] 17%|█▋        | 68560/400000 [00:11<00:46, 7070.52it/s] 17%|█▋        | 69292/400000 [00:11<00:46, 7142.18it/s] 18%|█▊        | 70007/400000 [00:11<00:46, 7097.95it/s] 18%|█▊        | 70718/400000 [00:11<00:46, 7049.07it/s] 18%|█▊        | 71424/400000 [00:11<00:46, 7017.50it/s] 18%|█▊        | 72127/400000 [00:11<00:48, 6720.01it/s] 18%|█▊        | 72836/400000 [00:11<00:47, 6823.70it/s] 18%|█▊        | 73542/400000 [00:11<00:47, 6891.09it/s] 19%|█▊        | 74265/400000 [00:11<00:46, 6987.55it/s] 19%|█▊        | 74966/400000 [00:12<00:46, 6955.84it/s] 19%|█▉        | 75693/400000 [00:12<00:46, 7045.31it/s] 19%|█▉        | 76399/400000 [00:12<00:46, 7004.38it/s] 19%|█▉        | 77119/400000 [00:12<00:45, 7061.86it/s] 19%|█▉        | 77826/400000 [00:12<00:45, 7016.27it/s] 20%|█▉        | 78529/400000 [00:12<00:45, 7006.16it/s] 20%|█▉        | 79231/400000 [00:12<00:46, 6853.42it/s] 20%|█▉        | 79961/400000 [00:12<00:45, 6978.87it/s] 20%|██        | 80661/400000 [00:12<00:46, 6940.66it/s] 20%|██        | 81357/400000 [00:12<00:46, 6915.85it/s] 21%|██        | 82082/400000 [00:13<00:45, 7011.95it/s] 21%|██        | 82784/400000 [00:13<00:45, 6981.68it/s] 21%|██        | 83503/400000 [00:13<00:44, 7041.39it/s] 21%|██        | 84241/400000 [00:13<00:44, 7137.62it/s] 21%|██        | 84956/400000 [00:13<00:44, 7064.03it/s] 21%|██▏       | 85664/400000 [00:13<00:44, 6987.22it/s] 22%|██▏       | 86396/400000 [00:13<00:44, 7083.76it/s] 22%|██▏       | 87122/400000 [00:13<00:43, 7135.09it/s] 22%|██▏       | 87852/400000 [00:13<00:43, 7182.63it/s] 22%|██▏       | 88582/400000 [00:13<00:43, 7215.07it/s] 22%|██▏       | 89304/400000 [00:14<00:43, 7145.55it/s] 23%|██▎       | 90019/400000 [00:14<00:44, 7012.68it/s] 23%|██▎       | 90751/400000 [00:14<00:43, 7100.70it/s] 23%|██▎       | 91487/400000 [00:14<00:42, 7175.93it/s] 23%|██▎       | 92220/400000 [00:14<00:42, 7219.05it/s] 23%|██▎       | 92943/400000 [00:14<00:42, 7191.24it/s] 23%|██▎       | 93663/400000 [00:14<00:43, 7071.20it/s] 24%|██▎       | 94371/400000 [00:14<00:43, 7031.62it/s] 24%|██▍       | 95089/400000 [00:14<00:43, 7075.41it/s] 24%|██▍       | 95821/400000 [00:14<00:42, 7144.93it/s] 24%|██▍       | 96550/400000 [00:15<00:42, 7184.99it/s] 24%|██▍       | 97269/400000 [00:15<00:42, 7041.61it/s] 24%|██▍       | 97975/400000 [00:15<00:42, 7041.49it/s] 25%|██▍       | 98688/400000 [00:15<00:42, 7067.22it/s] 25%|██▍       | 99398/400000 [00:15<00:42, 7075.64it/s] 25%|██▌       | 100106/400000 [00:15<00:44, 6790.07it/s] 25%|██▌       | 100788/400000 [00:15<00:44, 6742.45it/s] 25%|██▌       | 101465/400000 [00:15<00:44, 6638.31it/s] 26%|██▌       | 102131/400000 [00:15<00:45, 6597.03it/s] 26%|██▌       | 102794/400000 [00:15<00:44, 6605.24it/s] 26%|██▌       | 103514/400000 [00:16<00:43, 6771.87it/s] 26%|██▌       | 104231/400000 [00:16<00:42, 6880.02it/s] 26%|██▌       | 104943/400000 [00:16<00:42, 6950.13it/s] 26%|██▋       | 105662/400000 [00:16<00:41, 7019.21it/s] 27%|██▋       | 106365/400000 [00:16<00:41, 7002.38it/s] 27%|██▋       | 107066/400000 [00:16<00:42, 6851.25it/s] 27%|██▋       | 107779/400000 [00:16<00:42, 6931.63it/s] 27%|██▋       | 108490/400000 [00:16<00:41, 6982.25it/s] 27%|██▋       | 109222/400000 [00:16<00:41, 7078.36it/s] 27%|██▋       | 109953/400000 [00:16<00:40, 7146.01it/s] 28%|██▊       | 110669/400000 [00:17<00:40, 7064.90it/s] 28%|██▊       | 111377/400000 [00:17<00:41, 7012.68it/s] 28%|██▊       | 112109/400000 [00:17<00:40, 7101.73it/s] 28%|██▊       | 112838/400000 [00:17<00:40, 7155.26it/s] 28%|██▊       | 113572/400000 [00:17<00:39, 7208.80it/s] 29%|██▊       | 114294/400000 [00:17<00:41, 6915.64it/s] 29%|██▊       | 114989/400000 [00:17<00:41, 6786.95it/s] 29%|██▉       | 115671/400000 [00:17<00:42, 6678.14it/s] 29%|██▉       | 116392/400000 [00:17<00:41, 6826.62it/s] 29%|██▉       | 117078/400000 [00:18<00:42, 6686.66it/s] 29%|██▉       | 117749/400000 [00:18<00:43, 6553.14it/s] 30%|██▉       | 118407/400000 [00:18<00:43, 6542.33it/s] 30%|██▉       | 119135/400000 [00:18<00:41, 6745.83it/s] 30%|██▉       | 119813/400000 [00:18<00:41, 6720.16it/s] 30%|███       | 120537/400000 [00:18<00:40, 6866.95it/s] 30%|███       | 121258/400000 [00:18<00:40, 6963.66it/s] 30%|███       | 121983/400000 [00:18<00:39, 7046.68it/s] 31%|███       | 122690/400000 [00:18<00:40, 6874.79it/s] 31%|███       | 123380/400000 [00:18<00:40, 6782.04it/s] 31%|███       | 124060/400000 [00:19<00:42, 6523.18it/s] 31%|███       | 124716/400000 [00:19<00:42, 6516.66it/s] 31%|███▏      | 125392/400000 [00:19<00:41, 6587.64it/s] 32%|███▏      | 126127/400000 [00:19<00:40, 6798.54it/s] 32%|███▏      | 126856/400000 [00:19<00:39, 6936.90it/s] 32%|███▏      | 127553/400000 [00:19<00:40, 6767.46it/s] 32%|███▏      | 128233/400000 [00:19<00:42, 6372.55it/s] 32%|███▏      | 128923/400000 [00:19<00:41, 6521.67it/s] 32%|███▏      | 129658/400000 [00:19<00:40, 6747.89it/s] 33%|███▎      | 130393/400000 [00:20<00:38, 6916.20it/s] 33%|███▎      | 131118/400000 [00:20<00:38, 7011.01it/s] 33%|███▎      | 131849/400000 [00:20<00:37, 7095.97it/s] 33%|███▎      | 132562/400000 [00:20<00:37, 7083.31it/s] 33%|███▎      | 133299/400000 [00:20<00:37, 7166.44it/s] 34%|███▎      | 134038/400000 [00:20<00:36, 7229.89it/s] 34%|███▎      | 134763/400000 [00:20<00:37, 7057.59it/s] 34%|███▍      | 135474/400000 [00:20<00:37, 7071.35it/s] 34%|███▍      | 136196/400000 [00:20<00:37, 7113.56it/s] 34%|███▍      | 136909/400000 [00:20<00:37, 7050.65it/s] 34%|███▍      | 137636/400000 [00:21<00:36, 7113.29it/s] 35%|███▍      | 138364/400000 [00:21<00:36, 7162.11it/s] 35%|███▍      | 139092/400000 [00:21<00:36, 7197.06it/s] 35%|███▍      | 139813/400000 [00:21<00:36, 7054.87it/s] 35%|███▌      | 140520/400000 [00:21<00:36, 7028.41it/s] 35%|███▌      | 141224/400000 [00:21<00:37, 6955.78it/s] 35%|███▌      | 141941/400000 [00:21<00:36, 7016.18it/s] 36%|███▌      | 142676/400000 [00:21<00:36, 7112.45it/s] 36%|███▌      | 143411/400000 [00:21<00:35, 7181.57it/s] 36%|███▌      | 144147/400000 [00:21<00:35, 7233.45it/s] 36%|███▌      | 144878/400000 [00:22<00:35, 7253.36it/s] 36%|███▋      | 145604/400000 [00:22<00:35, 7150.89it/s] 37%|███▋      | 146320/400000 [00:22<00:36, 6954.19it/s] 37%|███▋      | 147018/400000 [00:22<00:37, 6793.26it/s] 37%|███▋      | 147700/400000 [00:22<00:37, 6691.03it/s] 37%|███▋      | 148371/400000 [00:22<00:37, 6631.14it/s] 37%|███▋      | 149036/400000 [00:22<00:37, 6623.26it/s] 37%|███▋      | 149703/400000 [00:22<00:37, 6633.93it/s] 38%|███▊      | 150413/400000 [00:22<00:36, 6766.66it/s] 38%|███▊      | 151139/400000 [00:22<00:36, 6906.15it/s] 38%|███▊      | 151832/400000 [00:23<00:36, 6837.68it/s] 38%|███▊      | 152517/400000 [00:23<00:37, 6554.77it/s] 38%|███▊      | 153176/400000 [00:23<00:38, 6398.80it/s] 38%|███▊      | 153820/400000 [00:23<00:38, 6397.76it/s] 39%|███▊      | 154491/400000 [00:23<00:37, 6487.26it/s] 39%|███▉      | 155206/400000 [00:23<00:36, 6671.78it/s] 39%|███▉      | 155927/400000 [00:23<00:35, 6823.28it/s] 39%|███▉      | 156660/400000 [00:23<00:34, 6965.64it/s] 39%|███▉      | 157376/400000 [00:23<00:34, 7021.78it/s] 40%|███▉      | 158102/400000 [00:23<00:34, 7089.92it/s] 40%|███▉      | 158813/400000 [00:24<00:34, 7008.70it/s] 40%|███▉      | 159516/400000 [00:24<00:34, 7001.16it/s] 40%|████      | 160249/400000 [00:24<00:33, 7095.52it/s] 40%|████      | 160977/400000 [00:24<00:33, 7147.07it/s] 40%|████      | 161700/400000 [00:24<00:33, 7168.82it/s] 41%|████      | 162418/400000 [00:24<00:34, 6915.39it/s] 41%|████      | 163112/400000 [00:24<00:35, 6646.24it/s] 41%|████      | 163781/400000 [00:24<00:35, 6582.74it/s] 41%|████      | 164469/400000 [00:24<00:35, 6668.24it/s] 41%|████▏     | 165206/400000 [00:25<00:34, 6863.09it/s] 41%|████▏     | 165942/400000 [00:25<00:33, 7002.79it/s] 42%|████▏     | 166660/400000 [00:25<00:33, 7053.32it/s] 42%|████▏     | 167368/400000 [00:25<00:33, 7006.74it/s] 42%|████▏     | 168110/400000 [00:25<00:32, 7125.58it/s] 42%|████▏     | 168840/400000 [00:25<00:32, 7175.43it/s] 42%|████▏     | 169559/400000 [00:25<00:33, 6958.76it/s] 43%|████▎     | 170258/400000 [00:25<00:33, 6810.74it/s] 43%|████▎     | 170942/400000 [00:25<00:34, 6688.42it/s] 43%|████▎     | 171613/400000 [00:25<00:35, 6474.10it/s] 43%|████▎     | 172266/400000 [00:26<00:35, 6488.24it/s] 43%|████▎     | 172963/400000 [00:26<00:34, 6624.55it/s] 43%|████▎     | 173628/400000 [00:26<00:34, 6541.19it/s] 44%|████▎     | 174338/400000 [00:26<00:33, 6697.71it/s] 44%|████▍     | 175068/400000 [00:26<00:32, 6866.00it/s] 44%|████▍     | 175758/400000 [00:26<00:33, 6699.01it/s] 44%|████▍     | 176475/400000 [00:26<00:32, 6831.07it/s] 44%|████▍     | 177172/400000 [00:26<00:32, 6865.77it/s] 44%|████▍     | 177913/400000 [00:26<00:31, 7018.92it/s] 45%|████▍     | 178639/400000 [00:26<00:31, 7089.06it/s] 45%|████▍     | 179371/400000 [00:27<00:30, 7156.70it/s] 45%|████▌     | 180090/400000 [00:27<00:30, 7164.65it/s] 45%|████▌     | 180808/400000 [00:27<00:31, 7009.16it/s] 45%|████▌     | 181511/400000 [00:27<00:31, 6847.01it/s] 46%|████▌     | 182198/400000 [00:27<00:32, 6763.56it/s] 46%|████▌     | 182876/400000 [00:27<00:33, 6573.99it/s] 46%|████▌     | 183536/400000 [00:27<00:33, 6532.85it/s] 46%|████▌     | 184191/400000 [00:27<00:34, 6290.58it/s] 46%|████▌     | 184916/400000 [00:27<00:32, 6549.63it/s] 46%|████▋     | 185635/400000 [00:28<00:31, 6727.25it/s] 47%|████▋     | 186359/400000 [00:28<00:31, 6873.27it/s] 47%|████▋     | 187075/400000 [00:28<00:30, 6954.40it/s] 47%|████▋     | 187799/400000 [00:28<00:30, 7034.87it/s] 47%|████▋     | 188505/400000 [00:28<00:30, 6878.01it/s] 47%|████▋     | 189196/400000 [00:28<00:31, 6778.18it/s] 47%|████▋     | 189876/400000 [00:28<00:31, 6715.33it/s] 48%|████▊     | 190550/400000 [00:28<00:31, 6609.78it/s] 48%|████▊     | 191213/400000 [00:28<00:31, 6573.14it/s] 48%|████▊     | 191927/400000 [00:28<00:30, 6731.64it/s] 48%|████▊     | 192602/400000 [00:29<00:31, 6680.07it/s] 48%|████▊     | 193278/400000 [00:29<00:30, 6700.88it/s] 48%|████▊     | 193994/400000 [00:29<00:30, 6831.83it/s] 49%|████▊     | 194731/400000 [00:29<00:29, 6982.78it/s] 49%|████▉     | 195431/400000 [00:29<00:29, 6922.06it/s] 49%|████▉     | 196125/400000 [00:29<00:29, 6909.83it/s] 49%|████▉     | 196832/400000 [00:29<00:29, 6956.54it/s] 49%|████▉     | 197541/400000 [00:29<00:28, 6995.20it/s] 50%|████▉     | 198268/400000 [00:29<00:28, 7073.90it/s] 50%|████▉     | 198999/400000 [00:29<00:28, 7140.32it/s] 50%|████▉     | 199714/400000 [00:30<00:28, 6941.27it/s] 50%|█████     | 200436/400000 [00:30<00:28, 7021.27it/s] 50%|█████     | 201140/400000 [00:30<00:28, 6961.79it/s] 50%|█████     | 201863/400000 [00:30<00:28, 7037.54it/s] 51%|█████     | 202568/400000 [00:30<00:28, 7023.21it/s] 51%|█████     | 203271/400000 [00:30<00:28, 6897.70it/s] 51%|█████     | 204013/400000 [00:30<00:27, 7046.36it/s] 51%|█████     | 204723/400000 [00:30<00:27, 7060.06it/s] 51%|█████▏    | 205431/400000 [00:30<00:28, 6928.26it/s] 52%|█████▏    | 206148/400000 [00:30<00:27, 6997.65it/s] 52%|█████▏    | 206864/400000 [00:31<00:27, 7045.19it/s] 52%|█████▏    | 207570/400000 [00:31<00:27, 7041.90it/s] 52%|█████▏    | 208291/400000 [00:31<00:27, 7089.28it/s] 52%|█████▏    | 209001/400000 [00:31<00:27, 7051.48it/s] 52%|█████▏    | 209707/400000 [00:31<00:27, 6938.38it/s] 53%|█████▎    | 210415/400000 [00:31<00:27, 6978.86it/s] 53%|█████▎    | 211143/400000 [00:31<00:26, 7064.31it/s] 53%|█████▎    | 211861/400000 [00:31<00:26, 7096.02it/s] 53%|█████▎    | 212572/400000 [00:31<00:26, 6984.75it/s] 53%|█████▎    | 213272/400000 [00:32<00:27, 6821.38it/s] 53%|█████▎    | 213956/400000 [00:32<00:27, 6676.90it/s] 54%|█████▎    | 214626/400000 [00:32<00:28, 6497.03it/s] 54%|█████▍    | 215348/400000 [00:32<00:27, 6697.71it/s] 54%|█████▍    | 216078/400000 [00:32<00:26, 6866.89it/s] 54%|█████▍    | 216797/400000 [00:32<00:26, 6959.55it/s] 54%|█████▍    | 217496/400000 [00:32<00:26, 6962.52it/s] 55%|█████▍    | 218195/400000 [00:32<00:26, 6940.99it/s] 55%|█████▍    | 218891/400000 [00:32<00:26, 6845.70it/s] 55%|█████▍    | 219618/400000 [00:32<00:25, 6967.44it/s] 55%|█████▌    | 220337/400000 [00:33<00:25, 7031.90it/s] 55%|█████▌    | 221070/400000 [00:33<00:25, 7116.80it/s] 55%|█████▌    | 221796/400000 [00:33<00:24, 7156.66it/s] 56%|█████▌    | 222513/400000 [00:33<00:25, 7073.08it/s] 56%|█████▌    | 223222/400000 [00:33<00:25, 6946.47it/s] 56%|█████▌    | 223918/400000 [00:33<00:26, 6771.58it/s] 56%|█████▌    | 224641/400000 [00:33<00:25, 6901.30it/s] 56%|█████▋    | 225355/400000 [00:33<00:25, 6970.27it/s] 57%|█████▋    | 226054/400000 [00:33<00:25, 6872.44it/s] 57%|█████▋    | 226743/400000 [00:33<00:26, 6617.94it/s] 57%|█████▋    | 227408/400000 [00:34<00:26, 6560.69it/s] 57%|█████▋    | 228067/400000 [00:34<00:26, 6516.39it/s] 57%|█████▋    | 228721/400000 [00:34<00:26, 6519.24it/s] 57%|█████▋    | 229375/400000 [00:34<00:26, 6458.12it/s] 58%|█████▊    | 230076/400000 [00:34<00:25, 6613.12it/s] 58%|█████▊    | 230756/400000 [00:34<00:25, 6667.14it/s] 58%|█████▊    | 231424/400000 [00:34<00:25, 6527.34it/s] 58%|█████▊    | 232151/400000 [00:34<00:24, 6731.93it/s] 58%|█████▊    | 232877/400000 [00:34<00:24, 6881.62it/s] 58%|█████▊    | 233585/400000 [00:34<00:23, 6939.90it/s] 59%|█████▊    | 234281/400000 [00:35<00:23, 6936.82it/s] 59%|█████▊    | 234977/400000 [00:35<00:24, 6868.02it/s] 59%|█████▉    | 235665/400000 [00:35<00:24, 6836.74it/s] 59%|█████▉    | 236391/400000 [00:35<00:23, 6958.19it/s] 59%|█████▉    | 237093/400000 [00:35<00:23, 6975.73it/s] 59%|█████▉    | 237810/400000 [00:35<00:23, 7028.62it/s] 60%|█████▉    | 238529/400000 [00:35<00:22, 7073.30it/s] 60%|█████▉    | 239260/400000 [00:35<00:22, 7140.26it/s] 60%|█████▉    | 239975/400000 [00:35<00:22, 7009.69it/s] 60%|██████    | 240677/400000 [00:36<00:23, 6838.90it/s] 60%|██████    | 241363/400000 [00:36<00:23, 6743.56it/s] 61%|██████    | 242039/400000 [00:36<00:23, 6665.22it/s] 61%|██████    | 242730/400000 [00:36<00:23, 6736.76it/s] 61%|██████    | 243439/400000 [00:36<00:22, 6838.09it/s] 61%|██████    | 244126/400000 [00:36<00:22, 6844.87it/s] 61%|██████    | 244828/400000 [00:36<00:22, 6895.80it/s] 61%|██████▏   | 245519/400000 [00:36<00:22, 6839.20it/s] 62%|██████▏   | 246244/400000 [00:36<00:22, 6956.42it/s] 62%|██████▏   | 246961/400000 [00:36<00:21, 7016.92it/s] 62%|██████▏   | 247687/400000 [00:37<00:21, 7087.83it/s] 62%|██████▏   | 248397/400000 [00:37<00:21, 7045.16it/s] 62%|██████▏   | 249128/400000 [00:37<00:21, 7120.71it/s] 62%|██████▏   | 249857/400000 [00:37<00:20, 7168.85it/s] 63%|██████▎   | 250587/400000 [00:37<00:20, 7205.78it/s] 63%|██████▎   | 251308/400000 [00:37<00:20, 7112.63it/s] 63%|██████▎   | 252020/400000 [00:37<00:21, 6899.16it/s] 63%|██████▎   | 252712/400000 [00:37<00:22, 6649.09it/s] 63%|██████▎   | 253381/400000 [00:37<00:22, 6586.76it/s] 64%|██████▎   | 254105/400000 [00:37<00:21, 6767.62it/s] 64%|██████▎   | 254829/400000 [00:38<00:21, 6902.13it/s] 64%|██████▍   | 255550/400000 [00:38<00:20, 6989.39it/s] 64%|██████▍   | 256273/400000 [00:38<00:20, 7058.54it/s] 64%|██████▍   | 256981/400000 [00:38<00:20, 6996.01it/s] 64%|██████▍   | 257688/400000 [00:38<00:20, 7015.51it/s] 65%|██████▍   | 258391/400000 [00:38<00:20, 6788.26it/s] 65%|██████▍   | 259073/400000 [00:38<00:20, 6716.74it/s] 65%|██████▍   | 259747/400000 [00:38<00:21, 6611.96it/s] 65%|██████▌   | 260410/400000 [00:38<00:21, 6612.92it/s] 65%|██████▌   | 261073/400000 [00:38<00:21, 6379.12it/s] 65%|██████▌   | 261803/400000 [00:39<00:20, 6609.36it/s] 66%|██████▌   | 262468/400000 [00:39<00:21, 6545.42it/s] 66%|██████▌   | 263195/400000 [00:39<00:20, 6744.76it/s] 66%|██████▌   | 263890/400000 [00:39<00:20, 6804.94it/s] 66%|██████▌   | 264585/400000 [00:39<00:19, 6845.60it/s] 66%|██████▋   | 265272/400000 [00:39<00:19, 6797.66it/s] 67%|██████▋   | 266001/400000 [00:39<00:19, 6937.39it/s] 67%|██████▋   | 266713/400000 [00:39<00:19, 6990.06it/s] 67%|██████▋   | 267414/400000 [00:39<00:19, 6922.59it/s] 67%|██████▋   | 268108/400000 [00:40<00:19, 6745.54it/s] 67%|██████▋   | 268785/400000 [00:40<00:19, 6701.36it/s] 67%|██████▋   | 269457/400000 [00:40<00:19, 6668.38it/s] 68%|██████▊   | 270125/400000 [00:40<00:19, 6516.46it/s] 68%|██████▊   | 270815/400000 [00:40<00:19, 6625.32it/s] 68%|██████▊   | 271530/400000 [00:40<00:18, 6774.29it/s] 68%|██████▊   | 272210/400000 [00:40<00:18, 6771.51it/s] 68%|██████▊   | 272889/400000 [00:40<00:18, 6735.11it/s] 68%|██████▊   | 273630/400000 [00:40<00:18, 6921.44it/s] 69%|██████▊   | 274325/400000 [00:40<00:18, 6890.96it/s] 69%|██████▉   | 275039/400000 [00:41<00:17, 6956.88it/s] 69%|██████▉   | 275736/400000 [00:41<00:18, 6857.75it/s] 69%|██████▉   | 276423/400000 [00:41<00:18, 6750.94it/s] 69%|██████▉   | 277100/400000 [00:41<00:18, 6752.76it/s] 69%|██████▉   | 277777/400000 [00:41<00:18, 6702.50it/s] 70%|██████▉   | 278451/400000 [00:41<00:18, 6712.86it/s] 70%|██████▉   | 279173/400000 [00:41<00:17, 6856.80it/s] 70%|██████▉   | 279891/400000 [00:41<00:17, 6949.40it/s] 70%|███████   | 280587/400000 [00:41<00:17, 6944.10it/s] 70%|███████   | 281305/400000 [00:41<00:16, 7011.38it/s] 71%|███████   | 282018/400000 [00:42<00:16, 7044.60it/s] 71%|███████   | 282723/400000 [00:42<00:16, 6990.76it/s] 71%|███████   | 283448/400000 [00:42<00:16, 7065.31it/s] 71%|███████   | 284156/400000 [00:42<00:16, 6950.43it/s] 71%|███████   | 284875/400000 [00:42<00:16, 7017.78it/s] 71%|███████▏  | 285578/400000 [00:42<00:16, 6974.33it/s] 72%|███████▏  | 286292/400000 [00:42<00:16, 7021.14it/s] 72%|███████▏  | 286995/400000 [00:42<00:16, 6823.04it/s] 72%|███████▏  | 287679/400000 [00:42<00:16, 6687.82it/s] 72%|███████▏  | 288386/400000 [00:42<00:16, 6795.70it/s] 72%|███████▏  | 289097/400000 [00:43<00:16, 6884.38it/s] 72%|███████▏  | 289819/400000 [00:43<00:15, 6980.84it/s] 73%|███████▎  | 290547/400000 [00:43<00:15, 7067.68it/s] 73%|███████▎  | 291255/400000 [00:43<00:15, 6952.79it/s] 73%|███████▎  | 291968/400000 [00:43<00:15, 7002.81it/s] 73%|███████▎  | 292670/400000 [00:43<00:15, 6994.76it/s] 73%|███████▎  | 293388/400000 [00:43<00:15, 7046.85it/s] 74%|███████▎  | 294105/400000 [00:43<00:14, 7082.30it/s] 74%|███████▎  | 294833/400000 [00:43<00:14, 7139.54it/s] 74%|███████▍  | 295548/400000 [00:43<00:14, 6964.72it/s] 74%|███████▍  | 296275/400000 [00:44<00:14, 7051.23it/s] 74%|███████▍  | 296982/400000 [00:44<00:14, 6939.26it/s] 74%|███████▍  | 297693/400000 [00:44<00:14, 6988.13it/s] 75%|███████▍  | 298413/400000 [00:44<00:14, 7047.57it/s] 75%|███████▍  | 299119/400000 [00:44<00:14, 6918.39it/s] 75%|███████▍  | 299812/400000 [00:44<00:15, 6609.54it/s] 75%|███████▌  | 300477/400000 [00:44<00:15, 6588.27it/s] 75%|███████▌  | 301139/400000 [00:44<00:15, 6567.65it/s] 75%|███████▌  | 301798/400000 [00:44<00:14, 6573.08it/s] 76%|███████▌  | 302457/400000 [00:45<00:14, 6557.70it/s] 76%|███████▌  | 303166/400000 [00:45<00:14, 6707.70it/s] 76%|███████▌  | 303844/400000 [00:45<00:14, 6727.39it/s] 76%|███████▌  | 304566/400000 [00:45<00:13, 6865.79it/s] 76%|███████▋  | 305294/400000 [00:45<00:13, 6979.94it/s] 76%|███████▋  | 305994/400000 [00:45<00:14, 6667.47it/s] 77%|███████▋  | 306665/400000 [00:45<00:14, 6589.39it/s] 77%|███████▋  | 307327/400000 [00:45<00:14, 6597.92it/s] 77%|███████▋  | 307989/400000 [00:45<00:14, 6526.95it/s] 77%|███████▋  | 308720/400000 [00:45<00:13, 6743.54it/s] 77%|███████▋  | 309445/400000 [00:46<00:13, 6887.36it/s] 78%|███████▊  | 310137/400000 [00:46<00:13, 6721.09it/s] 78%|███████▊  | 310812/400000 [00:46<00:13, 6605.79it/s] 78%|███████▊  | 311490/400000 [00:46<00:13, 6655.58it/s] 78%|███████▊  | 312215/400000 [00:46<00:12, 6820.52it/s] 78%|███████▊  | 312900/400000 [00:46<00:12, 6750.96it/s] 78%|███████▊  | 313577/400000 [00:46<00:12, 6682.61it/s] 79%|███████▊  | 314300/400000 [00:46<00:12, 6837.69it/s] 79%|███████▉  | 315018/400000 [00:46<00:12, 6934.83it/s] 79%|███████▉  | 315714/400000 [00:46<00:12, 6798.88it/s] 79%|███████▉  | 316396/400000 [00:47<00:12, 6687.00it/s] 79%|███████▉  | 317067/400000 [00:47<00:12, 6452.62it/s] 79%|███████▉  | 317786/400000 [00:47<00:12, 6655.87it/s] 80%|███████▉  | 318480/400000 [00:47<00:12, 6736.50it/s] 80%|███████▉  | 319206/400000 [00:47<00:11, 6883.89it/s] 80%|███████▉  | 319910/400000 [00:47<00:11, 6928.19it/s] 80%|████████  | 320605/400000 [00:47<00:11, 6828.86it/s] 80%|████████  | 321290/400000 [00:47<00:11, 6646.22it/s] 81%|████████  | 322010/400000 [00:47<00:11, 6801.79it/s] 81%|████████  | 322734/400000 [00:48<00:11, 6927.50it/s] 81%|████████  | 323451/400000 [00:48<00:10, 6997.36it/s] 81%|████████  | 324155/400000 [00:48<00:10, 7009.34it/s] 81%|████████  | 324886/400000 [00:48<00:10, 7095.67it/s] 81%|████████▏ | 325597/400000 [00:48<00:10, 7028.28it/s] 82%|████████▏ | 326318/400000 [00:48<00:10, 7081.30it/s] 82%|████████▏ | 327028/400000 [00:48<00:10, 7086.02it/s] 82%|████████▏ | 327738/400000 [00:48<00:10, 7041.85it/s] 82%|████████▏ | 328443/400000 [00:48<00:10, 6807.79it/s] 82%|████████▏ | 329126/400000 [00:48<00:10, 6700.63it/s] 82%|████████▏ | 329798/400000 [00:49<00:10, 6522.46it/s] 83%|████████▎ | 330478/400000 [00:49<00:10, 6603.07it/s] 83%|████████▎ | 331159/400000 [00:49<00:10, 6662.38it/s] 83%|████████▎ | 331827/400000 [00:49<00:10, 6661.93it/s] 83%|████████▎ | 332495/400000 [00:49<00:10, 6640.22it/s] 83%|████████▎ | 333217/400000 [00:49<00:09, 6803.47it/s] 83%|████████▎ | 333902/400000 [00:49<00:09, 6814.80it/s] 84%|████████▎ | 334626/400000 [00:49<00:09, 6935.99it/s] 84%|████████▍ | 335321/400000 [00:49<00:09, 6909.01it/s] 84%|████████▍ | 336041/400000 [00:49<00:09, 6991.12it/s] 84%|████████▍ | 336765/400000 [00:50<00:08, 7061.70it/s] 84%|████████▍ | 337472/400000 [00:50<00:08, 7060.02it/s] 85%|████████▍ | 338179/400000 [00:50<00:09, 6685.37it/s] 85%|████████▍ | 338853/400000 [00:50<00:09, 6663.67it/s] 85%|████████▍ | 339547/400000 [00:50<00:08, 6742.72it/s] 85%|████████▌ | 340230/400000 [00:50<00:08, 6767.98it/s] 85%|████████▌ | 340946/400000 [00:50<00:08, 6880.49it/s] 85%|████████▌ | 341636/400000 [00:50<00:08, 6794.82it/s] 86%|████████▌ | 342317/400000 [00:50<00:08, 6642.42it/s] 86%|████████▌ | 343006/400000 [00:50<00:08, 6712.58it/s] 86%|████████▌ | 343686/400000 [00:51<00:08, 6737.92it/s] 86%|████████▌ | 344404/400000 [00:51<00:08, 6863.39it/s] 86%|████████▋ | 345131/400000 [00:51<00:07, 6979.49it/s] 86%|████████▋ | 345855/400000 [00:51<00:07, 7053.72it/s] 87%|████████▋ | 346562/400000 [00:51<00:07, 6938.89it/s] 87%|████████▋ | 347279/400000 [00:51<00:07, 7006.44it/s] 87%|████████▋ | 347981/400000 [00:51<00:07, 6872.20it/s] 87%|████████▋ | 348692/400000 [00:51<00:07, 6941.82it/s] 87%|████████▋ | 349421/400000 [00:51<00:07, 7041.91it/s] 88%|████████▊ | 350127/400000 [00:51<00:07, 7001.04it/s] 88%|████████▊ | 350828/400000 [00:52<00:07, 6903.43it/s] 88%|████████▊ | 351534/400000 [00:52<00:06, 6947.77it/s] 88%|████████▊ | 352230/400000 [00:52<00:06, 6915.70it/s] 88%|████████▊ | 352949/400000 [00:52<00:06, 6993.64it/s] 88%|████████▊ | 353669/400000 [00:52<00:06, 7051.66it/s] 89%|████████▊ | 354375/400000 [00:52<00:06, 6973.85it/s] 89%|████████▉ | 355073/400000 [00:52<00:06, 6720.04it/s] 89%|████████▉ | 355764/400000 [00:52<00:06, 6773.73it/s] 89%|████████▉ | 356497/400000 [00:52<00:06, 6929.22it/s] 89%|████████▉ | 357197/400000 [00:53<00:06, 6949.76it/s] 89%|████████▉ | 357907/400000 [00:53<00:06, 6993.61it/s] 90%|████████▉ | 358608/400000 [00:53<00:05, 6997.75it/s] 90%|████████▉ | 359309/400000 [00:53<00:05, 6877.15it/s] 90%|█████████ | 360016/400000 [00:53<00:05, 6933.07it/s] 90%|█████████ | 360736/400000 [00:53<00:05, 7010.37it/s] 90%|█████████ | 361447/400000 [00:53<00:05, 7037.75it/s] 91%|█████████ | 362153/400000 [00:53<00:05, 7041.64it/s] 91%|█████████ | 362881/400000 [00:53<00:05, 7111.20it/s] 91%|█████████ | 363593/400000 [00:53<00:05, 6995.77it/s] 91%|█████████ | 364312/400000 [00:54<00:05, 7051.75it/s] 91%|█████████▏| 365031/400000 [00:54<00:04, 7090.08it/s] 91%|█████████▏| 365741/400000 [00:54<00:04, 6937.01it/s] 92%|█████████▏| 366447/400000 [00:54<00:04, 6971.15it/s] 92%|█████████▏| 367165/400000 [00:54<00:04, 7026.57it/s] 92%|█████████▏| 367869/400000 [00:54<00:04, 6909.12it/s] 92%|█████████▏| 368588/400000 [00:54<00:04, 6988.60it/s] 92%|█████████▏| 369296/400000 [00:54<00:04, 7015.13it/s] 93%|█████████▎| 370017/400000 [00:54<00:04, 7065.28it/s] 93%|█████████▎| 370745/400000 [00:54<00:04, 7126.79it/s] 93%|█████████▎| 371461/400000 [00:55<00:03, 7136.15it/s] 93%|█████████▎| 372175/400000 [00:55<00:03, 6989.20it/s] 93%|█████████▎| 372879/400000 [00:55<00:03, 7002.15it/s] 93%|█████████▎| 373607/400000 [00:55<00:03, 7082.11it/s] 94%|█████████▎| 374323/400000 [00:55<00:03, 7104.27it/s] 94%|█████████▍| 375037/400000 [00:55<00:03, 7112.94it/s] 94%|█████████▍| 375763/400000 [00:55<00:03, 7153.71it/s] 94%|█████████▍| 376479/400000 [00:55<00:03, 6986.99it/s] 94%|█████████▍| 377179/400000 [00:55<00:03, 6833.43it/s] 94%|█████████▍| 377899/400000 [00:55<00:03, 6936.68it/s] 95%|█████████▍| 378611/400000 [00:56<00:03, 6989.65it/s] 95%|█████████▍| 379338/400000 [00:56<00:02, 7070.39it/s] 95%|█████████▌| 380047/400000 [00:56<00:02, 7056.43it/s] 95%|█████████▌| 380754/400000 [00:56<00:02, 6880.21it/s] 95%|█████████▌| 381470/400000 [00:56<00:02, 6960.24it/s] 96%|█████████▌| 382179/400000 [00:56<00:02, 6996.68it/s] 96%|█████████▌| 382884/400000 [00:56<00:02, 7011.06it/s] 96%|█████████▌| 383605/400000 [00:56<00:02, 7068.71it/s] 96%|█████████▌| 384325/400000 [00:56<00:02, 7107.11it/s] 96%|█████████▋| 385037/400000 [00:56<00:02, 7035.34it/s] 96%|█████████▋| 385764/400000 [00:57<00:02, 7101.74it/s] 97%|█████████▋| 386487/400000 [00:57<00:01, 7138.06it/s] 97%|█████████▋| 387202/400000 [00:57<00:01, 7089.93it/s] 97%|█████████▋| 387921/400000 [00:57<00:01, 7117.50it/s] 97%|█████████▋| 388634/400000 [00:57<00:01, 7051.12it/s] 97%|█████████▋| 389340/400000 [00:57<00:01, 6734.26it/s] 98%|█████████▊| 390017/400000 [00:57<00:01, 6656.85it/s] 98%|█████████▊| 390686/400000 [00:57<00:01, 6533.40it/s] 98%|█████████▊| 391342/400000 [00:57<00:01, 6508.43it/s] 98%|█████████▊| 391995/400000 [00:58<00:01, 6482.13it/s] 98%|█████████▊| 392722/400000 [00:58<00:01, 6698.46it/s] 98%|█████████▊| 393395/400000 [00:58<00:00, 6615.33it/s] 99%|█████████▊| 394102/400000 [00:58<00:00, 6743.45it/s] 99%|█████████▊| 394798/400000 [00:58<00:00, 6806.03it/s] 99%|█████████▉| 395481/400000 [00:58<00:00, 6668.46it/s] 99%|█████████▉| 396150/400000 [00:58<00:00, 6604.27it/s] 99%|█████████▉| 396812/400000 [00:58<00:00, 6538.31it/s] 99%|█████████▉| 397467/400000 [00:58<00:00, 6407.07it/s]100%|█████████▉| 398110/400000 [00:58<00:00, 6411.62it/s]100%|█████████▉| 398761/400000 [00:59<00:00, 6438.56it/s]100%|█████████▉| 399406/400000 [00:59<00:00, 6326.66it/s]100%|█████████▉| 399999/400000 [00:59<00:00, 6751.52it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f47cf0670f0>, <torchtext.data.dataset.TabularDataset object at 0x7f47cf067240>, <torchtext.vocab.Vocab object at 0x7f47cf067160>), {}) 

