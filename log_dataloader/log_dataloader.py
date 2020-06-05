
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd67c613373df800885b9f1e6941d6f5879aa2c04', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d67c613373df800885b9f1e6941d6f5879aa2c04

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f04e54ce378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f04e54ce378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f05575230d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f05575230d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f04eecb49d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f04eecb49d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0504e078c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0504e078c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0504e078c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 11%|█         | 1081344/9912422 [00:00<00:00, 8835074.44it/s] 45%|████▌     | 4489216/9912422 [00:00<00:00, 10336413.15it/s] 78%|███████▊  | 7733248/9912422 [00:00<00:00, 12516748.80it/s]9920512it [00:00, 12312315.67it/s]                             
0it [00:00, ?it/s]32768it [00:00, 566293.86it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s] 36%|███▌      | 589824/1648877 [00:00<00:00, 5386019.66it/s]1654784it [00:00, 5248728.89it/s]                            
0it [00:00, ?it/s]8192it [00:00, 162911.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04e8462a20>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04e8462e10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0504e07510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0504e07510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0504e07510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:08,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:08,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:07,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:07,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:47,  3.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:47,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:33,  4.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:33,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:23,  6.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:23,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:22,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:22,  6.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:11, 11.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:08, 14.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 27.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 32.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 37.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 40.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 44.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 47.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 49.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 50.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 50.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 53.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 54.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 55.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 55.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 55.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 55.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 55.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 56.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 56.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 56.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 56.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 56.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 56.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 56.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 56.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 56.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 56.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 56.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 56.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.42s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 56.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 56.03 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.73s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.42s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 56.03 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.73s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.73s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.28 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.73s/ url]
0 examples [00:00, ? examples/s]2020-06-05 00:09:38.890168: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-05 00:09:38.904743: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-06-05 00:09:38.904985: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56084c2e4b80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-05 00:09:38.905004: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
28 examples [00:00, 278.68 examples/s]119 examples [00:00, 351.81 examples/s]212 examples [00:00, 432.14 examples/s]298 examples [00:00, 507.61 examples/s]386 examples [00:00, 581.27 examples/s]475 examples [00:00, 648.28 examples/s]560 examples [00:00, 696.82 examples/s]655 examples [00:00, 757.10 examples/s]748 examples [00:00, 800.23 examples/s]837 examples [00:01, 822.83 examples/s]929 examples [00:01, 849.05 examples/s]1023 examples [00:01, 872.68 examples/s]1116 examples [00:01, 886.99 examples/s]1210 examples [00:01, 902.09 examples/s]1302 examples [00:01, 882.65 examples/s]1406 examples [00:01, 922.41 examples/s]1500 examples [00:01, 905.19 examples/s]1597 examples [00:01, 921.23 examples/s]1690 examples [00:01, 918.89 examples/s]1783 examples [00:02, 911.04 examples/s]1875 examples [00:02, 904.82 examples/s]1966 examples [00:02, 880.74 examples/s]2056 examples [00:02, 886.08 examples/s]2150 examples [00:02, 901.40 examples/s]2241 examples [00:02, 900.89 examples/s]2332 examples [00:02, 891.27 examples/s]2428 examples [00:02, 909.80 examples/s]2520 examples [00:02, 907.18 examples/s]2611 examples [00:02, 902.75 examples/s]2706 examples [00:03, 915.59 examples/s]2798 examples [00:03, 915.78 examples/s]2890 examples [00:03, 907.93 examples/s]2983 examples [00:03, 914.20 examples/s]3075 examples [00:03, 895.30 examples/s]3170 examples [00:03, 908.81 examples/s]3263 examples [00:03, 913.54 examples/s]3355 examples [00:03, 910.63 examples/s]3447 examples [00:03, 908.36 examples/s]3538 examples [00:03, 889.49 examples/s]3632 examples [00:04, 900.92 examples/s]3723 examples [00:04, 894.32 examples/s]3815 examples [00:04, 900.75 examples/s]3908 examples [00:04, 907.47 examples/s]4010 examples [00:04, 936.34 examples/s]4104 examples [00:04, 919.52 examples/s]4197 examples [00:04, 882.45 examples/s]4286 examples [00:04, 858.77 examples/s]4374 examples [00:04, 863.33 examples/s]4462 examples [00:05, 866.75 examples/s]4549 examples [00:05, 864.30 examples/s]4636 examples [00:05, 862.69 examples/s]4724 examples [00:05, 865.49 examples/s]4811 examples [00:05, 860.07 examples/s]4902 examples [00:05, 871.89 examples/s]4991 examples [00:05, 873.98 examples/s]5079 examples [00:05, 859.80 examples/s]5176 examples [00:05, 888.36 examples/s]5266 examples [00:05, 889.85 examples/s]5360 examples [00:06, 903.33 examples/s]5451 examples [00:06, 887.85 examples/s]5540 examples [00:06, 880.73 examples/s]5632 examples [00:06, 889.92 examples/s]5722 examples [00:06, 883.11 examples/s]5821 examples [00:06, 911.29 examples/s]5913 examples [00:06, 901.44 examples/s]6004 examples [00:06, 896.08 examples/s]6094 examples [00:06, 885.73 examples/s]6184 examples [00:06, 888.36 examples/s]6284 examples [00:07, 918.74 examples/s]6377 examples [00:07, 891.86 examples/s]6467 examples [00:07, 869.63 examples/s]6555 examples [00:07, 847.75 examples/s]6644 examples [00:07, 857.81 examples/s]6731 examples [00:07, 849.00 examples/s]6824 examples [00:07, 871.20 examples/s]6912 examples [00:07, 854.37 examples/s]7005 examples [00:07, 874.53 examples/s]7098 examples [00:08, 888.94 examples/s]7192 examples [00:08, 901.65 examples/s]7283 examples [00:08, 894.87 examples/s]7373 examples [00:08, 890.88 examples/s]7465 examples [00:08, 896.83 examples/s]7555 examples [00:08, 890.20 examples/s]7645 examples [00:08, 889.79 examples/s]7737 examples [00:08, 897.70 examples/s]7829 examples [00:08, 902.74 examples/s]7920 examples [00:08, 885.99 examples/s]8009 examples [00:09, 859.34 examples/s]8096 examples [00:09, 855.63 examples/s]8182 examples [00:09, 849.16 examples/s]8268 examples [00:09, 829.42 examples/s]8354 examples [00:09, 836.54 examples/s]8449 examples [00:09, 866.29 examples/s]8543 examples [00:09, 884.90 examples/s]8636 examples [00:09, 895.92 examples/s]8730 examples [00:09, 906.89 examples/s]8823 examples [00:09, 912.45 examples/s]8925 examples [00:10, 940.22 examples/s]9020 examples [00:10, 943.03 examples/s]9115 examples [00:10, 937.37 examples/s]9209 examples [00:10, 913.69 examples/s]9309 examples [00:10, 936.09 examples/s]9403 examples [00:10, 925.95 examples/s]9496 examples [00:10, 909.46 examples/s]9588 examples [00:10, 885.69 examples/s]9682 examples [00:10, 900.51 examples/s]9773 examples [00:10, 903.20 examples/s]9867 examples [00:11, 911.66 examples/s]9967 examples [00:11, 935.19 examples/s]10061 examples [00:11, 854.58 examples/s]10156 examples [00:11, 880.68 examples/s]10256 examples [00:11, 911.05 examples/s]10349 examples [00:11, 905.71 examples/s]10448 examples [00:11, 929.13 examples/s]10542 examples [00:11, 922.97 examples/s]10637 examples [00:11, 930.52 examples/s]10731 examples [00:12, 926.00 examples/s]10824 examples [00:12, 923.03 examples/s]10918 examples [00:12, 927.24 examples/s]11011 examples [00:12, 888.98 examples/s]11101 examples [00:12, 861.55 examples/s]11191 examples [00:12, 871.32 examples/s]11284 examples [00:12, 885.47 examples/s]11373 examples [00:12, 882.82 examples/s]11462 examples [00:12, 871.75 examples/s]11551 examples [00:12, 875.81 examples/s]11643 examples [00:13, 887.94 examples/s]11732 examples [00:13, 882.84 examples/s]11823 examples [00:13, 890.29 examples/s]11913 examples [00:13, 891.65 examples/s]12004 examples [00:13, 896.97 examples/s]12094 examples [00:13, 887.76 examples/s]12188 examples [00:13, 902.66 examples/s]12279 examples [00:13, 867.50 examples/s]12367 examples [00:13, 864.00 examples/s]12455 examples [00:14, 868.35 examples/s]12543 examples [00:14, 853.30 examples/s]12630 examples [00:14, 856.79 examples/s]12716 examples [00:14, 850.10 examples/s]12807 examples [00:14, 865.34 examples/s]12899 examples [00:14, 878.99 examples/s]12988 examples [00:14, 862.08 examples/s]13077 examples [00:14, 867.91 examples/s]13164 examples [00:14, 866.13 examples/s]13252 examples [00:14, 868.82 examples/s]13344 examples [00:15, 881.53 examples/s]13437 examples [00:15, 893.56 examples/s]13527 examples [00:15, 887.05 examples/s]13617 examples [00:15, 889.42 examples/s]13708 examples [00:15, 895.36 examples/s]13798 examples [00:15, 860.47 examples/s]13885 examples [00:15, 845.75 examples/s]13970 examples [00:15, 818.60 examples/s]14053 examples [00:15, 812.16 examples/s]14144 examples [00:15, 837.40 examples/s]14234 examples [00:16, 853.94 examples/s]14325 examples [00:16, 867.40 examples/s]14414 examples [00:16, 871.90 examples/s]14502 examples [00:16, 874.10 examples/s]14594 examples [00:16, 886.26 examples/s]14683 examples [00:16, 886.27 examples/s]14774 examples [00:16, 891.63 examples/s]14868 examples [00:16, 904.76 examples/s]14959 examples [00:16, 862.45 examples/s]15046 examples [00:16, 861.57 examples/s]15135 examples [00:17, 867.34 examples/s]15224 examples [00:17, 873.76 examples/s]15315 examples [00:17, 882.04 examples/s]15404 examples [00:17, 860.01 examples/s]15491 examples [00:17, 849.36 examples/s]15577 examples [00:17, 849.49 examples/s]15664 examples [00:17, 855.18 examples/s]15756 examples [00:17, 871.75 examples/s]15846 examples [00:17, 879.00 examples/s]15935 examples [00:18, 872.18 examples/s]16027 examples [00:18, 883.69 examples/s]16116 examples [00:18, 869.75 examples/s]16208 examples [00:18, 883.60 examples/s]16299 examples [00:18, 889.59 examples/s]16390 examples [00:18, 895.21 examples/s]16485 examples [00:18, 908.81 examples/s]16576 examples [00:18, 904.60 examples/s]16667 examples [00:18, 904.37 examples/s]16760 examples [00:18, 909.98 examples/s]16852 examples [00:19, 908.47 examples/s]16943 examples [00:19, 896.99 examples/s]17033 examples [00:19, 889.35 examples/s]17124 examples [00:19, 894.24 examples/s]17221 examples [00:19, 912.72 examples/s]17314 examples [00:19, 917.80 examples/s]17406 examples [00:19, 912.85 examples/s]17498 examples [00:19, 910.68 examples/s]17590 examples [00:19, 900.88 examples/s]17681 examples [00:19, 884.82 examples/s]17772 examples [00:20, 891.86 examples/s]17862 examples [00:20, 879.14 examples/s]17951 examples [00:20, 846.31 examples/s]18046 examples [00:20, 873.72 examples/s]18136 examples [00:20, 878.99 examples/s]18228 examples [00:20, 888.06 examples/s]18323 examples [00:20, 905.28 examples/s]18414 examples [00:20, 875.16 examples/s]18502 examples [00:20, 872.36 examples/s]18590 examples [00:20, 867.85 examples/s]18690 examples [00:21, 901.62 examples/s]18781 examples [00:21, 897.79 examples/s]18872 examples [00:21, 879.24 examples/s]18966 examples [00:21, 894.07 examples/s]19056 examples [00:21, 878.54 examples/s]19145 examples [00:21, 874.62 examples/s]19237 examples [00:21, 885.60 examples/s]19326 examples [00:21, 882.66 examples/s]19418 examples [00:21, 890.73 examples/s]19508 examples [00:22, 878.21 examples/s]19600 examples [00:22, 888.31 examples/s]19692 examples [00:22, 894.80 examples/s]19784 examples [00:22, 900.90 examples/s]19875 examples [00:22, 896.19 examples/s]19965 examples [00:22, 885.73 examples/s]20054 examples [00:22, 841.56 examples/s]20147 examples [00:22, 864.86 examples/s]20239 examples [00:22, 877.97 examples/s]20333 examples [00:22, 894.16 examples/s]20423 examples [00:23, 893.98 examples/s]20521 examples [00:23, 915.36 examples/s]20613 examples [00:23, 915.83 examples/s]20705 examples [00:23, 895.57 examples/s]20800 examples [00:23, 909.72 examples/s]20892 examples [00:23, 866.73 examples/s]20980 examples [00:23, 868.07 examples/s]21068 examples [00:23, 870.65 examples/s]21163 examples [00:23, 891.13 examples/s]21262 examples [00:23, 917.94 examples/s]21356 examples [00:24, 923.42 examples/s]21449 examples [00:24, 888.76 examples/s]21539 examples [00:24, 864.46 examples/s]21626 examples [00:24, 864.28 examples/s]21716 examples [00:24, 872.39 examples/s]21813 examples [00:24, 898.91 examples/s]21904 examples [00:24, 878.42 examples/s]21993 examples [00:24, 881.61 examples/s]22085 examples [00:24, 890.64 examples/s]22179 examples [00:25, 902.36 examples/s]22270 examples [00:25, 895.31 examples/s]22366 examples [00:25, 913.40 examples/s]22459 examples [00:25, 916.56 examples/s]22558 examples [00:25, 934.92 examples/s]22652 examples [00:25, 935.66 examples/s]22746 examples [00:25, 927.81 examples/s]22839 examples [00:25, 915.01 examples/s]22931 examples [00:25, 878.59 examples/s]23020 examples [00:25, 859.33 examples/s]23111 examples [00:26, 871.02 examples/s]23204 examples [00:26, 885.93 examples/s]23293 examples [00:26, 873.60 examples/s]23384 examples [00:26, 881.83 examples/s]23479 examples [00:26, 899.13 examples/s]23571 examples [00:26, 902.65 examples/s]23662 examples [00:26, 900.27 examples/s]23755 examples [00:26, 907.42 examples/s]23848 examples [00:26, 913.54 examples/s]23940 examples [00:26, 899.45 examples/s]24032 examples [00:27, 904.26 examples/s]24123 examples [00:27, 902.97 examples/s]24222 examples [00:27, 927.42 examples/s]24315 examples [00:27, 915.35 examples/s]24407 examples [00:27, 902.27 examples/s]24498 examples [00:27, 881.53 examples/s]24587 examples [00:27, 882.01 examples/s]24678 examples [00:27, 888.66 examples/s]24767 examples [00:27, 879.99 examples/s]24856 examples [00:28, 877.84 examples/s]24949 examples [00:28, 890.95 examples/s]25042 examples [00:28, 902.21 examples/s]25139 examples [00:28, 921.29 examples/s]25232 examples [00:28, 922.47 examples/s]25325 examples [00:28, 912.61 examples/s]25421 examples [00:28, 923.81 examples/s]25514 examples [00:28, 894.49 examples/s]25604 examples [00:28, 872.89 examples/s]25694 examples [00:28, 879.82 examples/s]25785 examples [00:29, 887.82 examples/s]25874 examples [00:29, 864.19 examples/s]25961 examples [00:29, 865.24 examples/s]26048 examples [00:29, 857.31 examples/s]26134 examples [00:29, 853.80 examples/s]26229 examples [00:29, 880.18 examples/s]26318 examples [00:29, 873.90 examples/s]26409 examples [00:29, 883.49 examples/s]26502 examples [00:29, 895.36 examples/s]26593 examples [00:29, 898.19 examples/s]26683 examples [00:30, 898.49 examples/s]26773 examples [00:30, 887.32 examples/s]26862 examples [00:30, 876.20 examples/s]26957 examples [00:30, 895.88 examples/s]27049 examples [00:30, 902.75 examples/s]27141 examples [00:30, 905.58 examples/s]27232 examples [00:30, 903.72 examples/s]27323 examples [00:30, 884.73 examples/s]27412 examples [00:30, 865.34 examples/s]27499 examples [00:30, 856.40 examples/s]27586 examples [00:31, 858.87 examples/s]27672 examples [00:31, 855.78 examples/s]27758 examples [00:31, 847.86 examples/s]27851 examples [00:31, 867.66 examples/s]27942 examples [00:31, 878.85 examples/s]28037 examples [00:31, 896.67 examples/s]28128 examples [00:31, 900.18 examples/s]28219 examples [00:31, 902.58 examples/s]28310 examples [00:31, 898.40 examples/s]28400 examples [00:31, 896.19 examples/s]28491 examples [00:32, 899.80 examples/s]28587 examples [00:32, 915.99 examples/s]28679 examples [00:32, 909.60 examples/s]28771 examples [00:32, 892.10 examples/s]28865 examples [00:32, 903.01 examples/s]28958 examples [00:32, 910.07 examples/s]29054 examples [00:32, 923.82 examples/s]29147 examples [00:32, 923.54 examples/s]29240 examples [00:32, 920.72 examples/s]29333 examples [00:33, 909.96 examples/s]29425 examples [00:33, 903.90 examples/s]29516 examples [00:33, 897.95 examples/s]29606 examples [00:33, 883.59 examples/s]29702 examples [00:33, 903.04 examples/s]29800 examples [00:33, 923.52 examples/s]29893 examples [00:33, 919.82 examples/s]29986 examples [00:33, 898.57 examples/s]30077 examples [00:33, 849.57 examples/s]30166 examples [00:33, 860.39 examples/s]30264 examples [00:34, 892.22 examples/s]30354 examples [00:34, 870.65 examples/s]30442 examples [00:34, 824.86 examples/s]30532 examples [00:34, 845.88 examples/s]30625 examples [00:34, 867.15 examples/s]30726 examples [00:34, 903.59 examples/s]30820 examples [00:34, 913.32 examples/s]30920 examples [00:34, 935.49 examples/s]31015 examples [00:34, 933.22 examples/s]31109 examples [00:34, 928.45 examples/s]31203 examples [00:35, 930.99 examples/s]31297 examples [00:35, 907.27 examples/s]31389 examples [00:35, 898.91 examples/s]31480 examples [00:35, 863.13 examples/s]31575 examples [00:35, 885.37 examples/s]31665 examples [00:35, 887.72 examples/s]31757 examples [00:35, 896.04 examples/s]31847 examples [00:35, 875.36 examples/s]31935 examples [00:35, 875.69 examples/s]32023 examples [00:36, 869.45 examples/s]32113 examples [00:36, 876.56 examples/s]32201 examples [00:36, 868.31 examples/s]32291 examples [00:36, 876.78 examples/s]32381 examples [00:36, 881.56 examples/s]32470 examples [00:36, 883.09 examples/s]32562 examples [00:36, 892.11 examples/s]32652 examples [00:36, 879.53 examples/s]32751 examples [00:36, 908.50 examples/s]32846 examples [00:36, 919.62 examples/s]32942 examples [00:37, 930.39 examples/s]33036 examples [00:37, 931.25 examples/s]33131 examples [00:37, 934.11 examples/s]33225 examples [00:37, 903.94 examples/s]33316 examples [00:37, 903.41 examples/s]33408 examples [00:37, 908.15 examples/s]33499 examples [00:37, 862.88 examples/s]33590 examples [00:37, 875.64 examples/s]33689 examples [00:37, 904.45 examples/s]33780 examples [00:37, 896.64 examples/s]33871 examples [00:38, 887.48 examples/s]33967 examples [00:38, 905.11 examples/s]34058 examples [00:38, 901.97 examples/s]34150 examples [00:38, 905.65 examples/s]34241 examples [00:38, 880.73 examples/s]34334 examples [00:38, 893.81 examples/s]34424 examples [00:38, 883.43 examples/s]34513 examples [00:38, 840.32 examples/s]34600 examples [00:38, 847.25 examples/s]34692 examples [00:39, 866.29 examples/s]34783 examples [00:39, 876.57 examples/s]34872 examples [00:39, 879.59 examples/s]34961 examples [00:39, 867.47 examples/s]35052 examples [00:39, 879.18 examples/s]35141 examples [00:39, 881.17 examples/s]35244 examples [00:39, 919.20 examples/s]35337 examples [00:39, 896.99 examples/s]35428 examples [00:39, 877.74 examples/s]35521 examples [00:39, 890.85 examples/s]35618 examples [00:40, 911.23 examples/s]35710 examples [00:40, 911.14 examples/s]35804 examples [00:40, 919.53 examples/s]35897 examples [00:40, 902.60 examples/s]35988 examples [00:40, 899.23 examples/s]36082 examples [00:40, 911.08 examples/s]36180 examples [00:40, 928.51 examples/s]36277 examples [00:40, 937.82 examples/s]36371 examples [00:40, 920.70 examples/s]36464 examples [00:40, 904.63 examples/s]36555 examples [00:41, 903.63 examples/s]36651 examples [00:41, 918.62 examples/s]36744 examples [00:41, 917.27 examples/s]36836 examples [00:41, 913.39 examples/s]36928 examples [00:41, 905.35 examples/s]37019 examples [00:41, 879.28 examples/s]37122 examples [00:41, 918.17 examples/s]37220 examples [00:41, 934.10 examples/s]37314 examples [00:41, 900.10 examples/s]37407 examples [00:42, 906.83 examples/s]37499 examples [00:42, 891.83 examples/s]37590 examples [00:42, 896.33 examples/s]37686 examples [00:42, 913.08 examples/s]37782 examples [00:42, 926.25 examples/s]37875 examples [00:42, 924.41 examples/s]37968 examples [00:42, 918.73 examples/s]38060 examples [00:42, 902.07 examples/s]38153 examples [00:42, 908.47 examples/s]38244 examples [00:42, 906.40 examples/s]38345 examples [00:43, 933.57 examples/s]38450 examples [00:43, 965.61 examples/s]38548 examples [00:43, 952.08 examples/s]38644 examples [00:43, 920.55 examples/s]38737 examples [00:43, 910.82 examples/s]38834 examples [00:43, 924.83 examples/s]38927 examples [00:43, 910.20 examples/s]39022 examples [00:43, 918.94 examples/s]39115 examples [00:43, 908.08 examples/s]39206 examples [00:43, 893.24 examples/s]39301 examples [00:44, 908.49 examples/s]39393 examples [00:44, 909.76 examples/s]39485 examples [00:44, 897.03 examples/s]39576 examples [00:44, 898.50 examples/s]39666 examples [00:44, 893.89 examples/s]39756 examples [00:44, 891.04 examples/s]39849 examples [00:44, 901.89 examples/s]39940 examples [00:44, 883.42 examples/s]40029 examples [00:44, 814.57 examples/s]40121 examples [00:45, 842.82 examples/s]40212 examples [00:45, 861.16 examples/s]40306 examples [00:45, 882.02 examples/s]40399 examples [00:45, 895.51 examples/s]40491 examples [00:45, 901.37 examples/s]40583 examples [00:45, 905.99 examples/s]40675 examples [00:45, 908.81 examples/s]40767 examples [00:45, 904.63 examples/s]40859 examples [00:45, 908.05 examples/s]40950 examples [00:45, 887.32 examples/s]41039 examples [00:46, 880.57 examples/s]41129 examples [00:46, 882.47 examples/s]41218 examples [00:46, 876.48 examples/s]41308 examples [00:46, 882.94 examples/s]41398 examples [00:46, 886.10 examples/s]41487 examples [00:46, 881.64 examples/s]41578 examples [00:46, 888.32 examples/s]41667 examples [00:46, 872.29 examples/s]41757 examples [00:46, 877.86 examples/s]41850 examples [00:46, 892.45 examples/s]41940 examples [00:47, 883.46 examples/s]42030 examples [00:47, 886.40 examples/s]42121 examples [00:47, 891.26 examples/s]42211 examples [00:47, 892.08 examples/s]42301 examples [00:47, 889.60 examples/s]42390 examples [00:47, 874.84 examples/s]42478 examples [00:47, 824.04 examples/s]42562 examples [00:47, 827.57 examples/s]42650 examples [00:47, 840.65 examples/s]42743 examples [00:48, 865.51 examples/s]42845 examples [00:48, 904.66 examples/s]42937 examples [00:48, 898.69 examples/s]43028 examples [00:48, 897.77 examples/s]43119 examples [00:48, 900.61 examples/s]43210 examples [00:48, 902.58 examples/s]43301 examples [00:48, 898.79 examples/s]43392 examples [00:48, 889.73 examples/s]43482 examples [00:48, 877.95 examples/s]43570 examples [00:48, 876.38 examples/s]43660 examples [00:49, 881.73 examples/s]43749 examples [00:49, 868.43 examples/s]43840 examples [00:49, 877.63 examples/s]43931 examples [00:49, 885.84 examples/s]44020 examples [00:49, 866.11 examples/s]44107 examples [00:49, 865.99 examples/s]44194 examples [00:49, 866.39 examples/s]44292 examples [00:49, 894.94 examples/s]44386 examples [00:49, 907.42 examples/s]44479 examples [00:49, 912.63 examples/s]44571 examples [00:50, 897.44 examples/s]44666 examples [00:50, 910.64 examples/s]44760 examples [00:50, 916.39 examples/s]44852 examples [00:50, 901.62 examples/s]44943 examples [00:50, 897.58 examples/s]45033 examples [00:50, 873.71 examples/s]45121 examples [00:50, 840.83 examples/s]45213 examples [00:50, 861.23 examples/s]45303 examples [00:50, 871.75 examples/s]45392 examples [00:50, 876.99 examples/s]45480 examples [00:51, 844.41 examples/s]45569 examples [00:51, 856.34 examples/s]45658 examples [00:51, 866.04 examples/s]45745 examples [00:51, 866.53 examples/s]45832 examples [00:51, 866.22 examples/s]45919 examples [00:51, 858.42 examples/s]46013 examples [00:51, 879.52 examples/s]46102 examples [00:51, 860.42 examples/s]46197 examples [00:51, 883.60 examples/s]46288 examples [00:52, 888.95 examples/s]46381 examples [00:52, 899.88 examples/s]46472 examples [00:52, 892.05 examples/s]46569 examples [00:52, 913.21 examples/s]46664 examples [00:52, 922.50 examples/s]46757 examples [00:52, 918.11 examples/s]46849 examples [00:52, 902.07 examples/s]46940 examples [00:52, 892.50 examples/s]47030 examples [00:52, 879.27 examples/s]47122 examples [00:52, 890.43 examples/s]47215 examples [00:53, 901.66 examples/s]47311 examples [00:53, 917.50 examples/s]47403 examples [00:53, 906.56 examples/s]47494 examples [00:53, 890.15 examples/s]47584 examples [00:53, 882.65 examples/s]47675 examples [00:53, 888.69 examples/s]47764 examples [00:53, 875.91 examples/s]47852 examples [00:53, 869.21 examples/s]47943 examples [00:53, 879.53 examples/s]48037 examples [00:53, 893.94 examples/s]48133 examples [00:54, 909.73 examples/s]48226 examples [00:54, 914.83 examples/s]48319 examples [00:54, 916.41 examples/s]48411 examples [00:54, 889.02 examples/s]48502 examples [00:54, 895.08 examples/s]48593 examples [00:54, 899.24 examples/s]48684 examples [00:54, 893.82 examples/s]48774 examples [00:54, 884.48 examples/s]48863 examples [00:54, 878.18 examples/s]48958 examples [00:54, 896.64 examples/s]49050 examples [00:55, 901.72 examples/s]49143 examples [00:55, 908.82 examples/s]49234 examples [00:55, 897.77 examples/s]49329 examples [00:55, 909.96 examples/s]49421 examples [00:55, 904.57 examples/s]49513 examples [00:55, 907.56 examples/s]49606 examples [00:55, 913.63 examples/s]49698 examples [00:55, 904.95 examples/s]49789 examples [00:55, 875.72 examples/s]49877 examples [00:56, 872.94 examples/s]49965 examples [00:56, 835.53 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6695/50000 [00:00<00:00, 66949.90 examples/s] 36%|███▋      | 18232/50000 [00:00<00:00, 76593.25 examples/s] 58%|█████▊    | 29169/50000 [00:00<00:00, 84159.51 examples/s] 81%|████████  | 40612/50000 [00:00<00:00, 91390.22 examples/s]                                                               0 examples [00:00, ? examples/s]63 examples [00:00, 628.60 examples/s]151 examples [00:00, 686.88 examples/s]242 examples [00:00, 739.95 examples/s]339 examples [00:00, 794.86 examples/s]434 examples [00:00, 835.55 examples/s]526 examples [00:00, 858.98 examples/s]618 examples [00:00, 876.33 examples/s]713 examples [00:00, 894.60 examples/s]800 examples [00:00, 886.67 examples/s]894 examples [00:01, 901.55 examples/s]986 examples [00:01, 905.07 examples/s]1076 examples [00:01, 892.78 examples/s]1170 examples [00:01, 904.44 examples/s]1262 examples [00:01, 907.34 examples/s]1354 examples [00:01, 909.86 examples/s]1445 examples [00:01, 713.18 examples/s]1537 examples [00:01, 764.61 examples/s]1628 examples [00:01, 801.04 examples/s]1715 examples [00:02, 818.24 examples/s]1806 examples [00:02, 842.04 examples/s]1898 examples [00:02, 861.27 examples/s]1989 examples [00:02, 875.04 examples/s]2082 examples [00:02, 888.91 examples/s]2172 examples [00:02, 888.02 examples/s]2263 examples [00:02, 892.27 examples/s]2353 examples [00:02, 891.92 examples/s]2443 examples [00:02, 886.08 examples/s]2532 examples [00:02, 879.76 examples/s]2625 examples [00:03, 894.23 examples/s]2718 examples [00:03, 900.84 examples/s]2809 examples [00:03, 875.88 examples/s]2901 examples [00:03, 887.39 examples/s]2993 examples [00:03, 894.06 examples/s]3083 examples [00:03, 873.42 examples/s]3171 examples [00:03, 871.07 examples/s]3262 examples [00:03, 882.37 examples/s]3356 examples [00:03, 897.83 examples/s]3446 examples [00:03, 893.29 examples/s]3536 examples [00:04, 894.07 examples/s]3629 examples [00:04, 902.25 examples/s]3720 examples [00:04, 889.19 examples/s]3814 examples [00:04, 901.93 examples/s]3908 examples [00:04, 910.79 examples/s]4000 examples [00:04, 903.81 examples/s]4095 examples [00:04, 916.40 examples/s]4194 examples [00:04, 937.01 examples/s]4288 examples [00:04, 936.24 examples/s]4382 examples [00:04, 908.38 examples/s]4474 examples [00:05, 908.45 examples/s]4567 examples [00:05, 911.98 examples/s]4662 examples [00:05, 919.83 examples/s]4755 examples [00:05, 910.19 examples/s]4847 examples [00:05, 912.78 examples/s]4939 examples [00:05, 898.12 examples/s]5034 examples [00:05, 911.05 examples/s]5130 examples [00:05, 922.26 examples/s]5226 examples [00:05, 931.72 examples/s]5320 examples [00:05, 933.82 examples/s]5414 examples [00:06, 905.51 examples/s]5505 examples [00:06, 906.10 examples/s]5596 examples [00:06, 896.69 examples/s]5690 examples [00:06, 907.78 examples/s]5781 examples [00:06, 900.81 examples/s]5880 examples [00:06, 925.65 examples/s]5976 examples [00:06, 934.82 examples/s]6074 examples [00:06, 946.58 examples/s]6169 examples [00:06, 939.50 examples/s]6265 examples [00:07, 942.95 examples/s]6360 examples [00:07, 925.21 examples/s]6453 examples [00:07, 925.54 examples/s]6546 examples [00:07, 909.59 examples/s]6639 examples [00:07, 912.92 examples/s]6731 examples [00:07, 907.29 examples/s]6822 examples [00:07, 907.28 examples/s]6918 examples [00:07, 920.79 examples/s]7011 examples [00:07, 892.98 examples/s]7104 examples [00:07, 902.28 examples/s]7199 examples [00:08, 916.08 examples/s]7291 examples [00:08, 895.02 examples/s]7389 examples [00:08, 916.01 examples/s]7483 examples [00:08, 921.49 examples/s]7576 examples [00:08, 914.48 examples/s]7668 examples [00:08, 908.18 examples/s]7759 examples [00:08, 863.64 examples/s]7846 examples [00:08, 838.78 examples/s]7939 examples [00:08, 863.10 examples/s]8038 examples [00:08, 897.34 examples/s]8132 examples [00:09, 907.93 examples/s]8226 examples [00:09, 916.64 examples/s]8319 examples [00:09, 856.31 examples/s]8408 examples [00:09, 865.21 examples/s]8496 examples [00:09, 856.60 examples/s]8583 examples [00:09, 847.83 examples/s]8679 examples [00:09, 877.68 examples/s]8771 examples [00:09, 888.53 examples/s]8866 examples [00:09, 905.54 examples/s]8957 examples [00:10, 903.01 examples/s]9050 examples [00:10, 910.57 examples/s]9149 examples [00:10, 932.47 examples/s]9245 examples [00:10, 939.91 examples/s]9340 examples [00:10, 931.18 examples/s]9434 examples [00:10, 933.25 examples/s]9528 examples [00:10, 912.08 examples/s]9620 examples [00:10, 891.00 examples/s]9712 examples [00:10, 898.87 examples/s]9806 examples [00:10, 908.13 examples/s]9900 examples [00:11, 916.69 examples/s]9995 examples [00:11, 924.68 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9IM53T/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9IM53T/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0504e078c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0504e078c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0504e078c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0571281e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04e61e8828>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0504e078c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0504e078c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0504e078c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04e65d7390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04e8b162e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f04dc7a87b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f04dc7a87b8> 

  function with postional parmater data_info <function split_train_valid at 0x7f04dc7a87b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f04dc7a88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f04dc7a88c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f04dc7a88c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=db01ebf703cfa7daf531eb8853f5a8f1cb0da599316fb0ca168ea27eb163540b
  Stored in directory: /tmp/pip-ephem-wheel-cache-vbsjc_wh/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:08<262:12:06, 913B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:09<183:45:51, 1.30kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:09<128:39:53, 1.86kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:09<90:00:30, 2.66kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:09<62:48:42, 3.80kB/s].vector_cache/glove.6B.zip:   1%|          | 7.35M/862M [00:09<43:46:40, 5.42kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:09<30:28:03, 7.75kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:09<21:10:51, 11.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.2M/862M [00:09<14:44:24, 15.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:10<10:16:18, 22.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:10<7:08:31, 32.3kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:10<4:59:03, 46.1kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:10<3:28:00, 65.8kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:10<2:24:49, 93.9kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:10<1:41:02, 134kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:10<1:10:55, 191kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:11<49:49, 270kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:12<12:44:29, 17.6kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:12<8:55:41, 25.1kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:12<6:14:06, 35.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:14<4:27:15, 50.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:14<3:10:09, 70.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:14<2:13:46, 99.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.1M/862M [00:14<1:33:31, 142kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:16<1:16:28, 174kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:16<54:53, 242kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:16<38:38, 344kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:18<30:05, 440kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:18<22:23, 591kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:18<15:54, 830kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:20<14:16, 923kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:20<12:34, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:20<09:22, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:20<06:41, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:22<20:47, 630kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:22<15:54, 823kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:22<11:24, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:24<11:01, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:24<09:09, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:24<06:44, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:26<07:33, 1.71MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:26<08:09, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:26<06:25, 2.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:26<04:38, 2.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:28<17:48, 724kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:28<13:53, 928kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:28<10:03, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:30<09:50, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:30<09:43, 1.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:30<07:23, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:30<05:19, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:32<11:06, 1.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:32<09:10, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:32<06:45, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<07:30, 1.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:34<08:03, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:34<06:17, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:34<04:34, 2.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:36<16:34, 761kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:36<12:59, 971kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:36<09:21, 1.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:38<09:15, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<07:52, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:38<05:50, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<06:48, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<06:08, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:40<04:35, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<05:55, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<06:54, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:42<05:31, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:42<04:00, 3.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<16:02, 769kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<12:36, 979kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:44<09:06, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:46<09:02, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:46<09:02, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:46<06:56, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:46<05:00, 2.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:48<17:01, 716kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:48<13:17, 918kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:48<09:33, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:48<06:50, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:50<34:00, 357kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<26:33, 457kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<19:15, 629kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:50<13:35, 888kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:52<24:06, 500kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:52<18:13, 661kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:52<12:59, 926kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:52<09:13, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<46:39, 257kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<33:58, 353kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:54<24:03, 497kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<19:24, 614kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<14:54, 799kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:56<10:44, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:58<10:06, 1.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:58<09:42, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:58<07:26, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:58<05:21, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:00<18:00, 654kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:00<13:40, 861kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:00<09:48, 1.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:00<07:01, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:02<16:42, 701kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:02<12:56, 904kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:02<09:19, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:04<09:03, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:04<07:36, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:04<05:38, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:06<06:28, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:06<07:05, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:06<05:36, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:06<04:03, 2.83MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<16:47, 685kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<13:01, 883kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:08<09:24, 1.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:10<09:04, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:10<08:52, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:10<06:43, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:10<04:52, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:12<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:12<06:27, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:12<04:46, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:14<05:49, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:14<06:33, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:14<05:13, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:14<03:47, 2.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:16<16:15, 690kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:16<12:37, 889kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:16<09:07, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<08:48, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<07:22, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:18<05:27, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<06:15, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<06:51, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:20<05:17, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:20<03:51, 2.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<06:43, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<05:53, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:22<04:25, 2.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<05:29, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<06:16, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:24<04:59, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:24<03:37, 3.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:26<15:44, 691kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:26<12:11, 892kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:26<08:48, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<08:33, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<08:23, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:28<06:27, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:28<04:39, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<16:17, 659kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<12:22, 867kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:30<08:53, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:30<06:22, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<19:29, 547kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<16:00, 667kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:32<11:46, 905kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:32<08:20, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:33<17:10, 617kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:34<13:10, 805kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:34<09:27, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<08:54, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:36<07:22, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:36<05:26, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:37<06:05, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:38<06:34, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:38<05:10, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:38<03:44, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:39<15:14, 682kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:40<11:36, 894kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:40<08:22, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:40<06:01, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:41<15:48, 653kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:42<13:20, 773kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:42<09:54, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:42<07:01, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:43<15:54, 645kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<12:12, 839kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:44<08:45, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:45<08:23, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:46<08:07, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:46<06:08, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:46<04:27, 2.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:47<06:27, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:48<05:40, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<04:11, 2.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:49<05:06, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:50<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:50<03:25, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:50<03:00, 3.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:51<6:17:57, 26.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:51<4:24:16, 37.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:53<3:06:02, 53.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:53<2:12:23, 74.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:53<1:33:07, 106kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:53<1:04:59, 152kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:55<56:32, 174kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:55<40:37, 242kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:55<28:38, 343kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<22:06, 442kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<17:39, 553kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<12:48, 763kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:57<09:09, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:59<08:53, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:59<07:17, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:59<05:21, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<05:51, 1.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<06:21, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:01<04:58, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:01<03:35, 2.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<14:07, 678kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<10:56, 874kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:03<07:53, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:05<07:35, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:05<07:31, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:05<05:47, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:05<04:09, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:07<11:40, 808kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:07<09:10, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:07<06:39, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<06:44, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<06:47, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<05:11, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:09<03:44, 2.49MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<06:53, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<05:47, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:11<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<05:35, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:13<04:23, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:13<03:11, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<12:32, 730kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<09:44, 939kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:15<07:02, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<06:57, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<06:58, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:17<05:17, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:17<03:49, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:19<05:59, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:19<05:11, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:19<03:51, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<04:39, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<05:14, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<04:05, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:21<02:58, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:23<05:47, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:23<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:23<03:43, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:25<04:34, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:25<05:08, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:25<02:57, 2.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<12:40, 689kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<09:47, 892kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<07:03, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<06:52, 1.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<06:39, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:29<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:29<03:39, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:31<05:41, 1.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:31<04:55, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<04:26, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<04:59, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<02:50, 2.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:35<05:17, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:35<04:38, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:35<03:26, 2.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:37<04:14, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:37<04:50, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:37<03:50, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<02:47, 2.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:39<12:02, 691kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:39<09:09, 907kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<06:37, 1.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:41<06:27, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:41<05:26, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:41<03:59, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:43<04:34, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:43<05:01, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:43<03:58, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:43<02:52, 2.82MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:45<11:50, 685kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:45<09:08, 886kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<06:34, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:47<06:23, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:47<06:15, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:47<04:48, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:47<03:27, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:49<12:05, 659kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:49<09:20, 853kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<06:42, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:51<06:24, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:51<06:10, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<04:44, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<03:23, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:53<18:30, 423kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:53<13:48, 567kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:53<09:49, 794kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:55<08:32, 909kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<09:46, 794kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:55<07:48, 993kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:55<05:41, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:57<05:27, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:57<05:42, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:57<04:23, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<03:11, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<05:02, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:59<03:58, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<02:51, 2.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<07:37, 990kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<07:06, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:01<05:20, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<03:49, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:03<06:09, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:03<06:03, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<04:36, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<03:20, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:05<04:37, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:05<04:49, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:05<03:43, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:05<02:43, 2.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<04:07, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<04:41, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<03:42, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<02:40, 2.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<07:15, 1.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<06:47, 1.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:09<05:06, 1.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<03:40, 1.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<05:17, 1.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:11<05:22, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:11<04:06, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<03:00, 2.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:13<04:08, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:13<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:13<03:21, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:13<02:28, 2.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:15<04:11, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:15<03:13, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<02:21, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:17<04:15, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:17<04:24, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<03:25, 2.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:17<02:28, 2.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:19<06:22, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<06:04, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:19<04:39, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:19<03:20, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<07:35, 904kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:21<06:36, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<04:56, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:21<03:30, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:22<08:58, 756kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<07:33, 898kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<05:32, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<03:57, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:24<06:08, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:25<05:51, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<04:27, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<03:11, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:26<07:10, 926kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<06:34, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<04:59, 1.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:27<03:33, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:28<07:32, 874kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<06:49, 964kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<05:07, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:29<03:40, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:30<04:37, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<04:44, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<03:40, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<02:52, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:32<3:49:34, 28.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:32<2:40:47, 40.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:32<1:51:52, 57.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:34<1:23:13, 76.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:34<58:18, 109kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<40:51, 155kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:36<29:51, 212kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:36<22:21, 282kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<15:58, 395kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:36<11:11, 559kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:38<12:30, 499kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:38<10:11, 613kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:38<07:29, 833kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:38<05:17, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:40<08:37, 716kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:40<07:23, 835kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:40<05:30, 1.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:40<03:54, 1.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:42<07:53, 774kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:42<06:37, 922kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<04:54, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:42<03:29, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:44<25:33, 236kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:44<19:00, 318kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<13:33, 444kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:44<09:28, 630kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:46<24:55, 240kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:46<18:49, 317kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<13:27, 443kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<09:26, 627kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:48<09:31, 620kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:48<08:01, 735kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<05:57, 989kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:48<04:13, 1.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<07:24, 787kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<06:14, 934kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:50<04:35, 1.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:50<03:15, 1.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<10:24, 554kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<08:37, 668kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:52<06:19, 910kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<04:29, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:54<05:15, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<05:00, 1.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:54<02:43, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<06:39, 845kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<05:58, 942kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<04:26, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<03:10, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<04:08, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<04:11, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:58<03:12, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:58<02:18, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<03:43, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<03:38, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:00<02:45, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<01:59, 2.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<04:11, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<04:10, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<02:19, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:04<03:08, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:04<03:20, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<02:34, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<01:51, 2.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:06<03:59, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<04:04, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<03:09, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:06<02:15, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:08<05:26, 957kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<05:00, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<03:45, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:08<02:42, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<03:28, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<03:41, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<02:49, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:10<02:02, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<03:22, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<03:16, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<02:30, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:12<01:48, 2.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<2:24:32, 34.6kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<1:42:00, 49.0kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<1:11:42, 69.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:14<49:53, 99.3kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<36:46, 134kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<26:49, 184kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<19:00, 259kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<13:14, 368kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:18<13:20, 364kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:18<10:12, 476kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:18<07:20, 660kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:18<05:08, 932kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:20<59:14, 80.9kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<42:20, 113kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<29:45, 160kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:20<20:40, 229kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<48:35, 97.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<35:05, 135kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<24:46, 190kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:22<17:14, 271kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<15:48, 295kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<11:51, 392kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<08:26, 549kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:24<05:55, 776kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<07:10, 640kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<06:04, 755kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<04:27, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:26<03:09, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<04:12, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<03:41, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<02:44, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:28<01:57, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:29<04:42, 943kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:30<05:28, 812kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<04:20, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<03:09, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:31<03:04, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:32<02:53, 1.52MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:32<02:10, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:32<01:33, 2.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:33<11:44, 367kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:34<09:10, 470kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:34<06:36, 650kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:34<04:38, 917kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:34<04:11, 1.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:35<2:27:43, 28.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:35<1:43:19, 41.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:35<1:11:51, 58.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:37<51:27, 81.3kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:37<37:51, 110kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:37<26:54, 155kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:37<18:49, 221kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<13:57, 295kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<10:38, 387kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:39<07:38, 536kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:39<05:20, 758kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:41<09:05, 445kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<07:13, 560kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<05:13, 773kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:41<03:41, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<03:55, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<03:37, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<02:44, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:43<01:56, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<06:15, 624kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<05:12, 750kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:45<03:48, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:45<02:41, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:47<03:31, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:47<03:16, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:47<02:27, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:47<01:45, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:49<02:47, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:49<02:30, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:49<01:52, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:51<01:58, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:51<02:53, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:51<02:22, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:51<01:43, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:53<02:03, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<02:08, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:53<01:40, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:55<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<01:54, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:55<01:30, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:36, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:48, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:57<01:26, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<01:33, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<01:45, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:59<01:21, 2.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:59<00:58, 3.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<04:06, 817kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<03:32, 949kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:01<02:37, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<02:21, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<02:16, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:03<01:43, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<02:25, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:05<02:00, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:05<01:27, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<01:46, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<01:51, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<01:26, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:09<01:30, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:09<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<01:17, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:11<01:23, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:33, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:13, 2.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:13<01:19, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:30, 1.95MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:13<01:11, 2.45MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:15<01:17, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:15<01:27, 1.96MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:15<01:09, 2.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:16<01:15, 2.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:17<01:24, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:17<01:06, 2.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:18<01:13, 2.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:19<01:23, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:19<01:05, 2.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<01:11, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:21<01:21, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:21<01:03, 2.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:21<00:45, 3.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:21<01:59, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<1:37:55, 26.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:22<1:08:18, 38.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:22<46:50, 54.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<1:04:42, 39.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<45:54, 55.3kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<32:10, 78.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:24<22:11, 112kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<16:42, 148kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<12:06, 204kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:26<08:31, 288kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<06:15, 384kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<04:47, 500kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<03:26, 693kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<02:45, 846kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:30<02:20, 999kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<01:42, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:30<01:12, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<02:47, 812kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<02:19, 970kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:43, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<01:33, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:28, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:06, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:34<00:46, 2.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:36<02:51, 746kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<02:21, 899kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:36<01:44, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:32, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:25, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:38<01:05, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:04, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:06, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:40<00:50, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:40<00:35, 3.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:50, 1.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:57, 982kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:31, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:42<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:15, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:12, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<00:54, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<00:55, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<01:16, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<01:02, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<00:59, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:59, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:44, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:48<00:32, 3.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:50<01:17, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<01:11, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:50<00:53, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<00:52, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<01:09, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<00:55, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:52<00:40, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:48, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:54<00:37, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:56<00:40, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:43, 2.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:56<00:32, 2.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:56<00:23, 3.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<01:24, 976kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<01:12, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:53, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:48, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:36, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:37, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:02<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:02<00:29, 2.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<00:39, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:04<00:29, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:04<00:20, 3.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<01:35, 685kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<01:18, 836kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:06<00:56, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<00:48, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<00:44, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:08<00:32, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:08<00:22, 2.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:09<01:47, 532kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:10<01:25, 671kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:10<01:01, 921kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:11<00:49, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:43, 1.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:12<00:32, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:13<00:29, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:29, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:14<00:16, 2.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:15<29:37, 25.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:15<20:20, 36.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<13:20, 51.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<09:33, 72.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<06:40, 102kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:18<04:28, 145kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:19<03:09, 196kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:19<02:18, 268kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:19<01:35, 376kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<01:07, 492kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<00:58, 564kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:21<00:43, 753kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:21<00:29, 1.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:23<00:27, 1.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:23<00:23, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:23<00:17, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:25<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:14, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:10, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:27<00:10, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<00:10, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:27<00:08, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<00:08, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:29<00:05, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:05, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:05, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:31<00:04, 2.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:03, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:03, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:02, 2.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:33<00:01, 3.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:02, 2.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:02, 1.98MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:01, 2.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:37<00:00, 2.22MB/s].vector_cache/glove.6B.zip: 862MB [06:37, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 301/400000 [00:00<02:12, 3009.65it/s]  0%|          | 1101/400000 [00:00<01:47, 3702.16it/s]  0%|          | 1860/400000 [00:00<01:31, 4373.63it/s]  1%|          | 2662/400000 [00:00<01:18, 5063.16it/s]  1%|          | 3394/400000 [00:00<01:11, 5579.08it/s]  1%|          | 4116/400000 [00:00<01:06, 5987.26it/s]  1%|          | 4899/400000 [00:00<01:01, 6441.16it/s]  1%|▏         | 5688/400000 [00:00<00:57, 6816.17it/s]  2%|▏         | 6495/400000 [00:00<00:55, 7148.99it/s]  2%|▏         | 7237/400000 [00:01<00:55, 7107.09it/s]  2%|▏         | 7994/400000 [00:01<00:54, 7238.14it/s]  2%|▏         | 8753/400000 [00:01<00:53, 7339.72it/s]  2%|▏         | 9520/400000 [00:01<00:52, 7432.26it/s]  3%|▎         | 10271/400000 [00:01<00:52, 7396.50it/s]  3%|▎         | 11016/400000 [00:01<00:53, 7334.46it/s]  3%|▎         | 11754/400000 [00:01<00:53, 7252.69it/s]  3%|▎         | 12519/400000 [00:01<00:52, 7365.34it/s]  3%|▎         | 13276/400000 [00:01<00:52, 7424.93it/s]  4%|▎         | 14086/400000 [00:01<00:50, 7614.06it/s]  4%|▎         | 14850/400000 [00:02<00:51, 7538.80it/s]  4%|▍         | 15659/400000 [00:02<00:49, 7695.39it/s]  4%|▍         | 16431/400000 [00:02<00:50, 7580.44it/s]  4%|▍         | 17242/400000 [00:02<00:49, 7730.52it/s]  5%|▍         | 18018/400000 [00:02<00:50, 7554.64it/s]  5%|▍         | 18784/400000 [00:02<00:50, 7584.68it/s]  5%|▍         | 19545/400000 [00:02<00:50, 7561.80it/s]  5%|▌         | 20303/400000 [00:02<00:50, 7482.55it/s]  5%|▌         | 21070/400000 [00:02<00:50, 7536.65it/s]  5%|▌         | 21840/400000 [00:02<00:49, 7582.79it/s]  6%|▌         | 22599/400000 [00:03<00:50, 7515.54it/s]  6%|▌         | 23352/400000 [00:03<00:51, 7348.27it/s]  6%|▌         | 24141/400000 [00:03<00:50, 7502.29it/s]  6%|▌         | 24930/400000 [00:03<00:49, 7612.38it/s]  6%|▋         | 25710/400000 [00:03<00:48, 7665.91it/s]  7%|▋         | 26515/400000 [00:03<00:48, 7776.43it/s]  7%|▋         | 27294/400000 [00:03<00:49, 7563.56it/s]  7%|▋         | 28053/400000 [00:03<00:49, 7559.48it/s]  7%|▋         | 28864/400000 [00:03<00:48, 7715.53it/s]  7%|▋         | 29650/400000 [00:03<00:47, 7757.40it/s]  8%|▊         | 30428/400000 [00:04<00:48, 7624.57it/s]  8%|▊         | 31229/400000 [00:04<00:47, 7735.97it/s]  8%|▊         | 32005/400000 [00:04<00:47, 7733.94it/s]  8%|▊         | 32787/400000 [00:04<00:47, 7759.32it/s]  8%|▊         | 33633/400000 [00:04<00:46, 7955.40it/s]  9%|▊         | 34431/400000 [00:04<00:46, 7946.63it/s]  9%|▉         | 35242/400000 [00:04<00:45, 7992.36it/s]  9%|▉         | 36043/400000 [00:04<00:46, 7779.81it/s]  9%|▉         | 36831/400000 [00:04<00:46, 7806.62it/s]  9%|▉         | 37644/400000 [00:04<00:45, 7899.58it/s] 10%|▉         | 38436/400000 [00:05<00:45, 7888.34it/s] 10%|▉         | 39226/400000 [00:05<00:46, 7777.46it/s] 10%|█         | 40005/400000 [00:05<00:46, 7708.52it/s] 10%|█         | 40781/400000 [00:05<00:46, 7721.60it/s] 10%|█         | 41561/400000 [00:05<00:46, 7744.17it/s] 11%|█         | 42345/400000 [00:05<00:46, 7769.87it/s] 11%|█         | 43126/400000 [00:05<00:45, 7781.60it/s] 11%|█         | 43905/400000 [00:05<00:46, 7597.48it/s] 11%|█         | 44710/400000 [00:05<00:45, 7726.76it/s] 11%|█▏        | 45536/400000 [00:05<00:44, 7877.25it/s] 12%|█▏        | 46334/400000 [00:06<00:44, 7905.88it/s] 12%|█▏        | 47126/400000 [00:06<00:46, 7602.17it/s] 12%|█▏        | 47947/400000 [00:06<00:45, 7773.05it/s] 12%|█▏        | 48728/400000 [00:06<00:46, 7630.64it/s] 12%|█▏        | 49504/400000 [00:06<00:45, 7668.59it/s] 13%|█▎        | 50273/400000 [00:06<00:46, 7510.85it/s] 13%|█▎        | 51027/400000 [00:06<00:47, 7395.29it/s] 13%|█▎        | 51775/400000 [00:06<00:46, 7419.18it/s] 13%|█▎        | 52563/400000 [00:06<00:46, 7549.83it/s] 13%|█▎        | 53353/400000 [00:07<00:45, 7649.85it/s] 14%|█▎        | 54120/400000 [00:07<00:45, 7650.97it/s] 14%|█▎        | 54887/400000 [00:07<00:45, 7556.26it/s] 14%|█▍        | 55664/400000 [00:07<00:45, 7617.92it/s] 14%|█▍        | 56440/400000 [00:07<00:44, 7658.02it/s] 14%|█▍        | 57207/400000 [00:07<00:44, 7628.93it/s] 14%|█▍        | 57971/400000 [00:07<00:45, 7578.47it/s] 15%|█▍        | 58741/400000 [00:07<00:44, 7614.40it/s] 15%|█▍        | 59523/400000 [00:07<00:44, 7673.28it/s] 15%|█▌        | 60310/400000 [00:07<00:43, 7730.28it/s] 15%|█▌        | 61118/400000 [00:08<00:43, 7831.41it/s] 15%|█▌        | 61902/400000 [00:08<00:43, 7701.59it/s] 16%|█▌        | 62674/400000 [00:08<00:44, 7608.74it/s] 16%|█▌        | 63489/400000 [00:08<00:43, 7761.52it/s] 16%|█▌        | 64335/400000 [00:08<00:42, 7955.92it/s] 16%|█▋        | 65136/400000 [00:08<00:42, 7972.02it/s] 17%|█▋        | 66015/400000 [00:08<00:40, 8200.46it/s] 17%|█▋        | 66838/400000 [00:08<00:41, 7987.71it/s] 17%|█▋        | 67640/400000 [00:08<00:42, 7874.08it/s] 17%|█▋        | 68430/400000 [00:08<00:42, 7782.46it/s] 17%|█▋        | 69275/400000 [00:09<00:41, 7971.10it/s] 18%|█▊        | 70075/400000 [00:09<00:42, 7803.35it/s] 18%|█▊        | 70858/400000 [00:09<00:42, 7733.64it/s] 18%|█▊        | 71634/400000 [00:09<00:42, 7726.89it/s] 18%|█▊        | 72409/400000 [00:09<00:42, 7711.35it/s] 18%|█▊        | 73219/400000 [00:09<00:41, 7822.39it/s] 19%|█▊        | 74003/400000 [00:09<00:42, 7666.01it/s] 19%|█▊        | 74772/400000 [00:09<00:43, 7507.14it/s] 19%|█▉        | 75528/400000 [00:09<00:43, 7522.34it/s] 19%|█▉        | 76351/400000 [00:09<00:41, 7720.25it/s] 19%|█▉        | 77126/400000 [00:10<00:41, 7712.07it/s] 19%|█▉        | 77899/400000 [00:10<00:42, 7623.20it/s] 20%|█▉        | 78663/400000 [00:10<00:43, 7349.43it/s] 20%|█▉        | 79430/400000 [00:10<00:43, 7441.28it/s] 20%|██        | 80246/400000 [00:10<00:41, 7642.06it/s] 20%|██        | 81014/400000 [00:10<00:41, 7606.31it/s] 20%|██        | 81855/400000 [00:10<00:40, 7828.55it/s] 21%|██        | 82641/400000 [00:10<00:41, 7733.55it/s] 21%|██        | 83443/400000 [00:10<00:40, 7815.42it/s] 21%|██        | 84255/400000 [00:11<00:39, 7901.67it/s] 21%|██▏       | 85111/400000 [00:11<00:38, 8086.78it/s] 21%|██▏       | 85922/400000 [00:11<00:40, 7766.05it/s] 22%|██▏       | 86703/400000 [00:11<00:41, 7635.61it/s] 22%|██▏       | 87471/400000 [00:11<00:43, 7215.88it/s] 22%|██▏       | 88216/400000 [00:11<00:42, 7282.75it/s] 22%|██▏       | 88950/400000 [00:11<00:42, 7258.68it/s] 22%|██▏       | 89683/400000 [00:11<00:42, 7277.59it/s] 23%|██▎       | 90455/400000 [00:11<00:41, 7403.91it/s] 23%|██▎       | 91248/400000 [00:11<00:40, 7551.76it/s] 23%|██▎       | 92090/400000 [00:12<00:39, 7791.07it/s] 23%|██▎       | 92873/400000 [00:12<00:39, 7765.87it/s] 23%|██▎       | 93653/400000 [00:12<00:40, 7587.34it/s] 24%|██▎       | 94415/400000 [00:12<00:40, 7474.80it/s] 24%|██▍       | 95215/400000 [00:12<00:39, 7624.05it/s] 24%|██▍       | 96062/400000 [00:12<00:38, 7858.38it/s] 24%|██▍       | 96852/400000 [00:12<00:38, 7782.91it/s] 24%|██▍       | 97633/400000 [00:12<00:39, 7621.63it/s] 25%|██▍       | 98418/400000 [00:12<00:39, 7687.33it/s] 25%|██▍       | 99189/400000 [00:12<00:39, 7689.26it/s] 25%|██▍       | 99970/400000 [00:13<00:38, 7722.79it/s] 25%|██▌       | 100794/400000 [00:13<00:38, 7869.63it/s] 25%|██▌       | 101583/400000 [00:13<00:38, 7704.86it/s] 26%|██▌       | 102356/400000 [00:13<00:39, 7596.84it/s] 26%|██▌       | 103118/400000 [00:13<00:39, 7505.33it/s] 26%|██▌       | 104000/400000 [00:13<00:37, 7856.24it/s] 26%|██▌       | 104859/400000 [00:13<00:36, 8062.08it/s] 26%|██▋       | 105671/400000 [00:13<00:37, 7941.91it/s] 27%|██▋       | 106470/400000 [00:13<00:37, 7806.63it/s] 27%|██▋       | 107254/400000 [00:14<00:38, 7611.39it/s] 27%|██▋       | 108082/400000 [00:14<00:37, 7798.89it/s] 27%|██▋       | 108882/400000 [00:14<00:37, 7857.26it/s] 27%|██▋       | 109678/400000 [00:14<00:36, 7886.71it/s] 28%|██▊       | 110469/400000 [00:14<00:36, 7859.57it/s] 28%|██▊       | 111261/400000 [00:14<00:36, 7876.18it/s] 28%|██▊       | 112050/400000 [00:14<00:37, 7776.62it/s] 28%|██▊       | 112844/400000 [00:14<00:36, 7824.63it/s] 28%|██▊       | 113628/400000 [00:14<00:36, 7761.05it/s] 29%|██▊       | 114405/400000 [00:14<00:37, 7624.87it/s] 29%|██▉       | 115267/400000 [00:15<00:36, 7897.52it/s] 29%|██▉       | 116060/400000 [00:15<00:36, 7874.86it/s] 29%|██▉       | 116874/400000 [00:15<00:35, 7949.48it/s] 29%|██▉       | 117704/400000 [00:15<00:35, 8049.46it/s] 30%|██▉       | 118511/400000 [00:15<00:34, 8045.87it/s] 30%|██▉       | 119349/400000 [00:15<00:34, 8142.69it/s] 30%|███       | 120165/400000 [00:15<00:34, 8132.96it/s] 30%|███       | 120980/400000 [00:15<00:35, 7830.07it/s] 30%|███       | 121766/400000 [00:15<00:35, 7810.27it/s] 31%|███       | 122550/400000 [00:15<00:35, 7750.53it/s] 31%|███       | 123393/400000 [00:16<00:34, 7940.79it/s] 31%|███       | 124218/400000 [00:16<00:34, 8028.53it/s] 31%|███▏      | 125023/400000 [00:16<00:34, 7962.76it/s] 31%|███▏      | 125878/400000 [00:16<00:33, 8130.24it/s] 32%|███▏      | 126693/400000 [00:16<00:34, 7939.21it/s] 32%|███▏      | 127490/400000 [00:16<00:34, 7929.99it/s] 32%|███▏      | 128285/400000 [00:16<00:35, 7737.59it/s] 32%|███▏      | 129065/400000 [00:16<00:34, 7755.36it/s] 32%|███▏      | 129843/400000 [00:16<00:36, 7460.62it/s] 33%|███▎      | 130593/400000 [00:16<00:36, 7422.09it/s] 33%|███▎      | 131369/400000 [00:17<00:35, 7519.38it/s] 33%|███▎      | 132209/400000 [00:17<00:34, 7760.94it/s] 33%|███▎      | 133034/400000 [00:17<00:33, 7899.50it/s] 33%|███▎      | 133827/400000 [00:17<00:33, 7884.52it/s] 34%|███▎      | 134675/400000 [00:17<00:32, 8053.20it/s] 34%|███▍      | 135491/400000 [00:17<00:32, 8082.63it/s] 34%|███▍      | 136306/400000 [00:17<00:32, 8101.02it/s] 34%|███▍      | 137118/400000 [00:17<00:32, 7992.65it/s] 34%|███▍      | 137919/400000 [00:17<00:33, 7932.38it/s] 35%|███▍      | 138714/400000 [00:18<00:34, 7615.66it/s] 35%|███▍      | 139479/400000 [00:18<00:34, 7570.59it/s] 35%|███▌      | 140243/400000 [00:18<00:34, 7589.94it/s] 35%|███▌      | 141004/400000 [00:18<00:34, 7575.08it/s] 35%|███▌      | 141832/400000 [00:18<00:33, 7772.20it/s] 36%|███▌      | 142651/400000 [00:18<00:32, 7890.72it/s] 36%|███▌      | 143501/400000 [00:18<00:31, 8062.47it/s] 36%|███▌      | 144335/400000 [00:18<00:31, 8141.87it/s] 36%|███▋      | 145151/400000 [00:18<00:31, 8057.49it/s] 36%|███▋      | 145959/400000 [00:18<00:31, 7957.92it/s] 37%|███▋      | 146757/400000 [00:19<00:32, 7710.24it/s] 37%|███▋      | 147555/400000 [00:19<00:32, 7787.66it/s] 37%|███▋      | 148336/400000 [00:19<00:32, 7691.11it/s] 37%|███▋      | 149107/400000 [00:19<00:32, 7671.28it/s] 37%|███▋      | 149905/400000 [00:19<00:32, 7758.97it/s] 38%|███▊      | 150696/400000 [00:19<00:31, 7801.88it/s] 38%|███▊      | 151504/400000 [00:19<00:31, 7883.11it/s] 38%|███▊      | 152312/400000 [00:19<00:31, 7938.86it/s] 38%|███▊      | 153107/400000 [00:19<00:31, 7739.68it/s] 38%|███▊      | 153883/400000 [00:19<00:31, 7698.38it/s] 39%|███▊      | 154654/400000 [00:20<00:32, 7497.46it/s] 39%|███▉      | 155460/400000 [00:20<00:31, 7657.47it/s] 39%|███▉      | 156300/400000 [00:20<00:30, 7863.91it/s] 39%|███▉      | 157098/400000 [00:20<00:30, 7896.89it/s] 39%|███▉      | 157890/400000 [00:20<00:30, 7864.25it/s] 40%|███▉      | 158723/400000 [00:20<00:30, 7997.59it/s] 40%|███▉      | 159525/400000 [00:20<00:30, 7901.57it/s] 40%|████      | 160403/400000 [00:20<00:29, 8141.00it/s] 40%|████      | 161220/400000 [00:20<00:29, 8053.20it/s] 41%|████      | 162028/400000 [00:20<00:29, 8019.26it/s] 41%|████      | 162832/400000 [00:21<00:30, 7810.59it/s] 41%|████      | 163616/400000 [00:21<00:30, 7666.66it/s] 41%|████      | 164425/400000 [00:21<00:30, 7786.74it/s] 41%|████▏     | 165206/400000 [00:21<00:30, 7730.26it/s] 41%|████▏     | 165981/400000 [00:21<00:30, 7667.59it/s] 42%|████▏     | 166769/400000 [00:21<00:30, 7729.75it/s] 42%|████▏     | 167634/400000 [00:21<00:29, 7983.24it/s] 42%|████▏     | 168436/400000 [00:21<00:29, 7947.61it/s] 42%|████▏     | 169254/400000 [00:21<00:28, 8013.07it/s] 43%|████▎     | 170057/400000 [00:22<00:28, 7969.58it/s] 43%|████▎     | 170910/400000 [00:22<00:28, 8128.58it/s] 43%|████▎     | 171725/400000 [00:22<00:28, 7882.69it/s] 43%|████▎     | 172516/400000 [00:22<00:29, 7765.95it/s] 43%|████▎     | 173295/400000 [00:22<00:29, 7569.01it/s] 44%|████▎     | 174086/400000 [00:22<00:29, 7663.85it/s] 44%|████▎     | 174855/400000 [00:22<00:29, 7653.41it/s] 44%|████▍     | 175754/400000 [00:22<00:27, 8009.95it/s] 44%|████▍     | 176561/400000 [00:22<00:28, 7947.23it/s] 44%|████▍     | 177360/400000 [00:22<00:28, 7949.37it/s] 45%|████▍     | 178185/400000 [00:23<00:27, 8034.92it/s] 45%|████▍     | 178991/400000 [00:23<00:27, 7994.54it/s] 45%|████▍     | 179792/400000 [00:23<00:28, 7705.38it/s] 45%|████▌     | 180568/400000 [00:23<00:28, 7720.21it/s] 45%|████▌     | 181343/400000 [00:23<00:28, 7681.51it/s] 46%|████▌     | 182113/400000 [00:23<00:28, 7617.02it/s] 46%|████▌     | 182905/400000 [00:23<00:28, 7703.37it/s] 46%|████▌     | 183677/400000 [00:23<00:28, 7645.17it/s] 46%|████▌     | 184443/400000 [00:23<00:28, 7625.80it/s] 46%|████▋     | 185207/400000 [00:23<00:28, 7586.08it/s] 47%|████▋     | 186004/400000 [00:24<00:27, 7696.78it/s] 47%|████▋     | 186775/400000 [00:24<00:28, 7577.49it/s] 47%|████▋     | 187534/400000 [00:24<00:28, 7526.48it/s] 47%|████▋     | 188308/400000 [00:24<00:27, 7587.82it/s] 47%|████▋     | 189068/400000 [00:24<00:28, 7397.01it/s] 47%|████▋     | 189829/400000 [00:24<00:28, 7458.84it/s] 48%|████▊     | 190594/400000 [00:24<00:27, 7511.99it/s] 48%|████▊     | 191347/400000 [00:24<00:28, 7314.20it/s] 48%|████▊     | 192130/400000 [00:24<00:27, 7461.28it/s] 48%|████▊     | 192879/400000 [00:24<00:28, 7371.36it/s] 48%|████▊     | 193648/400000 [00:25<00:27, 7463.49it/s] 49%|████▊     | 194410/400000 [00:25<00:27, 7509.00it/s] 49%|████▉     | 195162/400000 [00:25<00:27, 7487.04it/s] 49%|████▉     | 195918/400000 [00:25<00:27, 7506.49it/s] 49%|████▉     | 196675/400000 [00:25<00:27, 7524.67it/s] 49%|████▉     | 197465/400000 [00:25<00:26, 7631.83it/s] 50%|████▉     | 198229/400000 [00:25<00:26, 7632.30it/s] 50%|████▉     | 198993/400000 [00:25<00:26, 7619.92it/s] 50%|████▉     | 199816/400000 [00:25<00:25, 7789.71it/s] 50%|█████     | 200597/400000 [00:26<00:25, 7726.38it/s] 50%|█████     | 201371/400000 [00:26<00:26, 7589.65it/s] 51%|█████     | 202155/400000 [00:26<00:25, 7661.12it/s] 51%|█████     | 202929/400000 [00:26<00:25, 7682.98it/s] 51%|█████     | 203699/400000 [00:26<00:25, 7600.46it/s] 51%|█████     | 204478/400000 [00:26<00:25, 7655.32it/s] 51%|█████▏    | 205245/400000 [00:26<00:25, 7573.54it/s] 52%|█████▏    | 206003/400000 [00:26<00:25, 7536.89it/s] 52%|█████▏    | 206783/400000 [00:26<00:25, 7613.03it/s] 52%|█████▏    | 207559/400000 [00:26<00:25, 7654.93it/s] 52%|█████▏    | 208354/400000 [00:27<00:24, 7739.23it/s] 52%|█████▏    | 209129/400000 [00:27<00:24, 7717.66it/s] 52%|█████▏    | 209902/400000 [00:27<00:24, 7675.03it/s] 53%|█████▎    | 210699/400000 [00:27<00:24, 7758.96it/s] 53%|█████▎    | 211479/400000 [00:27<00:24, 7770.09it/s] 53%|█████▎    | 212257/400000 [00:27<00:24, 7732.40it/s] 53%|█████▎    | 213031/400000 [00:27<00:24, 7639.21it/s] 53%|█████▎    | 213838/400000 [00:27<00:23, 7761.92it/s] 54%|█████▎    | 214668/400000 [00:27<00:23, 7915.31it/s] 54%|█████▍    | 215461/400000 [00:27<00:23, 7797.57it/s] 54%|█████▍    | 216243/400000 [00:28<00:23, 7773.68it/s] 54%|█████▍    | 217057/400000 [00:28<00:23, 7878.11it/s] 54%|█████▍    | 217846/400000 [00:28<00:23, 7786.78it/s] 55%|█████▍    | 218626/400000 [00:28<00:23, 7702.33it/s] 55%|█████▍    | 219398/400000 [00:28<00:23, 7681.95it/s] 55%|█████▌    | 220167/400000 [00:28<00:23, 7649.88it/s] 55%|█████▌    | 220933/400000 [00:28<00:23, 7638.19it/s] 55%|█████▌    | 221698/400000 [00:28<00:23, 7564.17it/s] 56%|█████▌    | 222455/400000 [00:28<00:23, 7523.12it/s] 56%|█████▌    | 223208/400000 [00:28<00:24, 7341.18it/s] 56%|█████▌    | 223944/400000 [00:29<00:24, 7329.08it/s] 56%|█████▌    | 224683/400000 [00:29<00:23, 7344.97it/s] 56%|█████▋    | 225442/400000 [00:29<00:23, 7416.09it/s] 57%|█████▋    | 226203/400000 [00:29<00:23, 7470.65it/s] 57%|█████▋    | 226969/400000 [00:29<00:22, 7526.03it/s] 57%|█████▋    | 227723/400000 [00:29<00:23, 7439.04it/s] 57%|█████▋    | 228492/400000 [00:29<00:22, 7512.43it/s] 57%|█████▋    | 229253/400000 [00:29<00:22, 7540.87it/s] 58%|█████▊    | 230009/400000 [00:29<00:22, 7545.69it/s] 58%|█████▊    | 230764/400000 [00:29<00:22, 7450.58it/s] 58%|█████▊    | 231510/400000 [00:30<00:23, 7316.66it/s] 58%|█████▊    | 232304/400000 [00:30<00:22, 7490.99it/s] 58%|█████▊    | 233102/400000 [00:30<00:21, 7630.81it/s] 58%|█████▊    | 233885/400000 [00:30<00:21, 7687.78it/s] 59%|█████▊    | 234656/400000 [00:30<00:21, 7673.81it/s] 59%|█████▉    | 235425/400000 [00:30<00:21, 7582.20it/s] 59%|█████▉    | 236185/400000 [00:30<00:21, 7491.85it/s] 59%|█████▉    | 236936/400000 [00:30<00:22, 7333.20it/s] 59%|█████▉    | 237711/400000 [00:30<00:21, 7451.93it/s] 60%|█████▉    | 238458/400000 [00:30<00:21, 7442.74it/s] 60%|█████▉    | 239234/400000 [00:31<00:21, 7533.31it/s] 60%|█████▉    | 239989/400000 [00:31<00:21, 7497.33it/s] 60%|██████    | 240782/400000 [00:31<00:20, 7619.64it/s] 60%|██████    | 241545/400000 [00:31<00:20, 7555.68it/s] 61%|██████    | 242302/400000 [00:31<00:20, 7512.53it/s] 61%|██████    | 243054/400000 [00:31<00:21, 7365.15it/s] 61%|██████    | 243800/400000 [00:31<00:21, 7392.23it/s] 61%|██████    | 244541/400000 [00:31<00:21, 7396.54it/s] 61%|██████▏   | 245290/400000 [00:31<00:20, 7423.21it/s] 62%|██████▏   | 246037/400000 [00:32<00:20, 7436.18it/s] 62%|██████▏   | 246781/400000 [00:32<00:20, 7417.83it/s] 62%|██████▏   | 247545/400000 [00:32<00:20, 7481.84it/s] 62%|██████▏   | 248351/400000 [00:32<00:19, 7644.58it/s] 62%|██████▏   | 249117/400000 [00:32<00:19, 7612.00it/s] 62%|██████▏   | 249901/400000 [00:32<00:19, 7678.80it/s] 63%|██████▎   | 250670/400000 [00:32<00:19, 7525.05it/s] 63%|██████▎   | 251439/400000 [00:32<00:19, 7573.19it/s] 63%|██████▎   | 252229/400000 [00:32<00:19, 7666.25it/s] 63%|██████▎   | 252997/400000 [00:32<00:19, 7640.12it/s] 63%|██████▎   | 253791/400000 [00:33<00:18, 7725.62it/s] 64%|██████▎   | 254565/400000 [00:33<00:19, 7572.47it/s] 64%|██████▍   | 255324/400000 [00:33<00:19, 7489.35it/s] 64%|██████▍   | 256087/400000 [00:33<00:19, 7529.76it/s] 64%|██████▍   | 256865/400000 [00:33<00:18, 7600.42it/s] 64%|██████▍   | 257626/400000 [00:33<00:19, 7478.86it/s] 65%|██████▍   | 258386/400000 [00:33<00:18, 7512.45it/s] 65%|██████▍   | 259138/400000 [00:33<00:18, 7448.67it/s] 65%|██████▍   | 259944/400000 [00:33<00:18, 7620.46it/s] 65%|██████▌   | 260711/400000 [00:33<00:18, 7632.37it/s] 65%|██████▌   | 261476/400000 [00:34<00:18, 7600.89it/s] 66%|██████▌   | 262237/400000 [00:34<00:18, 7584.41it/s] 66%|██████▌   | 262996/400000 [00:34<00:18, 7398.59it/s] 66%|██████▌   | 263786/400000 [00:34<00:18, 7542.03it/s] 66%|██████▌   | 264566/400000 [00:34<00:17, 7614.19it/s] 66%|██████▋   | 265357/400000 [00:34<00:17, 7698.89it/s] 67%|██████▋   | 266129/400000 [00:34<00:17, 7624.19it/s] 67%|██████▋   | 266920/400000 [00:34<00:17, 7707.69it/s] 67%|██████▋   | 267692/400000 [00:34<00:17, 7617.87it/s] 67%|██████▋   | 268459/400000 [00:34<00:17, 7632.84it/s] 67%|██████▋   | 269232/400000 [00:35<00:17, 7658.77it/s] 67%|██████▋   | 269999/400000 [00:35<00:17, 7521.40it/s] 68%|██████▊   | 270753/400000 [00:35<00:17, 7444.63it/s] 68%|██████▊   | 271499/400000 [00:35<00:17, 7447.18it/s] 68%|██████▊   | 272281/400000 [00:35<00:16, 7554.65it/s] 68%|██████▊   | 273038/400000 [00:35<00:16, 7503.19it/s] 68%|██████▊   | 273790/400000 [00:35<00:16, 7507.78it/s] 69%|██████▊   | 274607/400000 [00:35<00:16, 7694.31it/s] 69%|██████▉   | 275378/400000 [00:35<00:16, 7635.17it/s] 69%|██████▉   | 276191/400000 [00:35<00:15, 7775.28it/s] 69%|██████▉   | 277036/400000 [00:36<00:15, 7964.83it/s] 69%|██████▉   | 277835/400000 [00:36<00:15, 7795.15it/s] 70%|██████▉   | 278617/400000 [00:36<00:15, 7760.03it/s] 70%|██████▉   | 279395/400000 [00:36<00:15, 7758.14it/s] 70%|███████   | 280182/400000 [00:36<00:15, 7790.44it/s] 70%|███████   | 280973/400000 [00:36<00:15, 7825.30it/s] 70%|███████   | 281757/400000 [00:36<00:15, 7733.75it/s] 71%|███████   | 282532/400000 [00:36<00:15, 7719.64it/s] 71%|███████   | 283305/400000 [00:36<00:15, 7612.87it/s] 71%|███████   | 284107/400000 [00:36<00:14, 7729.99it/s] 71%|███████   | 284892/400000 [00:37<00:14, 7763.35it/s] 71%|███████▏  | 285670/400000 [00:37<00:14, 7637.29it/s] 72%|███████▏  | 286435/400000 [00:37<00:15, 7366.12it/s] 72%|███████▏  | 287175/400000 [00:37<00:15, 7307.00it/s] 72%|███████▏  | 287961/400000 [00:37<00:15, 7462.55it/s] 72%|███████▏  | 288744/400000 [00:37<00:14, 7568.32it/s] 72%|███████▏  | 289503/400000 [00:37<00:14, 7531.65it/s] 73%|███████▎  | 290294/400000 [00:37<00:14, 7639.37it/s] 73%|███████▎  | 291060/400000 [00:37<00:14, 7508.09it/s] 73%|███████▎  | 291813/400000 [00:38<00:14, 7495.57it/s] 73%|███████▎  | 292564/400000 [00:38<00:14, 7477.72it/s] 73%|███████▎  | 293359/400000 [00:38<00:14, 7612.61it/s] 74%|███████▎  | 294122/400000 [00:38<00:14, 7548.55it/s] 74%|███████▎  | 294878/400000 [00:38<00:13, 7540.10it/s] 74%|███████▍  | 295633/400000 [00:38<00:13, 7470.64it/s] 74%|███████▍  | 296422/400000 [00:38<00:13, 7590.35it/s] 74%|███████▍  | 297187/400000 [00:38<00:13, 7606.48it/s] 75%|███████▍  | 298002/400000 [00:38<00:13, 7761.54it/s] 75%|███████▍  | 298817/400000 [00:38<00:12, 7873.85it/s] 75%|███████▍  | 299606/400000 [00:39<00:13, 7666.70it/s] 75%|███████▌  | 300375/400000 [00:39<00:13, 7562.89it/s] 75%|███████▌  | 301141/400000 [00:39<00:13, 7589.40it/s] 75%|███████▌  | 301902/400000 [00:39<00:13, 7441.57it/s] 76%|███████▌  | 302648/400000 [00:39<00:13, 7428.75it/s] 76%|███████▌  | 303448/400000 [00:39<00:12, 7589.80it/s] 76%|███████▌  | 304209/400000 [00:39<00:12, 7577.23it/s] 76%|███████▌  | 304968/400000 [00:39<00:12, 7526.05it/s] 76%|███████▋  | 305748/400000 [00:39<00:12, 7604.94it/s] 77%|███████▋  | 306628/400000 [00:39<00:11, 7926.39it/s] 77%|███████▋  | 307425/400000 [00:40<00:11, 7732.92it/s] 77%|███████▋  | 308203/400000 [00:40<00:11, 7711.30it/s] 77%|███████▋  | 308977/400000 [00:40<00:11, 7627.76it/s] 77%|███████▋  | 309742/400000 [00:40<00:11, 7587.36it/s] 78%|███████▊  | 310505/400000 [00:40<00:11, 7598.88it/s] 78%|███████▊  | 311280/400000 [00:40<00:11, 7642.59it/s] 78%|███████▊  | 312084/400000 [00:40<00:11, 7757.33it/s] 78%|███████▊  | 312861/400000 [00:40<00:11, 7618.36it/s] 78%|███████▊  | 313625/400000 [00:40<00:11, 7618.71it/s] 79%|███████▊  | 314393/400000 [00:40<00:11, 7635.20it/s] 79%|███████▉  | 315158/400000 [00:41<00:11, 7598.94it/s] 79%|███████▉  | 315934/400000 [00:41<00:10, 7645.63it/s] 79%|███████▉  | 316704/400000 [00:41<00:10, 7661.25it/s] 79%|███████▉  | 317471/400000 [00:41<00:10, 7591.39it/s] 80%|███████▉  | 318239/400000 [00:41<00:10, 7617.13it/s] 80%|███████▉  | 319049/400000 [00:41<00:10, 7755.81it/s] 80%|███████▉  | 319826/400000 [00:41<00:10, 7758.42it/s] 80%|████████  | 320603/400000 [00:41<00:10, 7677.15it/s] 80%|████████  | 321372/400000 [00:41<00:10, 7534.77it/s] 81%|████████  | 322173/400000 [00:41<00:10, 7670.97it/s] 81%|████████  | 322942/400000 [00:42<00:10, 7664.10it/s] 81%|████████  | 323719/400000 [00:42<00:09, 7694.79it/s] 81%|████████  | 324490/400000 [00:42<00:09, 7663.81it/s] 81%|████████▏ | 325257/400000 [00:42<00:09, 7574.91it/s] 82%|████████▏ | 326016/400000 [00:42<00:09, 7544.78it/s] 82%|████████▏ | 326830/400000 [00:42<00:09, 7710.27it/s] 82%|████████▏ | 327603/400000 [00:42<00:09, 7606.71it/s] 82%|████████▏ | 328391/400000 [00:42<00:09, 7686.06it/s] 82%|████████▏ | 329207/400000 [00:42<00:09, 7820.29it/s] 82%|████████▏ | 329991/400000 [00:43<00:09, 7720.99it/s] 83%|████████▎ | 330765/400000 [00:43<00:09, 7659.29it/s] 83%|████████▎ | 331532/400000 [00:43<00:08, 7646.92it/s] 83%|████████▎ | 332298/400000 [00:43<00:08, 7588.77it/s] 83%|████████▎ | 333058/400000 [00:43<00:08, 7587.24it/s] 83%|████████▎ | 333836/400000 [00:43<00:08, 7643.48it/s] 84%|████████▎ | 334638/400000 [00:43<00:08, 7752.63it/s] 84%|████████▍ | 335414/400000 [00:43<00:08, 7652.62it/s] 84%|████████▍ | 336214/400000 [00:43<00:08, 7753.05it/s] 84%|████████▍ | 337052/400000 [00:43<00:07, 7928.39it/s] 84%|████████▍ | 337847/400000 [00:44<00:07, 7916.46it/s] 85%|████████▍ | 338640/400000 [00:44<00:07, 7821.57it/s] 85%|████████▍ | 339454/400000 [00:44<00:07, 7911.65it/s] 85%|████████▌ | 340247/400000 [00:44<00:07, 7847.36it/s] 85%|████████▌ | 341033/400000 [00:44<00:07, 7683.60it/s] 85%|████████▌ | 341808/400000 [00:44<00:07, 7703.04it/s] 86%|████████▌ | 342647/400000 [00:44<00:07, 7895.32it/s] 86%|████████▌ | 343439/400000 [00:44<00:07, 7890.28it/s] 86%|████████▌ | 344230/400000 [00:44<00:07, 7752.04it/s] 86%|████████▋ | 345029/400000 [00:44<00:07, 7820.13it/s] 86%|████████▋ | 345813/400000 [00:45<00:07, 7724.96it/s] 87%|████████▋ | 346591/400000 [00:45<00:06, 7738.92it/s] 87%|████████▋ | 347366/400000 [00:45<00:06, 7659.38it/s] 87%|████████▋ | 348216/400000 [00:45<00:06, 7892.94it/s] 87%|████████▋ | 349008/400000 [00:45<00:06, 7806.50it/s] 87%|████████▋ | 349791/400000 [00:45<00:06, 7755.45it/s] 88%|████████▊ | 350588/400000 [00:45<00:06, 7816.45it/s] 88%|████████▊ | 351416/400000 [00:45<00:06, 7948.60it/s] 88%|████████▊ | 352243/400000 [00:45<00:05, 8040.70it/s] 88%|████████▊ | 353049/400000 [00:45<00:05, 7837.11it/s] 88%|████████▊ | 353881/400000 [00:46<00:05, 7974.50it/s] 89%|████████▊ | 354681/400000 [00:46<00:05, 7858.05it/s] 89%|████████▉ | 355469/400000 [00:46<00:05, 7802.10it/s] 89%|████████▉ | 356251/400000 [00:46<00:05, 7699.65it/s] 89%|████████▉ | 357031/400000 [00:46<00:05, 7729.36it/s] 89%|████████▉ | 357805/400000 [00:46<00:05, 7696.79it/s] 90%|████████▉ | 358590/400000 [00:46<00:05, 7740.11it/s] 90%|████████▉ | 359365/400000 [00:46<00:05, 7723.72it/s] 90%|█████████ | 360138/400000 [00:46<00:05, 7705.10it/s] 90%|█████████ | 360909/400000 [00:46<00:05, 7662.66it/s] 90%|█████████ | 361676/400000 [00:47<00:05, 7568.11it/s] 91%|█████████ | 362466/400000 [00:47<00:04, 7663.06it/s] 91%|█████████ | 363233/400000 [00:47<00:04, 7467.52it/s] 91%|█████████ | 363984/400000 [00:47<00:04, 7479.75it/s] 91%|█████████ | 364761/400000 [00:47<00:04, 7563.90it/s] 91%|█████████▏| 365546/400000 [00:47<00:04, 7647.20it/s] 92%|█████████▏| 366312/400000 [00:47<00:04, 7629.42it/s] 92%|█████████▏| 367086/400000 [00:47<00:04, 7661.82it/s] 92%|█████████▏| 367853/400000 [00:47<00:04, 7536.14it/s] 92%|█████████▏| 368608/400000 [00:48<00:04, 7532.14it/s] 92%|█████████▏| 369362/400000 [00:48<00:04, 7527.46it/s] 93%|█████████▎| 370153/400000 [00:48<00:03, 7636.86it/s] 93%|█████████▎| 370976/400000 [00:48<00:03, 7802.14it/s] 93%|█████████▎| 371758/400000 [00:48<00:03, 7716.46it/s] 93%|█████████▎| 372531/400000 [00:48<00:03, 7608.39it/s] 93%|█████████▎| 373295/400000 [00:48<00:03, 7615.36it/s] 94%|█████████▎| 374071/400000 [00:48<00:03, 7657.78it/s] 94%|█████████▎| 374838/400000 [00:48<00:03, 7632.30it/s] 94%|█████████▍| 375649/400000 [00:48<00:03, 7768.50it/s] 94%|█████████▍| 376427/400000 [00:49<00:03, 7583.62it/s] 94%|█████████▍| 377187/400000 [00:49<00:03, 7558.55it/s] 94%|█████████▍| 377945/400000 [00:49<00:02, 7531.10it/s] 95%|█████████▍| 378844/400000 [00:49<00:02, 7916.21it/s] 95%|█████████▍| 379663/400000 [00:49<00:02, 7994.72it/s] 95%|█████████▌| 380467/400000 [00:49<00:02, 7716.27it/s] 95%|█████████▌| 381244/400000 [00:49<00:02, 7331.96it/s] 96%|█████████▌| 382002/400000 [00:49<00:02, 7402.77it/s] 96%|█████████▌| 382748/400000 [00:49<00:02, 7405.72it/s] 96%|█████████▌| 383493/400000 [00:49<00:02, 7301.28it/s] 96%|█████████▌| 384227/400000 [00:50<00:02, 7228.26it/s] 96%|█████████▌| 384953/400000 [00:50<00:02, 6945.52it/s] 96%|█████████▋| 385705/400000 [00:50<00:02, 7106.83it/s] 97%|█████████▋| 386508/400000 [00:50<00:01, 7359.76it/s] 97%|█████████▋| 387270/400000 [00:50<00:01, 7434.33it/s] 97%|█████████▋| 388038/400000 [00:50<00:01, 7505.91it/s] 97%|█████████▋| 388835/400000 [00:50<00:01, 7636.90it/s] 97%|█████████▋| 389625/400000 [00:50<00:01, 7712.16it/s] 98%|█████████▊| 390448/400000 [00:50<00:01, 7858.37it/s] 98%|█████████▊| 391236/400000 [00:50<00:01, 7813.35it/s] 98%|█████████▊| 392019/400000 [00:51<00:01, 7736.74it/s] 98%|█████████▊| 392794/400000 [00:51<00:00, 7648.81it/s] 98%|█████████▊| 393560/400000 [00:51<00:00, 7620.06it/s] 99%|█████████▊| 394402/400000 [00:51<00:00, 7841.24it/s] 99%|█████████▉| 395189/400000 [00:51<00:00, 7685.29it/s] 99%|█████████▉| 395960/400000 [00:51<00:00, 7616.50it/s] 99%|█████████▉| 396773/400000 [00:51<00:00, 7762.94it/s] 99%|█████████▉| 397552/400000 [00:51<00:00, 7520.96it/s]100%|█████████▉| 398308/400000 [00:51<00:00, 7276.75it/s]100%|█████████▉| 399040/400000 [00:52<00:00, 7234.07it/s]100%|█████████▉| 399831/400000 [00:52<00:00, 7423.39it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7670.31it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f04dc870128>, <torchtext.data.dataset.TabularDataset object at 0x7f04dc870278>, <torchtext.vocab.Vocab object at 0x7f04dc870198>), {}) 

