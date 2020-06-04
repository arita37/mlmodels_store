
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdc659d5378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdc659d5378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fdcd7a260d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fdcd7a260d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdc6f1b99d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdc6f1b99d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc8530a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc8530a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc8530a8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 85%|████████▌ | 8445952/9912422 [00:00<00:00, 84437455.92it/s]9920512it [00:00, 38994770.82it/s]                             
0it [00:00, ?it/s]32768it [00:00, 849038.48it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 10487179.60it/s]         
0it [00:00, ?it/s]8192it [00:00, 182164.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc6e8c29e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc689a4fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fdc8530a510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fdc8530a510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fdc8530a510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:07,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:07,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:06,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:06,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:05,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:05,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:45,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:45,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:44,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:44,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:44,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:43,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:28,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:28,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:18,  6.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:12,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:12,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:12,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:12,  9.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 36.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.75 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.75 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.81 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ url]
0 examples [00:00, ? examples/s]2020-06-04 06:08:44.708960: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-04 06:08:44.722284: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095090000 Hz
2020-06-04 06:08:44.722465: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560225bc1020 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-04 06:08:44.722483: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
26 examples [00:00, 259.25 examples/s]130 examples [00:00, 334.29 examples/s]235 examples [00:00, 419.78 examples/s]333 examples [00:00, 506.13 examples/s]440 examples [00:00, 600.44 examples/s]551 examples [00:00, 696.02 examples/s]660 examples [00:00, 780.51 examples/s]771 examples [00:00, 855.34 examples/s]881 examples [00:00, 914.86 examples/s]990 examples [00:01, 960.78 examples/s]1102 examples [00:01, 1002.69 examples/s]1209 examples [00:01, 1002.97 examples/s]1314 examples [00:01, 1006.28 examples/s]1423 examples [00:01, 1028.16 examples/s]1533 examples [00:01, 1046.89 examples/s]1640 examples [00:01, 1039.00 examples/s]1753 examples [00:01, 1064.65 examples/s]1865 examples [00:01, 1080.33 examples/s]1974 examples [00:01, 1076.14 examples/s]2083 examples [00:02, 1064.25 examples/s]2192 examples [00:02, 1071.65 examples/s]2300 examples [00:02, 1051.76 examples/s]2406 examples [00:02, 1044.91 examples/s]2512 examples [00:02, 1048.14 examples/s]2620 examples [00:02, 1057.35 examples/s]2731 examples [00:02, 1071.53 examples/s]2841 examples [00:02, 1078.13 examples/s]2949 examples [00:02, 1068.71 examples/s]3056 examples [00:02, 1060.41 examples/s]3163 examples [00:03, 1039.30 examples/s]3272 examples [00:03, 1053.05 examples/s]3378 examples [00:03, 1052.36 examples/s]3484 examples [00:03, 1051.73 examples/s]3593 examples [00:03, 1062.55 examples/s]3700 examples [00:03, 1052.67 examples/s]3808 examples [00:03, 1059.22 examples/s]3914 examples [00:03, 1032.89 examples/s]4027 examples [00:03, 1058.03 examples/s]4134 examples [00:03, 1050.74 examples/s]4241 examples [00:04, 1055.07 examples/s]4347 examples [00:04, 1052.10 examples/s]4461 examples [00:04, 1075.46 examples/s]4569 examples [00:04, 1068.27 examples/s]4679 examples [00:04, 1077.53 examples/s]4794 examples [00:04, 1096.92 examples/s]4904 examples [00:04, 1058.12 examples/s]5016 examples [00:04, 1073.73 examples/s]5124 examples [00:04, 1069.32 examples/s]5232 examples [00:05, 1051.78 examples/s]5341 examples [00:05, 1061.63 examples/s]5456 examples [00:05, 1086.40 examples/s]5566 examples [00:05, 1085.41 examples/s]5675 examples [00:05, 1063.88 examples/s]5787 examples [00:05, 1077.95 examples/s]5898 examples [00:05, 1087.07 examples/s]6007 examples [00:05, 1086.75 examples/s]6118 examples [00:05, 1093.55 examples/s]6231 examples [00:05, 1103.74 examples/s]6342 examples [00:06, 1105.12 examples/s]6453 examples [00:06, 1104.23 examples/s]6567 examples [00:06, 1112.61 examples/s]6679 examples [00:06, 1105.74 examples/s]6790 examples [00:06, 1092.85 examples/s]6900 examples [00:06, 1048.03 examples/s]7011 examples [00:06, 1064.84 examples/s]7124 examples [00:06, 1083.52 examples/s]7237 examples [00:06, 1097.01 examples/s]7347 examples [00:06, 1040.20 examples/s]7452 examples [00:07, 1039.22 examples/s]7557 examples [00:07, 1030.62 examples/s]7670 examples [00:07, 1057.79 examples/s]7777 examples [00:07, 1046.16 examples/s]7882 examples [00:07, 1046.46 examples/s]7993 examples [00:07, 1062.51 examples/s]8103 examples [00:07, 1073.45 examples/s]8211 examples [00:07, 1039.53 examples/s]8320 examples [00:07, 1052.82 examples/s]8428 examples [00:07, 1058.42 examples/s]8543 examples [00:08, 1081.88 examples/s]8652 examples [00:08, 1047.75 examples/s]8763 examples [00:08, 1064.13 examples/s]8872 examples [00:08, 1069.01 examples/s]8980 examples [00:08, 1029.69 examples/s]9084 examples [00:08, 1019.99 examples/s]9187 examples [00:08, 1000.44 examples/s]9288 examples [00:08, 1003.08 examples/s]9392 examples [00:08, 1012.79 examples/s]9497 examples [00:09, 1010.16 examples/s]9608 examples [00:09, 1035.75 examples/s]9712 examples [00:09, 1009.28 examples/s]9823 examples [00:09, 1037.30 examples/s]9928 examples [00:09, 1038.96 examples/s]10033 examples [00:09, 998.16 examples/s]10143 examples [00:09, 1026.62 examples/s]10254 examples [00:09, 1048.98 examples/s]10365 examples [00:09, 1064.56 examples/s]10472 examples [00:09, 1058.21 examples/s]10581 examples [00:10, 1065.37 examples/s]10688 examples [00:10, 1057.66 examples/s]10797 examples [00:10, 1065.97 examples/s]10904 examples [00:10, 1039.99 examples/s]11012 examples [00:10, 1049.00 examples/s]11124 examples [00:10, 1068.18 examples/s]11232 examples [00:10, 1068.75 examples/s]11345 examples [00:10, 1081.48 examples/s]11454 examples [00:10, 1043.38 examples/s]11559 examples [00:10, 1037.33 examples/s]11664 examples [00:11, 1013.36 examples/s]11766 examples [00:11, 975.62 examples/s] 11865 examples [00:11, 959.41 examples/s]11965 examples [00:11, 968.81 examples/s]12063 examples [00:11, 967.26 examples/s]12174 examples [00:11, 1005.67 examples/s]12284 examples [00:11, 1030.64 examples/s]12388 examples [00:11, 1030.09 examples/s]12496 examples [00:11, 1043.98 examples/s]12605 examples [00:12, 1055.20 examples/s]12719 examples [00:12, 1078.41 examples/s]12832 examples [00:12, 1092.64 examples/s]12942 examples [00:12, 1079.97 examples/s]13051 examples [00:12, 1050.33 examples/s]13160 examples [00:12, 1059.46 examples/s]13272 examples [00:12, 1076.78 examples/s]13392 examples [00:12, 1110.21 examples/s]13508 examples [00:12, 1124.40 examples/s]13623 examples [00:12, 1131.57 examples/s]13737 examples [00:13, 1118.07 examples/s]13850 examples [00:13, 1111.04 examples/s]13968 examples [00:13, 1128.19 examples/s]14081 examples [00:13, 1094.58 examples/s]14192 examples [00:13, 1096.49 examples/s]14303 examples [00:13, 1100.02 examples/s]14417 examples [00:13, 1109.82 examples/s]14529 examples [00:13, 1100.42 examples/s]14640 examples [00:13, 1072.25 examples/s]14757 examples [00:13, 1099.33 examples/s]14868 examples [00:14, 1099.97 examples/s]14979 examples [00:14, 1098.81 examples/s]15091 examples [00:14, 1103.36 examples/s]15202 examples [00:14, 1102.13 examples/s]15313 examples [00:14, 1081.49 examples/s]15422 examples [00:14, 1068.91 examples/s]15532 examples [00:14, 1077.76 examples/s]15642 examples [00:14, 1083.03 examples/s]15752 examples [00:14, 1087.48 examples/s]15861 examples [00:14, 1086.43 examples/s]15970 examples [00:15, 1069.71 examples/s]16078 examples [00:15, 1064.95 examples/s]16185 examples [00:15, 1045.31 examples/s]16290 examples [00:15, 1005.08 examples/s]16392 examples [00:15, 1008.68 examples/s]16501 examples [00:15, 1031.55 examples/s]16610 examples [00:15, 1045.69 examples/s]16724 examples [00:15, 1069.66 examples/s]16832 examples [00:15, 1072.34 examples/s]16940 examples [00:16, 1073.65 examples/s]17048 examples [00:16, 1043.57 examples/s]17153 examples [00:16, 990.93 examples/s] 17253 examples [00:16, 956.49 examples/s]17354 examples [00:16, 970.49 examples/s]17452 examples [00:16, 952.08 examples/s]17558 examples [00:16, 980.33 examples/s]17666 examples [00:16, 1005.96 examples/s]17768 examples [00:16, 993.45 examples/s] 17868 examples [00:16, 977.36 examples/s]17975 examples [00:17, 1000.95 examples/s]18079 examples [00:17, 1011.25 examples/s]18185 examples [00:17, 1023.23 examples/s]18290 examples [00:17, 1030.86 examples/s]18397 examples [00:17, 1039.44 examples/s]18502 examples [00:17, 1019.42 examples/s]18605 examples [00:17, 1016.85 examples/s]18711 examples [00:17, 1027.83 examples/s]18819 examples [00:17, 1041.46 examples/s]18924 examples [00:17, 1042.98 examples/s]19032 examples [00:18, 1047.90 examples/s]19137 examples [00:18, 1033.90 examples/s]19241 examples [00:18, 1033.30 examples/s]19348 examples [00:18, 1042.23 examples/s]19457 examples [00:18, 1054.24 examples/s]19563 examples [00:18, 1054.78 examples/s]19669 examples [00:18, 1046.92 examples/s]19774 examples [00:18, 1023.10 examples/s]19882 examples [00:18, 1039.37 examples/s]19990 examples [00:19, 1051.11 examples/s]20096 examples [00:19, 1005.58 examples/s]20198 examples [00:19, 1009.25 examples/s]20302 examples [00:19, 1016.23 examples/s]20411 examples [00:19, 1036.31 examples/s]20521 examples [00:19, 1054.17 examples/s]20627 examples [00:19, 1051.00 examples/s]20735 examples [00:19, 1058.73 examples/s]20842 examples [00:19, 1061.76 examples/s]20949 examples [00:19, 1061.06 examples/s]21056 examples [00:20, 1040.07 examples/s]21167 examples [00:20, 1059.69 examples/s]21274 examples [00:20, 990.03 examples/s] 21375 examples [00:20, 957.90 examples/s]21472 examples [00:20, 926.31 examples/s]21566 examples [00:20, 912.41 examples/s]21661 examples [00:20, 922.92 examples/s]21760 examples [00:20, 937.69 examples/s]21855 examples [00:20, 902.71 examples/s]21946 examples [00:21, 888.32 examples/s]22036 examples [00:21, 867.88 examples/s]22132 examples [00:21, 888.01 examples/s]22230 examples [00:21, 913.49 examples/s]22327 examples [00:21, 928.37 examples/s]22427 examples [00:21, 942.57 examples/s]22525 examples [00:21, 950.90 examples/s]22630 examples [00:21, 977.85 examples/s]22735 examples [00:21, 998.09 examples/s]22837 examples [00:21, 1004.36 examples/s]22938 examples [00:22, 999.66 examples/s] 23039 examples [00:22, 989.86 examples/s]23145 examples [00:22, 1009.01 examples/s]23253 examples [00:22, 1028.26 examples/s]23359 examples [00:22, 1036.48 examples/s]23473 examples [00:22, 1063.53 examples/s]23584 examples [00:22, 1074.53 examples/s]23698 examples [00:22, 1092.28 examples/s]23816 examples [00:22, 1116.46 examples/s]23932 examples [00:22, 1127.17 examples/s]24045 examples [00:23, 1085.65 examples/s]24155 examples [00:23, 1066.77 examples/s]24268 examples [00:23, 1082.76 examples/s]24380 examples [00:23, 1091.18 examples/s]24490 examples [00:23, 1036.46 examples/s]24596 examples [00:23, 1042.61 examples/s]24701 examples [00:23, 1019.49 examples/s]24804 examples [00:23, 1015.54 examples/s]24906 examples [00:23, 1005.61 examples/s]25007 examples [00:24, 935.63 examples/s] 25117 examples [00:24, 977.61 examples/s]25221 examples [00:24, 994.07 examples/s]25322 examples [00:24, 983.48 examples/s]25430 examples [00:24, 1008.53 examples/s]25540 examples [00:24, 1032.70 examples/s]25648 examples [00:24, 1045.79 examples/s]25756 examples [00:24, 1054.38 examples/s]25862 examples [00:24, 1054.29 examples/s]25968 examples [00:24, 1037.55 examples/s]26076 examples [00:25, 1047.58 examples/s]26187 examples [00:25, 1064.51 examples/s]26298 examples [00:25, 1075.34 examples/s]26406 examples [00:25, 1055.33 examples/s]26512 examples [00:25, 1040.80 examples/s]26621 examples [00:25, 1054.96 examples/s]26731 examples [00:25, 1065.51 examples/s]26838 examples [00:25, 1059.85 examples/s]26952 examples [00:25, 1080.79 examples/s]27061 examples [00:25, 1071.25 examples/s]27169 examples [00:26, 1026.24 examples/s]27277 examples [00:26, 1041.64 examples/s]27382 examples [00:26, 1027.47 examples/s]27493 examples [00:26, 1049.29 examples/s]27599 examples [00:26, 1047.70 examples/s]27705 examples [00:26, 1047.98 examples/s]27812 examples [00:26, 1053.90 examples/s]27922 examples [00:26, 1065.80 examples/s]28029 examples [00:26, 1055.35 examples/s]28135 examples [00:26, 1056.32 examples/s]28241 examples [00:27, 1043.40 examples/s]28346 examples [00:27, 947.92 examples/s] 28443 examples [00:27, 940.29 examples/s]28539 examples [00:27, 904.56 examples/s]28631 examples [00:27, 907.43 examples/s]28735 examples [00:27, 943.34 examples/s]28838 examples [00:27, 966.03 examples/s]28944 examples [00:27, 989.15 examples/s]29044 examples [00:27, 959.66 examples/s]29143 examples [00:28, 968.47 examples/s]29250 examples [00:28, 995.41 examples/s]29355 examples [00:28, 1010.79 examples/s]29457 examples [00:28, 1004.45 examples/s]29563 examples [00:28, 1017.88 examples/s]29666 examples [00:28, 1013.35 examples/s]29774 examples [00:28, 1031.79 examples/s]29878 examples [00:28, 1012.61 examples/s]29981 examples [00:28, 1017.66 examples/s]30083 examples [00:28, 988.46 examples/s] 30191 examples [00:29, 1013.26 examples/s]30293 examples [00:29, 1005.19 examples/s]30400 examples [00:29, 1021.55 examples/s]30503 examples [00:29, 1018.77 examples/s]30609 examples [00:29, 1030.09 examples/s]30713 examples [00:29, 1017.04 examples/s]30824 examples [00:29, 1042.83 examples/s]30929 examples [00:29, 1026.89 examples/s]31035 examples [00:29, 1034.70 examples/s]31139 examples [00:30, 1029.17 examples/s]31247 examples [00:30, 1041.47 examples/s]31356 examples [00:30, 1053.53 examples/s]31463 examples [00:30, 1055.95 examples/s]31569 examples [00:30, 1009.54 examples/s]31677 examples [00:30, 1028.54 examples/s]31781 examples [00:30, 1031.77 examples/s]31891 examples [00:30, 1051.06 examples/s]32003 examples [00:30, 1070.65 examples/s]32115 examples [00:30, 1083.26 examples/s]32224 examples [00:31, 1067.94 examples/s]32332 examples [00:31, 1070.49 examples/s]32440 examples [00:31, 1064.84 examples/s]32547 examples [00:31, 1065.72 examples/s]32654 examples [00:31, 1053.71 examples/s]32761 examples [00:31, 1057.83 examples/s]32872 examples [00:31, 1072.52 examples/s]32984 examples [00:31, 1085.19 examples/s]33094 examples [00:31, 1087.88 examples/s]33203 examples [00:31, 1082.48 examples/s]33312 examples [00:32, 1080.44 examples/s]33421 examples [00:32, 1079.54 examples/s]33529 examples [00:32, 1075.26 examples/s]33637 examples [00:32, 1056.66 examples/s]33743 examples [00:32, 1040.05 examples/s]33848 examples [00:32, 997.07 examples/s] 33949 examples [00:32, 997.84 examples/s]34057 examples [00:32, 1021.08 examples/s]34160 examples [00:32, 991.18 examples/s] 34265 examples [00:32, 1007.58 examples/s]34377 examples [00:33, 1036.53 examples/s]34484 examples [00:33, 1045.23 examples/s]34589 examples [00:33, 1034.45 examples/s]34693 examples [00:33, 1029.37 examples/s]34804 examples [00:33, 1050.59 examples/s]34914 examples [00:33, 1063.57 examples/s]35021 examples [00:33, 1053.01 examples/s]35130 examples [00:33, 1060.98 examples/s]35239 examples [00:33, 1067.73 examples/s]35346 examples [00:34, 1045.40 examples/s]35453 examples [00:34, 1051.45 examples/s]35560 examples [00:34, 1055.60 examples/s]35667 examples [00:34, 1057.49 examples/s]35777 examples [00:34, 1069.25 examples/s]35886 examples [00:34, 1073.92 examples/s]35994 examples [00:34, 1065.26 examples/s]36106 examples [00:34, 1079.06 examples/s]36215 examples [00:34, 1080.59 examples/s]36324 examples [00:34, 1080.18 examples/s]36433 examples [00:35, 1077.22 examples/s]36544 examples [00:35, 1086.48 examples/s]36653 examples [00:35, 1077.90 examples/s]36764 examples [00:35, 1086.44 examples/s]36873 examples [00:35, 1053.86 examples/s]36982 examples [00:35, 1061.70 examples/s]37089 examples [00:35, 1055.68 examples/s]37195 examples [00:35, 1045.91 examples/s]37306 examples [00:35, 1062.02 examples/s]37413 examples [00:35, 1063.17 examples/s]37522 examples [00:36, 1068.98 examples/s]37630 examples [00:36, 1070.20 examples/s]37742 examples [00:36, 1082.51 examples/s]37851 examples [00:36, 1081.09 examples/s]37960 examples [00:36, 1080.70 examples/s]38069 examples [00:36, 1077.71 examples/s]38177 examples [00:36, 1075.78 examples/s]38285 examples [00:36, 1072.42 examples/s]38396 examples [00:36, 1081.12 examples/s]38506 examples [00:36, 1085.14 examples/s]38615 examples [00:37, 1068.43 examples/s]38725 examples [00:37, 1075.71 examples/s]38833 examples [00:37, 1067.91 examples/s]38940 examples [00:37, 1050.91 examples/s]39048 examples [00:37, 1057.12 examples/s]39154 examples [00:37, 1023.67 examples/s]39261 examples [00:37, 1036.85 examples/s]39369 examples [00:37, 1047.10 examples/s]39476 examples [00:37, 1053.36 examples/s]39586 examples [00:37, 1064.93 examples/s]39694 examples [00:38, 1067.12 examples/s]39801 examples [00:38, 1060.06 examples/s]39913 examples [00:38, 1076.01 examples/s]40021 examples [00:38, 1011.81 examples/s]40131 examples [00:38, 1034.96 examples/s]40237 examples [00:38, 1040.59 examples/s]40342 examples [00:38, 1016.25 examples/s]40445 examples [00:38, 1008.93 examples/s]40552 examples [00:38, 1026.46 examples/s]40656 examples [00:39, 1030.40 examples/s]40766 examples [00:39, 1049.77 examples/s]40879 examples [00:39, 1070.25 examples/s]40987 examples [00:39, 1059.70 examples/s]41094 examples [00:39, 1056.07 examples/s]41206 examples [00:39, 1072.95 examples/s]41315 examples [00:39, 1076.94 examples/s]41426 examples [00:39, 1084.68 examples/s]41538 examples [00:39, 1093.24 examples/s]41648 examples [00:39, 1090.84 examples/s]41758 examples [00:40, 1079.86 examples/s]41867 examples [00:40, 1077.34 examples/s]41975 examples [00:40, 1057.24 examples/s]42085 examples [00:40, 1068.42 examples/s]42192 examples [00:40, 1044.08 examples/s]42300 examples [00:40, 1053.11 examples/s]42412 examples [00:40, 1070.11 examples/s]42527 examples [00:40, 1090.72 examples/s]42637 examples [00:40, 1093.13 examples/s]42747 examples [00:40, 1089.22 examples/s]42860 examples [00:41, 1098.44 examples/s]42972 examples [00:41, 1102.27 examples/s]43083 examples [00:41, 1093.42 examples/s]43196 examples [00:41, 1101.58 examples/s]43307 examples [00:41, 1084.78 examples/s]43420 examples [00:41, 1096.07 examples/s]43534 examples [00:41, 1107.89 examples/s]43645 examples [00:41, 1063.76 examples/s]43752 examples [00:41, 1047.90 examples/s]43858 examples [00:41, 1049.61 examples/s]43965 examples [00:42, 1055.27 examples/s]44071 examples [00:42, 1056.09 examples/s]44181 examples [00:42, 1068.11 examples/s]44288 examples [00:42, 1049.80 examples/s]44394 examples [00:42, 1033.59 examples/s]44507 examples [00:42, 1059.78 examples/s]44623 examples [00:42, 1085.33 examples/s]44735 examples [00:42, 1093.70 examples/s]44845 examples [00:42, 1091.35 examples/s]44955 examples [00:43, 1076.08 examples/s]45063 examples [00:43, 1077.01 examples/s]45171 examples [00:43, 1074.34 examples/s]45282 examples [00:43, 1082.64 examples/s]45392 examples [00:43, 1086.80 examples/s]45501 examples [00:43, 1086.72 examples/s]45612 examples [00:43, 1092.81 examples/s]45722 examples [00:43, 1076.62 examples/s]45830 examples [00:43, 1065.17 examples/s]45940 examples [00:43, 1073.45 examples/s]46048 examples [00:44, 1066.71 examples/s]46155 examples [00:44, 1044.56 examples/s]46268 examples [00:44, 1066.31 examples/s]46376 examples [00:44, 1068.96 examples/s]46486 examples [00:44, 1077.31 examples/s]46594 examples [00:44, 1069.58 examples/s]46710 examples [00:44, 1093.10 examples/s]46821 examples [00:44, 1095.76 examples/s]46931 examples [00:44, 1050.39 examples/s]47044 examples [00:44, 1070.22 examples/s]47155 examples [00:45, 1081.84 examples/s]47269 examples [00:45, 1097.68 examples/s]47380 examples [00:45, 1084.74 examples/s]47492 examples [00:45, 1093.56 examples/s]47602 examples [00:45, 1066.88 examples/s]47710 examples [00:45, 1070.73 examples/s]47818 examples [00:45, 1063.59 examples/s]47925 examples [00:45, 1060.68 examples/s]48033 examples [00:45, 1066.03 examples/s]48140 examples [00:45, 1055.45 examples/s]48246 examples [00:46, 1055.16 examples/s]48352 examples [00:46, 1035.85 examples/s]48460 examples [00:46, 1045.88 examples/s]48567 examples [00:46, 1052.05 examples/s]48673 examples [00:46, 1016.56 examples/s]48775 examples [00:46, 1017.34 examples/s]48885 examples [00:46, 1039.92 examples/s]48995 examples [00:46, 1056.32 examples/s]49101 examples [00:46, 1039.48 examples/s]49207 examples [00:47, 1042.78 examples/s]49312 examples [00:47, 1033.57 examples/s]49417 examples [00:47, 1037.06 examples/s]49521 examples [00:47, 1034.05 examples/s]49625 examples [00:47, 1032.25 examples/s]49735 examples [00:47, 1049.92 examples/s]49841 examples [00:47, 1035.60 examples/s]49955 examples [00:47, 1062.33 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7611/50000 [00:00<00:00, 76109.35 examples/s] 42%|████▏     | 21233/50000 [00:00<00:00, 87720.59 examples/s] 70%|██████▉   | 34784/50000 [00:00<00:00, 98097.66 examples/s] 97%|█████████▋| 48456/50000 [00:00<00:00, 107180.54 examples/s]                                                                0 examples [00:00, ? examples/s]85 examples [00:00, 849.19 examples/s]197 examples [00:00, 914.55 examples/s]308 examples [00:00, 965.16 examples/s]414 examples [00:00, 989.76 examples/s]526 examples [00:00, 1025.09 examples/s]636 examples [00:00, 1045.42 examples/s]743 examples [00:00, 1050.59 examples/s]853 examples [00:00, 1063.92 examples/s]963 examples [00:00, 1074.42 examples/s]1077 examples [00:01, 1091.70 examples/s]1185 examples [00:01, 1076.92 examples/s]1297 examples [00:01, 1087.47 examples/s]1405 examples [00:01, 864.50 examples/s] 1516 examples [00:01, 925.72 examples/s]1624 examples [00:01, 965.52 examples/s]1733 examples [00:01, 998.33 examples/s]1842 examples [00:01, 1022.84 examples/s]1950 examples [00:01, 1037.00 examples/s]2062 examples [00:01, 1058.23 examples/s]2170 examples [00:02, 1051.22 examples/s]2281 examples [00:02, 1065.77 examples/s]2390 examples [00:02, 1072.37 examples/s]2498 examples [00:02, 1073.26 examples/s]2609 examples [00:02, 1083.37 examples/s]2721 examples [00:02, 1092.67 examples/s]2831 examples [00:02, 1031.51 examples/s]2936 examples [00:02, 1021.68 examples/s]3044 examples [00:02, 1036.86 examples/s]3153 examples [00:03, 1050.73 examples/s]3262 examples [00:03, 1062.11 examples/s]3369 examples [00:03, 1063.70 examples/s]3477 examples [00:03, 1067.11 examples/s]3584 examples [00:03, 1065.27 examples/s]3693 examples [00:03, 1071.63 examples/s]3801 examples [00:03, 1073.86 examples/s]3910 examples [00:03, 1069.85 examples/s]4018 examples [00:03, 1056.56 examples/s]4124 examples [00:03, 1048.12 examples/s]4229 examples [00:04, 1019.15 examples/s]4337 examples [00:04, 1035.03 examples/s]4441 examples [00:04, 1031.82 examples/s]4551 examples [00:04, 1049.69 examples/s]4660 examples [00:04, 1060.08 examples/s]4770 examples [00:04, 1069.18 examples/s]4882 examples [00:04, 1082.28 examples/s]4991 examples [00:04, 1075.84 examples/s]5100 examples [00:04, 1079.35 examples/s]5209 examples [00:04, 1081.06 examples/s]5320 examples [00:05, 1088.22 examples/s]5429 examples [00:05, 1063.25 examples/s]5536 examples [00:05, 1045.51 examples/s]5643 examples [00:05, 1052.71 examples/s]5751 examples [00:05, 1059.30 examples/s]5862 examples [00:05, 1073.55 examples/s]5970 examples [00:05, 1071.80 examples/s]6078 examples [00:05, 1027.57 examples/s]6190 examples [00:05, 1051.49 examples/s]6302 examples [00:05, 1070.62 examples/s]6420 examples [00:06, 1100.22 examples/s]6531 examples [00:06, 1096.50 examples/s]6645 examples [00:06, 1108.31 examples/s]6759 examples [00:06, 1115.11 examples/s]6871 examples [00:06, 1110.20 examples/s]6984 examples [00:06, 1114.99 examples/s]7097 examples [00:06, 1117.33 examples/s]7209 examples [00:06, 1091.56 examples/s]7319 examples [00:06, 1091.47 examples/s]7429 examples [00:07, 1090.96 examples/s]7539 examples [00:07, 1054.34 examples/s]7650 examples [00:07, 1069.85 examples/s]7760 examples [00:07, 1076.40 examples/s]7870 examples [00:07, 1080.83 examples/s]7982 examples [00:07, 1091.49 examples/s]8092 examples [00:07, 1091.89 examples/s]8204 examples [00:07, 1099.74 examples/s]8315 examples [00:07, 1092.12 examples/s]8425 examples [00:07, 1053.82 examples/s]8531 examples [00:08, 1015.67 examples/s]8634 examples [00:08, 1012.12 examples/s]8745 examples [00:08, 1037.49 examples/s]8855 examples [00:08, 1053.93 examples/s]8967 examples [00:08, 1072.12 examples/s]9080 examples [00:08, 1086.45 examples/s]9196 examples [00:08, 1107.41 examples/s]9308 examples [00:08, 1109.89 examples/s]9420 examples [00:08, 1078.89 examples/s]9529 examples [00:08, 1081.88 examples/s]9638 examples [00:09, 1082.83 examples/s]9749 examples [00:09, 1088.00 examples/s]9858 examples [00:09, 1067.93 examples/s]9969 examples [00:09, 1078.53 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRUK0EK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRUK0EK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc8530a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc8530a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc8530a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdcf11835c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc6e8c26d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc8530a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc8530a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc8530a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc5ccf7d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc5ccf7eb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fdc5d421510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fdc5d421510> 

  function with postional parmater data_info <function split_train_valid at 0x7fdc5d421510> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fdc5d421620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fdc5d421620> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fdc5d421620> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=dde644dfb44324c6d0a35c69b96a0aed56a1d1ef9f809d72bd751781ccf45bd0
  Stored in directory: /tmp/pip-ephem-wheel-cache-xsfbt48t/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:30:48, 9.39kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:05:29, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:43:01, 18.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:54:32, 26.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.58M/862M [00:01<6:13:12, 38.3kB/s].vector_cache/glove.6B.zip:   1%|          | 7.90M/862M [00:01<4:20:01, 54.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<3:01:02, 78.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.1M/862M [00:01<2:06:22, 112kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:01<1:27:58, 159kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<1:01:15, 227kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:02<42:41, 324kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:02<29:44, 461kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<20:52, 655kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:02<14:35, 930kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<10:18, 1.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:03<08:40, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<07:58, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<07:26, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<05:40, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<06:29, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<06:18, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<04:46, 2.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<03:29, 3.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<1:56:19, 114kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<1:22:56, 160kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<58:20, 227kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<43:29, 304kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<33:07, 399kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<23:51, 554kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<16:48, 783kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<1:59:13, 110kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<1:24:45, 155kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:13<59:33, 220kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<44:38, 293kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<33:54, 386kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<24:22, 536kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:15<17:07, 760kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<56:41, 230kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<41:01, 317kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:17<28:56, 449kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<23:14, 557kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<18:53, 685kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:19<13:46, 939kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:19<09:47, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<13:43, 939kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<10:54, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<07:57, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<08:33, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:17, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<05:25, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<06:47, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<07:20, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<05:44, 2.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:10, 3.04MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<55:03, 230kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<39:50, 318kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<28:09, 449kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<22:36, 558kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<18:23, 686kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<13:30, 932kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:27, 1.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:17, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<06:48, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:41, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:54, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:05, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<04:24, 2.81MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:38, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:00, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:55, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:01, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:10, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:37, 2.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:07, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:19, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:11, 2.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:05, 3.96MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<1:32:58, 131kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<1:07:33, 181kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<47:51, 255kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<35:19, 343kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<25:58, 467kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<18:24, 657kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:41, 769kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:25, 898kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:56, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<07:04, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:59, 750kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<12:26, 963kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:59, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:04, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:47, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:39, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:50, 2.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<08:07, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:41, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:14, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:46, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<50:07, 235kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<36:17, 325kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<25:38, 458kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<20:40, 567kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<15:39, 748kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<11:13, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<10:35, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:35, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:17, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<07:08, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:10, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:33, 2.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:00, 2.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:30, 2.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:11, 1.85MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:49, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:30, 3.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:23, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:44, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:39, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:36, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:54, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:53, 2.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:42, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:33, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:49, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:59, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:19, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:00, 2.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:26, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:58, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:40, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:22, 3.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<24:03, 459kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<17:58, 613kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<12:47, 860kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<11:30, 953kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<10:16, 1.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:39, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:29, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<08:57, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:22, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:26, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:20, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:38, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:08, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:41, 2.91MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<18:03, 596kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<13:43, 783kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<09:49, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:22, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<08:44, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:33, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<04:43, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:07, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:29, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:29, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:18, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:33, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:07, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:45, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:33, 2.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:19, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:59, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:50, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:30, 2.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<41:00, 252kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<29:45, 347kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<21:02, 490kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<17:04, 602kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<12:48, 802kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:25, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<06:40, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<43:17, 236kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<31:09, 327kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<22:01, 462kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<15:28, 655kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<51:41, 196kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<38:14, 265kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<27:15, 372kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<19:04, 528kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<3:29:23, 48.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<2:27:30, 68.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<1:43:14, 97.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<1:14:21, 135kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<54:04, 185kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<38:18, 261kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<28:17, 351kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<20:39, 481kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<14:39, 676kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<10:21, 953kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<25:52, 381kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<20:06, 490kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<14:33, 676kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<10:15, 956kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<9:32:59, 17.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<6:41:41, 24.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<4:40:51, 34.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<3:15:54, 49.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<2:56:16, 55.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<2:05:20, 77.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<1:28:02, 110kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<1:01:31, 157kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<48:11, 200kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<34:41, 278kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<24:27, 394kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<19:18, 497kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<14:28, 662kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<10:21, 923kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<09:28, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<07:35, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:30, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<06:06, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:13, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:53, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:56, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:16, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:11, 2.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<02:20, 3.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<36:47, 253kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<26:40, 349kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<18:51, 492kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<15:20, 603kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<12:37, 732kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<09:14, 999kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:34, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<08:51, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<07:08, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:11, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:45, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:54, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:35, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:21, 2.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:40, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:03, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:47, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:31, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:56, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<02:49, 3.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:45:56, 84.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:15:00, 119kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<52:35, 169kB/s]  .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<38:43, 228kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<28:54, 305kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<20:35, 428kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<14:28, 607kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<14:52, 590kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<11:18, 775kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<08:07, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:42, 1.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:14, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:26, 1.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:54, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<07:32, 1.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<06:10, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:32, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:09, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:28, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:18, 2.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:20, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:46, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:44, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:42, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:22:42, 102kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<58:43, 143kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<41:10, 204kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<30:39, 273kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<22:17, 375kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<15:46, 528kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<12:57, 640kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<09:54, 836kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<07:07, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:54, 1.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:40, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:09, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:49, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:02, 1.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:56, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<04:03, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:39, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:45, 2.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:48, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:17, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:24, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:39, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:15, 2.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:29, 3.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:33, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:17, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<02:27, 3.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:35, 2.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:06, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:16, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:31, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:15, 2.37MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:26, 3.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:30, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:14, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:25, 3.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:30, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:06, 2.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:21, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:10, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<02:22, 3.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<02:22, 3.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:53, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:05, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:13, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<20:10, 363kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<14:43, 497kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<10:26, 700kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<07:21, 986kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<45:09, 161kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<33:04, 219kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<23:25, 309kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<16:24, 439kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<15:58, 450kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<11:55, 603kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<08:28, 845kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:34, 940kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:48, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<05:05, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:36, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<15:31, 454kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<11:35, 608kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<08:15, 850kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:23, 944kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:53, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:36, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:37, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:33, 2.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<58:31, 117kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<41:37, 164kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<29:12, 233kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<21:55, 309kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<16:42, 405kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<11:59, 564kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<08:25, 797kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<54:12, 124kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<38:36, 174kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<27:04, 247kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<20:24, 326kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<15:37, 425kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<11:15, 589kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<08:53, 740kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:54, 952kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:57, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:58, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:48, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:41, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:36, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:22, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:08, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:26, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:41, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<01:56, 3.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<05:32, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:32, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:17, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:46, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:17, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:25, 2.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:08, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:49, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:07, 2.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:33, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<01:53, 3.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<01:24, 4.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<23:47, 253kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<17:15, 349kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<12:09, 493kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<09:52, 604kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<08:06, 734kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:58, 995kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<04:12, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<45:32, 129kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<32:26, 181kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<22:45, 257kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<17:12, 338kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<13:10, 442kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<09:29, 611kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<07:31, 765kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<05:51, 981kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:13, 1.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:16, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<04:11, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:13, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:18, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<41:43, 135kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<29:45, 188kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<20:52, 267kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<15:49, 351kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<11:37, 476kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<08:14, 669kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<07:01, 779kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<05:27, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:56, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:01, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:56, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:59, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:10, 2.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:50, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:07, 2.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:42, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:58, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:18, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:40, 3.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:07, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:24, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:29, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:56, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:29, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:52, 2.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:29, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:46, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:09, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:33, 3.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:52, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:55, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:51, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:08, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:12, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:27, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:45, 2.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:03, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:20, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:48, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:27, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:48, 2.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:23, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:08, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:36, 2.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:13, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:28, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:55, 2.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:23, 3.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:59, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:16, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:22, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:42, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:22, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:44, 2.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:15, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:02, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:31, 2.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<02:06, 2.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:54, 2.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:26, 3.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:01, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:17, 1.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:47, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:17, 3.30MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<06:27, 657kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<04:56, 856kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:33, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:26, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:49, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:23, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:05, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:33, 2.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:02, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:50, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:21, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:53, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:07, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:39, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:11, 3.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:25, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:47, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:02, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:19, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:26, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:53, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:21, 2.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<27:49, 135kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<19:50, 189kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<13:52, 269kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<10:29, 352kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<07:42, 479kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<05:26, 672kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:38, 782kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:59, 907kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:56, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:05, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:45, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:17, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:41, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:59, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:07, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:38, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:10, 2.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<14:55, 229kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<10:46, 317kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<07:34, 447kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:01, 556kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:29, 745kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:18, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<02:18, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<15:49, 207kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<11:23, 288kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<07:58, 408kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<06:17, 511kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<05:03, 635kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:39, 873kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<02:34, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:50, 820kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:00, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:09, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:13, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:51, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:22, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:39, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:44, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:58, 3.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:53, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:37, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:12, 2.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:30, 1.90MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:38, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:15, 2.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:54, 3.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<02:23, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:54, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:23, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:59, 2.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<11:42, 233kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<08:27, 322kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<05:55, 456kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<04:43, 564kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:51, 691kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:47, 947kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:58, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:20, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:54, 1.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:23, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:33, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:20, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:59, 2.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:15, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:22, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:04, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:46, 3.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:46, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:28, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:05, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:17, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:08, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:50, 2.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:07, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:58, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:43, 3.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:01, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:09, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:54, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:53, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:40, 3.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:56, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:51, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:38, 3.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:54, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:33<01:02, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:48, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:35, 3.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:14, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:04, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:46, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:58, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:51, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:38, 2.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:46, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:34, 3.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:47, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:32, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:40, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:28, 3.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<1:31:01, 17.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<1:03:37, 24.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<43:50, 35.0kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<30:18, 49.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<21:15, 70.3kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<14:40, 100kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<10:02, 143kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<12:47, 112kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<09:03, 157kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<06:15, 223kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<04:35, 297kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<03:29, 390kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:28, 544kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<01:41, 770kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:16, 571kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:42, 752kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:12, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:06, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:01, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:46, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:31, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<08:37, 134kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<06:07, 188kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<04:12, 267kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:06, 350kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:16, 475kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:34, 669kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:18, 778kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:07, 906kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:49, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:34, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:48, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:39, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:31, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:32, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:24, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 2.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:21, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:23, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:30, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:25, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:20, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:17, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:12, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:16, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:17, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:09, 3.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:21, 1.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:03, 3.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:01, 3.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.13MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<32:42:40,  3.40it/s]  0%|          | 876/400000 [00:00<22:51:05,  4.85it/s]  0%|          | 1767/400000 [00:00<15:57:50,  6.93it/s]  1%|          | 2641/400000 [00:00<11:09:14,  9.90it/s]  1%|          | 3486/400000 [00:00<7:47:42, 14.13it/s]   1%|          | 4247/400000 [00:00<5:27:01, 20.17it/s]  1%|▏         | 5139/400000 [00:00<3:48:37, 28.78it/s]  2%|▏         | 6037/400000 [00:00<2:39:53, 41.06it/s]  2%|▏         | 6921/400000 [00:01<1:51:53, 58.55it/s]  2%|▏         | 7777/400000 [00:01<1:18:23, 83.39it/s]  2%|▏         | 8671/400000 [00:01<54:57, 118.66it/s]   2%|▏         | 9559/400000 [00:01<38:36, 168.55it/s]  3%|▎         | 10459/400000 [00:01<27:10, 238.87it/s]  3%|▎         | 11332/400000 [00:01<19:12, 337.28it/s]  3%|▎         | 12207/400000 [00:01<13:38, 473.99it/s]  3%|▎         | 13106/400000 [00:01<09:44, 662.17it/s]  3%|▎         | 13991/400000 [00:01<07:01, 916.56it/s]  4%|▎         | 14874/400000 [00:01<05:07, 1253.51it/s]  4%|▍         | 15756/400000 [00:02<03:47, 1685.66it/s]  4%|▍         | 16632/400000 [00:02<02:52, 2224.42it/s]  4%|▍         | 17508/400000 [00:02<02:14, 2848.60it/s]  5%|▍         | 18374/400000 [00:02<01:47, 3566.39it/s]  5%|▍         | 19272/400000 [00:02<01:27, 4353.73it/s]  5%|▌         | 20145/400000 [00:02<01:15, 5055.64it/s]  5%|▌         | 21006/400000 [00:02<01:05, 5769.78it/s]  5%|▌         | 21907/400000 [00:02<00:58, 6466.56it/s]  6%|▌         | 22799/400000 [00:02<00:53, 7047.18it/s]  6%|▌         | 23699/400000 [00:03<00:49, 7537.66it/s]  6%|▌         | 24583/400000 [00:03<00:48, 7806.43it/s]  6%|▋         | 25481/400000 [00:03<00:46, 8121.39it/s]  7%|▋         | 26362/400000 [00:03<00:45, 8230.43it/s]  7%|▋         | 27240/400000 [00:03<00:44, 8386.46it/s]  7%|▋         | 28121/400000 [00:03<00:43, 8507.70it/s]  7%|▋         | 28997/400000 [00:03<00:43, 8573.11it/s]  7%|▋         | 29872/400000 [00:03<00:43, 8594.89it/s]  8%|▊         | 30779/400000 [00:03<00:42, 8731.90it/s]  8%|▊         | 31676/400000 [00:03<00:41, 8800.60it/s]  8%|▊         | 32565/400000 [00:04<00:41, 8825.59it/s]  8%|▊         | 33453/400000 [00:04<00:41, 8814.88it/s]  9%|▊         | 34338/400000 [00:04<00:41, 8782.27it/s]  9%|▉         | 35234/400000 [00:04<00:41, 8833.81it/s]  9%|▉         | 36131/400000 [00:04<00:41, 8871.75it/s]  9%|▉         | 37020/400000 [00:04<00:40, 8872.82it/s]  9%|▉         | 37909/400000 [00:04<00:41, 8812.69it/s] 10%|▉         | 38811/400000 [00:04<00:40, 8873.18it/s] 10%|▉         | 39699/400000 [00:04<00:40, 8850.42it/s] 10%|█         | 40585/400000 [00:04<00:40, 8820.49it/s] 10%|█         | 41468/400000 [00:05<00:41, 8708.62it/s] 11%|█         | 42340/400000 [00:05<00:41, 8614.00it/s] 11%|█         | 43227/400000 [00:05<00:41, 8688.50it/s] 11%|█         | 44127/400000 [00:05<00:40, 8776.72it/s] 11%|█▏        | 45032/400000 [00:05<00:40, 8854.85it/s] 11%|█▏        | 45929/400000 [00:05<00:39, 8888.21it/s] 12%|█▏        | 46819/400000 [00:05<00:40, 8788.52it/s] 12%|█▏        | 47712/400000 [00:05<00:39, 8829.18it/s] 12%|█▏        | 48596/400000 [00:05<00:39, 8829.75it/s] 12%|█▏        | 49480/400000 [00:05<00:39, 8829.04it/s] 13%|█▎        | 50375/400000 [00:06<00:39, 8864.77it/s] 13%|█▎        | 51262/400000 [00:06<00:39, 8842.50it/s] 13%|█▎        | 52153/400000 [00:06<00:39, 8862.62it/s] 13%|█▎        | 53058/400000 [00:06<00:38, 8917.63it/s] 13%|█▎        | 53978/400000 [00:06<00:38, 8998.73it/s] 14%|█▎        | 54879/400000 [00:06<00:38, 8992.28it/s] 14%|█▍        | 55779/400000 [00:06<00:38, 8951.45it/s] 14%|█▍        | 56675/400000 [00:06<00:38, 8934.79it/s] 14%|█▍        | 57569/400000 [00:06<00:38, 8889.57it/s] 15%|█▍        | 58461/400000 [00:06<00:38, 8896.19it/s] 15%|█▍        | 59351/400000 [00:07<00:39, 8704.95it/s] 15%|█▌        | 60223/400000 [00:07<00:39, 8693.10it/s] 15%|█▌        | 61097/400000 [00:07<00:38, 8706.27it/s] 15%|█▌        | 61969/400000 [00:07<00:39, 8643.74it/s] 16%|█▌        | 62898/400000 [00:07<00:38, 8826.01it/s] 16%|█▌        | 63801/400000 [00:07<00:37, 8885.76it/s] 16%|█▌        | 64691/400000 [00:07<00:38, 8792.34it/s] 16%|█▋        | 65610/400000 [00:07<00:37, 8907.79it/s] 17%|█▋        | 66523/400000 [00:07<00:37, 8972.75it/s] 17%|█▋        | 67430/400000 [00:07<00:36, 9000.96it/s] 17%|█▋        | 68345/400000 [00:08<00:36, 9042.52it/s] 17%|█▋        | 69257/400000 [00:08<00:36, 9064.33it/s] 18%|█▊        | 70164/400000 [00:08<00:36, 8970.93it/s] 18%|█▊        | 71077/400000 [00:08<00:36, 9015.92it/s] 18%|█▊        | 72014/400000 [00:08<00:35, 9117.31it/s] 18%|█▊        | 72966/400000 [00:08<00:35, 9232.31it/s] 18%|█▊        | 73897/400000 [00:08<00:35, 9254.47it/s] 19%|█▊        | 74823/400000 [00:08<00:35, 9229.49it/s] 19%|█▉        | 75747/400000 [00:08<00:35, 9143.87it/s] 19%|█▉        | 76662/400000 [00:08<00:35, 9035.30it/s] 19%|█▉        | 77571/400000 [00:09<00:35, 9051.13it/s] 20%|█▉        | 78477/400000 [00:09<00:35, 8999.26it/s] 20%|█▉        | 79378/400000 [00:09<00:35, 8933.77it/s] 20%|██        | 80272/400000 [00:09<00:36, 8783.16it/s] 20%|██        | 81219/400000 [00:09<00:35, 8978.36it/s] 21%|██        | 82159/400000 [00:09<00:34, 9098.97it/s] 21%|██        | 83071/400000 [00:09<00:34, 9060.38it/s] 21%|██        | 84002/400000 [00:09<00:34, 9131.89it/s] 21%|██        | 84917/400000 [00:09<00:34, 9112.13it/s] 21%|██▏       | 85829/400000 [00:09<00:35, 8930.04it/s] 22%|██▏       | 86765/400000 [00:10<00:34, 9054.77it/s] 22%|██▏       | 87679/400000 [00:10<00:34, 9079.10it/s] 22%|██▏       | 88588/400000 [00:10<00:34, 8966.86it/s] 22%|██▏       | 89486/400000 [00:10<00:34, 8921.66it/s] 23%|██▎       | 90387/400000 [00:10<00:34, 8947.06it/s] 23%|██▎       | 91316/400000 [00:10<00:34, 9044.99it/s] 23%|██▎       | 92224/400000 [00:10<00:33, 9054.67it/s] 23%|██▎       | 93130/400000 [00:10<00:34, 9023.85it/s] 24%|██▎       | 94048/400000 [00:10<00:33, 9069.52it/s] 24%|██▎       | 94971/400000 [00:11<00:33, 9115.09it/s] 24%|██▍       | 95900/400000 [00:11<00:33, 9164.40it/s] 24%|██▍       | 96817/400000 [00:11<00:33, 9143.06it/s] 24%|██▍       | 97736/400000 [00:11<00:33, 9155.17it/s] 25%|██▍       | 98652/400000 [00:11<00:32, 9143.75it/s] 25%|██▍       | 99573/400000 [00:11<00:32, 9160.59it/s] 25%|██▌       | 100500/400000 [00:11<00:32, 9192.73it/s] 25%|██▌       | 101420/400000 [00:11<00:32, 9149.53it/s] 26%|██▌       | 102336/400000 [00:11<00:33, 8940.55it/s] 26%|██▌       | 103232/400000 [00:11<00:33, 8863.49it/s] 26%|██▌       | 104120/400000 [00:12<00:33, 8787.21it/s] 26%|██▋       | 105000/400000 [00:12<00:33, 8784.46it/s] 26%|██▋       | 105880/400000 [00:12<00:33, 8735.00it/s] 27%|██▋       | 106754/400000 [00:12<00:33, 8731.92it/s] 27%|██▋       | 107641/400000 [00:12<00:33, 8772.71it/s] 27%|██▋       | 108538/400000 [00:12<00:33, 8828.95it/s] 27%|██▋       | 109477/400000 [00:12<00:32, 8988.12it/s] 28%|██▊       | 110377/400000 [00:12<00:32, 8955.01it/s] 28%|██▊       | 111274/400000 [00:12<00:32, 8806.52it/s] 28%|██▊       | 112156/400000 [00:12<00:33, 8575.90it/s] 28%|██▊       | 113023/400000 [00:13<00:33, 8602.26it/s] 28%|██▊       | 113922/400000 [00:13<00:32, 8705.76it/s] 29%|██▊       | 114794/400000 [00:13<00:32, 8661.09it/s] 29%|██▉       | 115662/400000 [00:13<00:33, 8551.69it/s] 29%|██▉       | 116523/400000 [00:13<00:33, 8567.01it/s] 29%|██▉       | 117388/400000 [00:13<00:32, 8590.07it/s] 30%|██▉       | 118270/400000 [00:13<00:32, 8655.17it/s] 30%|██▉       | 119147/400000 [00:13<00:32, 8688.59it/s] 30%|███       | 120044/400000 [00:13<00:31, 8770.53it/s] 30%|███       | 120931/400000 [00:13<00:31, 8798.86it/s] 30%|███       | 121812/400000 [00:14<00:31, 8782.73it/s] 31%|███       | 122702/400000 [00:14<00:31, 8816.62it/s] 31%|███       | 123605/400000 [00:14<00:31, 8879.22it/s] 31%|███       | 124494/400000 [00:14<00:31, 8808.08it/s] 31%|███▏      | 125416/400000 [00:14<00:30, 8926.30it/s] 32%|███▏      | 126310/400000 [00:14<00:31, 8796.48it/s] 32%|███▏      | 127225/400000 [00:14<00:30, 8895.04it/s] 32%|███▏      | 128116/400000 [00:14<00:30, 8771.94it/s] 32%|███▏      | 128995/400000 [00:14<00:31, 8503.11it/s] 32%|███▏      | 129882/400000 [00:14<00:31, 8608.71it/s] 33%|███▎      | 130770/400000 [00:15<00:30, 8687.43it/s] 33%|███▎      | 131666/400000 [00:15<00:30, 8765.09it/s] 33%|███▎      | 132551/400000 [00:15<00:30, 8787.38it/s] 33%|███▎      | 133431/400000 [00:15<00:30, 8781.97it/s] 34%|███▎      | 134321/400000 [00:15<00:30, 8816.87it/s] 34%|███▍      | 135216/400000 [00:15<00:29, 8855.45it/s] 34%|███▍      | 136120/400000 [00:15<00:29, 8907.66it/s] 34%|███▍      | 137012/400000 [00:15<00:29, 8903.75it/s] 34%|███▍      | 137903/400000 [00:15<00:29, 8826.94it/s] 35%|███▍      | 138787/400000 [00:15<00:29, 8718.41it/s] 35%|███▍      | 139727/400000 [00:16<00:29, 8910.44it/s] 35%|███▌      | 140641/400000 [00:16<00:28, 8977.74it/s] 35%|███▌      | 141540/400000 [00:16<00:28, 8964.57it/s] 36%|███▌      | 142468/400000 [00:16<00:28, 9055.95it/s] 36%|███▌      | 143375/400000 [00:16<00:28, 9038.67it/s] 36%|███▌      | 144288/400000 [00:16<00:28, 9064.81it/s] 36%|███▋      | 145195/400000 [00:16<00:28, 9025.69it/s] 37%|███▋      | 146098/400000 [00:16<00:28, 8837.84it/s] 37%|███▋      | 146989/400000 [00:16<00:28, 8859.10it/s] 37%|███▋      | 147888/400000 [00:16<00:28, 8896.36it/s] 37%|███▋      | 148809/400000 [00:17<00:27, 8987.60it/s] 37%|███▋      | 149709/400000 [00:17<00:28, 8888.54it/s] 38%|███▊      | 150599/400000 [00:17<00:28, 8685.79it/s] 38%|███▊      | 151470/400000 [00:17<00:28, 8684.17it/s] 38%|███▊      | 152362/400000 [00:17<00:28, 8752.78it/s] 38%|███▊      | 153239/400000 [00:17<00:28, 8710.71it/s] 39%|███▊      | 154131/400000 [00:17<00:28, 8770.96it/s] 39%|███▉      | 155018/400000 [00:17<00:27, 8799.07it/s] 39%|███▉      | 155925/400000 [00:17<00:27, 8877.70it/s] 39%|███▉      | 156814/400000 [00:18<00:27, 8854.62it/s] 39%|███▉      | 157702/400000 [00:18<00:27, 8861.67it/s] 40%|███▉      | 158589/400000 [00:18<00:27, 8808.35it/s] 40%|███▉      | 159476/400000 [00:18<00:27, 8823.95it/s] 40%|████      | 160359/400000 [00:18<00:27, 8723.07it/s] 40%|████      | 161256/400000 [00:18<00:27, 8794.30it/s] 41%|████      | 162149/400000 [00:18<00:26, 8832.69it/s] 41%|████      | 163033/400000 [00:18<00:26, 8808.45it/s] 41%|████      | 163915/400000 [00:18<00:26, 8769.37it/s] 41%|████      | 164801/400000 [00:18<00:26, 8794.96it/s] 41%|████▏     | 165681/400000 [00:19<00:26, 8792.07it/s] 42%|████▏     | 166577/400000 [00:19<00:26, 8841.23it/s] 42%|████▏     | 167462/400000 [00:19<00:26, 8801.81it/s] 42%|████▏     | 168360/400000 [00:19<00:26, 8852.58it/s] 42%|████▏     | 169261/400000 [00:19<00:25, 8896.84it/s] 43%|████▎     | 170168/400000 [00:19<00:25, 8947.77it/s] 43%|████▎     | 171110/400000 [00:19<00:25, 9081.77it/s] 43%|████▎     | 172019/400000 [00:19<00:25, 9047.30it/s] 43%|████▎     | 172925/400000 [00:19<00:25, 9033.46it/s] 43%|████▎     | 173829/400000 [00:19<00:25, 9023.60it/s] 44%|████▎     | 174732/400000 [00:20<00:25, 9009.11it/s] 44%|████▍     | 175640/400000 [00:20<00:24, 9027.58it/s] 44%|████▍     | 176553/400000 [00:20<00:24, 9055.68it/s] 44%|████▍     | 177478/400000 [00:20<00:24, 9112.00it/s] 45%|████▍     | 178390/400000 [00:20<00:24, 9092.82it/s] 45%|████▍     | 179303/400000 [00:20<00:24, 9101.40it/s] 45%|████▌     | 180214/400000 [00:20<00:24, 9062.42it/s] 45%|████▌     | 181121/400000 [00:20<00:24, 8995.45it/s] 46%|████▌     | 182079/400000 [00:20<00:23, 9160.87it/s] 46%|████▌     | 182996/400000 [00:20<00:23, 9042.29it/s] 46%|████▌     | 183912/400000 [00:21<00:23, 9077.07it/s] 46%|████▌     | 184821/400000 [00:21<00:23, 9003.54it/s] 46%|████▋     | 185737/400000 [00:21<00:23, 9047.44it/s] 47%|████▋     | 186706/400000 [00:21<00:23, 9230.40it/s] 47%|████▋     | 187631/400000 [00:21<00:23, 9130.96it/s] 47%|████▋     | 188546/400000 [00:21<00:23, 8984.56it/s] 47%|████▋     | 189446/400000 [00:21<00:23, 8943.87it/s] 48%|████▊     | 190365/400000 [00:21<00:23, 9014.61it/s] 48%|████▊     | 191293/400000 [00:21<00:22, 9091.94it/s] 48%|████▊     | 192203/400000 [00:21<00:23, 8999.95it/s] 48%|████▊     | 193144/400000 [00:22<00:22, 9117.11it/s] 49%|████▊     | 194092/400000 [00:22<00:22, 9221.02it/s] 49%|████▉     | 195015/400000 [00:22<00:22, 9160.79it/s] 49%|████▉     | 195932/400000 [00:22<00:22, 9063.54it/s] 49%|████▉     | 196840/400000 [00:22<00:22, 8958.42it/s] 49%|████▉     | 197737/400000 [00:22<00:22, 8906.04it/s] 50%|████▉     | 198665/400000 [00:22<00:22, 9012.93it/s] 50%|████▉     | 199568/400000 [00:22<00:22, 8996.01it/s] 50%|█████     | 200496/400000 [00:22<00:21, 9079.34it/s] 50%|█████     | 201405/400000 [00:22<00:22, 9002.65it/s] 51%|█████     | 202329/400000 [00:23<00:21, 9071.20it/s] 51%|█████     | 203251/400000 [00:23<00:21, 9115.33it/s] 51%|█████     | 204163/400000 [00:23<00:21, 9046.90it/s] 51%|█████▏    | 205069/400000 [00:23<00:21, 9003.32it/s] 51%|█████▏    | 205970/400000 [00:23<00:21, 8838.57it/s] 52%|█████▏    | 206881/400000 [00:23<00:21, 8917.07it/s] 52%|█████▏    | 207834/400000 [00:23<00:21, 9092.28it/s] 52%|█████▏    | 208746/400000 [00:23<00:21, 9100.45it/s] 52%|█████▏    | 209664/400000 [00:23<00:20, 9122.17it/s] 53%|█████▎    | 210577/400000 [00:23<00:21, 9009.65it/s] 53%|█████▎    | 211502/400000 [00:24<00:20, 9079.27it/s] 53%|█████▎    | 212454/400000 [00:24<00:20, 9204.28it/s] 53%|█████▎    | 213376/400000 [00:24<00:20, 9119.78it/s] 54%|█████▎    | 214289/400000 [00:24<00:20, 9061.55it/s] 54%|█████▍    | 215196/400000 [00:24<00:20, 8995.00it/s] 54%|█████▍    | 216137/400000 [00:24<00:20, 9114.01it/s] 54%|█████▍    | 217050/400000 [00:24<00:20, 9100.99it/s] 54%|█████▍    | 217961/400000 [00:24<00:20, 9033.42it/s] 55%|█████▍    | 218865/400000 [00:24<00:20, 8992.87it/s] 55%|█████▍    | 219765/400000 [00:24<00:20, 8828.93it/s] 55%|█████▌    | 220678/400000 [00:25<00:20, 8915.42it/s] 55%|█████▌    | 221596/400000 [00:25<00:19, 8990.72it/s] 56%|█████▌    | 222496/400000 [00:25<00:19, 8926.91it/s] 56%|█████▌    | 223390/400000 [00:25<00:19, 8871.65it/s] 56%|█████▌    | 224278/400000 [00:25<00:19, 8846.60it/s] 56%|█████▋    | 225174/400000 [00:25<00:19, 8878.25it/s] 57%|█████▋    | 226077/400000 [00:25<00:19, 8920.39it/s] 57%|█████▋    | 226973/400000 [00:25<00:19, 8931.18it/s] 57%|█████▋    | 227867/400000 [00:25<00:19, 8928.27it/s] 57%|█████▋    | 228760/400000 [00:26<00:19, 8827.07it/s] 57%|█████▋    | 229644/400000 [00:26<00:19, 8785.75it/s] 58%|█████▊    | 230543/400000 [00:26<00:19, 8841.76it/s] 58%|█████▊    | 231428/400000 [00:26<00:19, 8836.00it/s] 58%|█████▊    | 232338/400000 [00:26<00:18, 8911.69it/s] 58%|█████▊    | 233267/400000 [00:26<00:18, 9020.36it/s] 59%|█████▊    | 234171/400000 [00:26<00:18, 9023.74it/s] 59%|█████▉    | 235074/400000 [00:26<00:18, 8903.04it/s] 59%|█████▉    | 235965/400000 [00:26<00:18, 8677.40it/s] 59%|█████▉    | 236835/400000 [00:26<00:18, 8658.73it/s] 59%|█████▉    | 237748/400000 [00:27<00:18, 8793.66it/s] 60%|█████▉    | 238691/400000 [00:27<00:17, 8974.39it/s] 60%|█████▉    | 239591/400000 [00:27<00:17, 8973.64it/s] 60%|██████    | 240525/400000 [00:27<00:17, 9079.13it/s] 60%|██████    | 241475/400000 [00:27<00:17, 9201.36it/s] 61%|██████    | 242397/400000 [00:27<00:17, 9069.81it/s] 61%|██████    | 243307/400000 [00:27<00:17, 9077.22it/s] 61%|██████    | 244216/400000 [00:27<00:17, 8964.77it/s] 61%|██████▏   | 245114/400000 [00:27<00:17, 8952.39it/s] 62%|██████▏   | 246010/400000 [00:27<00:17, 8937.90it/s] 62%|██████▏   | 246913/400000 [00:28<00:17, 8964.59it/s] 62%|██████▏   | 247820/400000 [00:28<00:16, 8994.92it/s] 62%|██████▏   | 248720/400000 [00:28<00:16, 8996.04it/s] 62%|██████▏   | 249620/400000 [00:28<00:16, 8983.08it/s] 63%|██████▎   | 250519/400000 [00:28<00:16, 8828.04it/s] 63%|██████▎   | 251403/400000 [00:28<00:16, 8811.26it/s] 63%|██████▎   | 252285/400000 [00:28<00:17, 8584.74it/s] 63%|██████▎   | 253156/400000 [00:28<00:17, 8619.67it/s] 64%|██████▎   | 254046/400000 [00:28<00:16, 8700.19it/s] 64%|██████▎   | 254958/400000 [00:28<00:16, 8819.97it/s] 64%|██████▍   | 255842/400000 [00:29<00:16, 8797.60it/s] 64%|██████▍   | 256737/400000 [00:29<00:16, 8841.74it/s] 64%|██████▍   | 257622/400000 [00:29<00:16, 8807.96it/s] 65%|██████▍   | 258543/400000 [00:29<00:15, 8923.77it/s] 65%|██████▍   | 259437/400000 [00:29<00:15, 8889.45it/s] 65%|██████▌   | 260351/400000 [00:29<00:15, 8962.81it/s] 65%|██████▌   | 261248/400000 [00:29<00:15, 8925.69it/s] 66%|██████▌   | 262141/400000 [00:29<00:15, 8830.52it/s] 66%|██████▌   | 263027/400000 [00:29<00:15, 8838.50it/s] 66%|██████▌   | 263918/400000 [00:29<00:15, 8857.82it/s] 66%|██████▌   | 264805/400000 [00:30<00:15, 8729.31it/s] 66%|██████▋   | 265679/400000 [00:30<00:15, 8698.50it/s] 67%|██████▋   | 266553/400000 [00:30<00:15, 8710.70it/s] 67%|██████▋   | 267446/400000 [00:30<00:15, 8774.92it/s] 67%|██████▋   | 268347/400000 [00:30<00:14, 8842.79it/s] 67%|██████▋   | 269232/400000 [00:30<00:14, 8829.14it/s] 68%|██████▊   | 270116/400000 [00:30<00:14, 8826.22it/s] 68%|██████▊   | 271026/400000 [00:30<00:14, 8904.43it/s] 68%|██████▊   | 271945/400000 [00:30<00:14, 8987.89it/s] 68%|██████▊   | 272856/400000 [00:30<00:14, 9022.96it/s] 68%|██████▊   | 273759/400000 [00:31<00:14, 8865.70it/s] 69%|██████▊   | 274648/400000 [00:31<00:14, 8871.83it/s] 69%|██████▉   | 275551/400000 [00:31<00:13, 8916.19it/s] 69%|██████▉   | 276469/400000 [00:31<00:13, 8992.07it/s] 69%|██████▉   | 277369/400000 [00:31<00:13, 8927.68it/s] 70%|██████▉   | 278263/400000 [00:31<00:13, 8863.36it/s] 70%|██████▉   | 279179/400000 [00:31<00:13, 8948.61it/s] 70%|███████   | 280075/400000 [00:31<00:13, 8823.55it/s] 70%|███████   | 280983/400000 [00:31<00:13, 8896.95it/s] 70%|███████   | 281879/400000 [00:31<00:13, 8914.28it/s] 71%|███████   | 282771/400000 [00:32<00:13, 8900.53it/s] 71%|███████   | 283679/400000 [00:32<00:12, 8951.25it/s] 71%|███████   | 284575/400000 [00:32<00:12, 8932.99it/s] 71%|███████▏  | 285469/400000 [00:32<00:12, 8885.68it/s] 72%|███████▏  | 286358/400000 [00:32<00:12, 8871.25it/s] 72%|███████▏  | 287246/400000 [00:32<00:12, 8806.91it/s] 72%|███████▏  | 288137/400000 [00:32<00:12, 8835.85it/s] 72%|███████▏  | 289021/400000 [00:32<00:12, 8747.46it/s] 72%|███████▏  | 289897/400000 [00:32<00:12, 8649.89it/s] 73%|███████▎  | 290763/400000 [00:32<00:12, 8633.54it/s] 73%|███████▎  | 291629/400000 [00:33<00:12, 8639.16it/s] 73%|███████▎  | 292546/400000 [00:33<00:12, 8790.67it/s] 73%|███████▎  | 293426/400000 [00:33<00:12, 8752.14it/s] 74%|███████▎  | 294310/400000 [00:33<00:12, 8777.61it/s] 74%|███████▍  | 295189/400000 [00:33<00:11, 8757.64it/s] 74%|███████▍  | 296066/400000 [00:33<00:11, 8715.21it/s] 74%|███████▍  | 296962/400000 [00:33<00:11, 8785.54it/s] 74%|███████▍  | 297841/400000 [00:33<00:11, 8744.69it/s] 75%|███████▍  | 298757/400000 [00:33<00:11, 8864.36it/s] 75%|███████▍  | 299645/400000 [00:34<00:11, 8756.29it/s] 75%|███████▌  | 300538/400000 [00:34<00:11, 8805.11it/s] 75%|███████▌  | 301448/400000 [00:34<00:11, 8888.70it/s] 76%|███████▌  | 302338/400000 [00:34<00:11, 8818.23it/s] 76%|███████▌  | 303221/400000 [00:34<00:11, 8785.12it/s] 76%|███████▌  | 304109/400000 [00:34<00:10, 8812.35it/s] 76%|███████▋  | 305027/400000 [00:34<00:10, 8918.53it/s] 76%|███████▋  | 305941/400000 [00:34<00:10, 8982.33it/s] 77%|███████▋  | 306840/400000 [00:34<00:10, 8884.77it/s] 77%|███████▋  | 307731/400000 [00:34<00:10, 8892.14it/s] 77%|███████▋  | 308621/400000 [00:35<00:10, 8855.93it/s] 77%|███████▋  | 309507/400000 [00:35<00:10, 8805.60it/s] 78%|███████▊  | 310393/400000 [00:35<00:10, 8820.77it/s] 78%|███████▊  | 311276/400000 [00:35<00:10, 8807.58it/s] 78%|███████▊  | 312157/400000 [00:35<00:10, 8769.30it/s] 78%|███████▊  | 313037/400000 [00:35<00:09, 8778.12it/s] 78%|███████▊  | 313915/400000 [00:35<00:09, 8722.57it/s] 79%|███████▊  | 314830/400000 [00:35<00:09, 8845.96it/s] 79%|███████▉  | 315716/400000 [00:35<00:09, 8833.59it/s] 79%|███████▉  | 316644/400000 [00:35<00:09, 8961.03it/s] 79%|███████▉  | 317541/400000 [00:36<00:09, 8942.90it/s] 80%|███████▉  | 318436/400000 [00:36<00:09, 8926.69it/s] 80%|███████▉  | 319330/400000 [00:36<00:09, 8838.11it/s] 80%|████████  | 320249/400000 [00:36<00:08, 8939.44it/s] 80%|████████  | 321144/400000 [00:36<00:08, 8889.63it/s] 81%|████████  | 322034/400000 [00:36<00:08, 8827.24it/s] 81%|████████  | 322926/400000 [00:36<00:08, 8853.88it/s] 81%|████████  | 323812/400000 [00:36<00:08, 8802.33it/s] 81%|████████  | 324693/400000 [00:36<00:08, 8776.07it/s] 81%|████████▏ | 325571/400000 [00:36<00:08, 8745.36it/s] 82%|████████▏ | 326454/400000 [00:37<00:08, 8768.04it/s] 82%|████████▏ | 327350/400000 [00:37<00:08, 8823.59it/s] 82%|████████▏ | 328233/400000 [00:37<00:08, 8761.16it/s] 82%|████████▏ | 329121/400000 [00:37<00:08, 8795.97it/s] 83%|████████▎ | 330016/400000 [00:37<00:07, 8840.05it/s] 83%|████████▎ | 330953/400000 [00:37<00:07, 8989.90it/s] 83%|████████▎ | 331890/400000 [00:37<00:07, 9099.82it/s] 83%|████████▎ | 332801/400000 [00:37<00:07, 9096.12it/s] 83%|████████▎ | 333712/400000 [00:37<00:07, 9023.63it/s] 84%|████████▎ | 334615/400000 [00:37<00:07, 9006.15it/s] 84%|████████▍ | 335521/400000 [00:38<00:07, 9022.07it/s] 84%|████████▍ | 336424/400000 [00:38<00:07, 8994.27it/s] 84%|████████▍ | 337324/400000 [00:38<00:07, 8951.25it/s] 85%|████████▍ | 338220/400000 [00:38<00:06, 8872.70it/s] 85%|████████▍ | 339108/400000 [00:38<00:06, 8857.24it/s] 85%|████████▌ | 340008/400000 [00:38<00:06, 8899.30it/s] 85%|████████▌ | 340899/400000 [00:38<00:06, 8880.56it/s] 85%|████████▌ | 341793/400000 [00:38<00:06, 8896.95it/s] 86%|████████▌ | 342683/400000 [00:38<00:06, 8694.33it/s] 86%|████████▌ | 343554/400000 [00:38<00:06, 8676.69it/s] 86%|████████▌ | 344423/400000 [00:39<00:06, 8584.90it/s] 86%|████████▋ | 345320/400000 [00:39<00:06, 8695.10it/s] 87%|████████▋ | 346201/400000 [00:39<00:06, 8727.67it/s] 87%|████████▋ | 347136/400000 [00:39<00:05, 8904.17it/s] 87%|████████▋ | 348064/400000 [00:39<00:05, 9013.25it/s] 87%|████████▋ | 348967/400000 [00:39<00:05, 8966.24it/s] 87%|████████▋ | 349878/400000 [00:39<00:05, 9006.18it/s] 88%|████████▊ | 350780/400000 [00:39<00:05, 8908.14it/s] 88%|████████▊ | 351713/400000 [00:39<00:05, 9029.67it/s] 88%|████████▊ | 352630/400000 [00:39<00:05, 9065.71it/s] 88%|████████▊ | 353538/400000 [00:40<00:05, 8973.23it/s] 89%|████████▊ | 354474/400000 [00:40<00:05, 9084.41it/s] 89%|████████▉ | 355391/400000 [00:40<00:04, 9108.07it/s] 89%|████████▉ | 356341/400000 [00:40<00:04, 9221.36it/s] 89%|████████▉ | 357273/400000 [00:40<00:04, 9247.65it/s] 90%|████████▉ | 358211/400000 [00:40<00:04, 9286.01it/s] 90%|████████▉ | 359141/400000 [00:40<00:04, 9159.57it/s] 90%|█████████ | 360084/400000 [00:40<00:04, 9237.74it/s] 90%|█████████ | 361021/400000 [00:40<00:04, 9275.43it/s] 90%|█████████ | 361950/400000 [00:40<00:04, 9232.48it/s] 91%|█████████ | 362874/400000 [00:41<00:04, 9202.13it/s] 91%|█████████ | 363795/400000 [00:41<00:03, 9068.29it/s] 91%|█████████ | 364703/400000 [00:41<00:03, 9057.21it/s] 91%|█████████▏| 365610/400000 [00:41<00:03, 8870.81it/s] 92%|█████████▏| 366523/400000 [00:41<00:03, 8944.64it/s] 92%|█████████▏| 367438/400000 [00:41<00:03, 9003.65it/s] 92%|█████████▏| 368340/400000 [00:41<00:03, 8499.46it/s] 92%|█████████▏| 369198/400000 [00:41<00:03, 8523.30it/s] 93%|█████████▎| 370055/400000 [00:41<00:03, 8489.67it/s] 93%|█████████▎| 370928/400000 [00:42<00:03, 8559.83it/s] 93%|█████████▎| 371830/400000 [00:42<00:03, 8691.17it/s] 93%|█████████▎| 372702/400000 [00:42<00:03, 8682.50it/s] 93%|█████████▎| 373593/400000 [00:42<00:03, 8747.36it/s] 94%|█████████▎| 374506/400000 [00:42<00:02, 8856.48it/s] 94%|█████████▍| 375428/400000 [00:42<00:02, 8962.28it/s] 94%|█████████▍| 376365/400000 [00:42<00:02, 9079.79it/s] 94%|█████████▍| 377275/400000 [00:42<00:02, 9009.87it/s] 95%|█████████▍| 378180/400000 [00:42<00:02, 9019.66it/s] 95%|█████████▍| 379083/400000 [00:42<00:02, 8968.21it/s] 95%|█████████▍| 379981/400000 [00:43<00:02, 8961.35it/s] 95%|█████████▌| 380878/400000 [00:43<00:02, 8901.54it/s] 95%|█████████▌| 381769/400000 [00:43<00:02, 8883.65it/s] 96%|█████████▌| 382658/400000 [00:43<00:01, 8718.24it/s] 96%|█████████▌| 383567/400000 [00:43<00:01, 8824.78it/s] 96%|█████████▌| 384475/400000 [00:43<00:01, 8899.34it/s] 96%|█████████▋| 385375/400000 [00:43<00:01, 8926.56it/s] 97%|█████████▋| 386269/400000 [00:43<00:01, 8875.51it/s] 97%|█████████▋| 387169/400000 [00:43<00:01, 8912.33it/s] 97%|█████████▋| 388068/400000 [00:43<00:01, 8934.63it/s] 97%|█████████▋| 388962/400000 [00:44<00:01, 8749.05it/s] 97%|█████████▋| 389838/400000 [00:44<00:01, 8741.99it/s] 98%|█████████▊| 390730/400000 [00:44<00:01, 8793.83it/s] 98%|█████████▊| 391651/400000 [00:44<00:00, 8912.35it/s] 98%|█████████▊| 392577/400000 [00:44<00:00, 9010.88it/s] 98%|█████████▊| 393479/400000 [00:44<00:00, 9008.41it/s] 99%|█████████▊| 394381/400000 [00:44<00:00, 8965.62it/s] 99%|█████████▉| 395279/400000 [00:44<00:00, 8877.22it/s] 99%|█████████▉| 396168/400000 [00:44<00:00, 8814.52it/s] 99%|█████████▉| 397050/400000 [00:44<00:00, 8730.03it/s] 99%|█████████▉| 397924/400000 [00:45<00:00, 8672.87it/s]100%|█████████▉| 398850/400000 [00:45<00:00, 8840.37it/s]100%|█████████▉| 399744/400000 [00:45<00:00, 8869.51it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8832.62it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdc5d48b0f0>, <torchtext.data.dataset.TabularDataset object at 0x7fdc5d48b240>, <torchtext.vocab.Vocab object at 0x7fdc5d48b160>), {}) 

