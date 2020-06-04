
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f58d41f4378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f58d41f4378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f59462440d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f59462440d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f58dd9d89d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f58dd9d89d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58f3b288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58f3b288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f58f3b288c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 65536/9912422 [00:00<00:22, 432987.69it/s]  4%|▍         | 425984/9912422 [00:00<00:16, 575306.93it/s] 17%|█▋        | 1638400/9912422 [00:00<00:10, 801246.88it/s] 48%|████▊     | 4784128/9912422 [00:00<00:04, 1132208.80it/s] 67%|██████▋   | 6651904/9912422 [00:01<00:02, 1572816.39it/s] 88%|████████▊ | 8716288/9912422 [00:01<00:00, 2167121.60it/s]9920512it [00:01, 8804386.10it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 114572.94it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  4%|▍         | 65536/1648877 [00:00<00:03, 475133.28it/s] 26%|██▌       | 425984/1648877 [00:00<00:01, 627491.39it/s]1654784it [00:00, 2220369.05it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 46454.30it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f58d7200d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f58d7200fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f58f3b28510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f58f3b28510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f58f3b28510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:09,  1.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:09,  1.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:09,  1.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:08,  1.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:28,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<01:02,  2.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:02,  2.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:02,  2.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:01,  2.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:01,  2.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:01,  2.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:43,  3.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:43,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:43,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:30,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:30,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:21,  6.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:21,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:21,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  8.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  8.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:14,  8.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  8.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 22.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 27.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 27.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 27.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 27.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 27.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 27.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 31.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 31.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 35.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 35.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 38.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 41.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 44.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 44.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 44.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 44.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 44.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 44.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 45.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 45.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 47.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 48.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 49.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 49.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 49.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 49.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 49.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 49.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 43.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 43.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 43.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 43.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 43.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 43.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 43.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 43.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 43.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 43.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 43.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 43.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 44.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 44.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 44.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 44.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 44.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 44.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 44.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 44.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 46.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 46.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 46.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 46.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 46.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 46.05 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 45.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 45.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 45.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 45.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 45.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 45.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 45.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 45.73 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.48s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 45.73 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.48s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.48s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.01 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.48s/ url]
0 examples [00:00, ? examples/s]2020-06-04 18:09:46.886169: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-04 18:09:46.901935: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-04 18:09:46.902799: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56290fe90110 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-04 18:09:46.902818: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
27 examples [00:00, 269.15 examples/s]121 examples [00:00, 342.45 examples/s]216 examples [00:00, 423.37 examples/s]301 examples [00:00, 498.05 examples/s]392 examples [00:00, 576.10 examples/s]487 examples [00:00, 651.99 examples/s]584 examples [00:00, 722.83 examples/s]679 examples [00:00, 778.27 examples/s]766 examples [00:00, 786.74 examples/s]858 examples [00:01, 821.98 examples/s]955 examples [00:01, 860.93 examples/s]1046 examples [00:01, 874.20 examples/s]1137 examples [00:01, 877.21 examples/s]1231 examples [00:01, 892.86 examples/s]1328 examples [00:01, 914.07 examples/s]1426 examples [00:01, 930.50 examples/s]1520 examples [00:01, 903.81 examples/s]1612 examples [00:01, 899.25 examples/s]1703 examples [00:01, 896.46 examples/s]1796 examples [00:02, 905.01 examples/s]1890 examples [00:02, 912.20 examples/s]1984 examples [00:02, 919.38 examples/s]2077 examples [00:02, 909.61 examples/s]2169 examples [00:02, 908.92 examples/s]2260 examples [00:02, 898.47 examples/s]2350 examples [00:02, 895.23 examples/s]2447 examples [00:02, 916.19 examples/s]2539 examples [00:02, 902.65 examples/s]2630 examples [00:02, 891.08 examples/s]2728 examples [00:03, 913.38 examples/s]2825 examples [00:03, 926.91 examples/s]2918 examples [00:03, 921.96 examples/s]3011 examples [00:03, 920.20 examples/s]3104 examples [00:03, 914.42 examples/s]3198 examples [00:03, 921.05 examples/s]3297 examples [00:03, 938.93 examples/s]3392 examples [00:03, 932.02 examples/s]3486 examples [00:03, 926.59 examples/s]3579 examples [00:03, 901.69 examples/s]3672 examples [00:04, 909.30 examples/s]3767 examples [00:04, 919.90 examples/s]3860 examples [00:04, 921.53 examples/s]3953 examples [00:04, 914.70 examples/s]4045 examples [00:04, 909.52 examples/s]4145 examples [00:04, 932.69 examples/s]4241 examples [00:04, 940.38 examples/s]4337 examples [00:04, 942.30 examples/s]4433 examples [00:04, 946.32 examples/s]4528 examples [00:04, 931.98 examples/s]4627 examples [00:05, 946.37 examples/s]4725 examples [00:05, 955.96 examples/s]4821 examples [00:05, 940.91 examples/s]4919 examples [00:05, 951.09 examples/s]5015 examples [00:05, 917.74 examples/s]5110 examples [00:05, 926.51 examples/s]5203 examples [00:05, 916.59 examples/s]5295 examples [00:05, 901.44 examples/s]5389 examples [00:05, 910.22 examples/s]5481 examples [00:06, 887.11 examples/s]5570 examples [00:06, 886.08 examples/s]5661 examples [00:06, 892.94 examples/s]5751 examples [00:06, 893.96 examples/s]5841 examples [00:06, 890.56 examples/s]5933 examples [00:06, 896.36 examples/s]6027 examples [00:06, 906.35 examples/s]6121 examples [00:06, 913.81 examples/s]6217 examples [00:06, 926.80 examples/s]6314 examples [00:06, 938.77 examples/s]6408 examples [00:07, 923.01 examples/s]6508 examples [00:07, 944.70 examples/s]6603 examples [00:07, 939.48 examples/s]6698 examples [00:07, 926.47 examples/s]6791 examples [00:07, 894.00 examples/s]6881 examples [00:07, 879.03 examples/s]6971 examples [00:07, 883.67 examples/s]7064 examples [00:07, 889.30 examples/s]7154 examples [00:07, 863.61 examples/s]7244 examples [00:07, 871.89 examples/s]7335 examples [00:08, 882.72 examples/s]7431 examples [00:08, 904.05 examples/s]7522 examples [00:08, 900.60 examples/s]7613 examples [00:08, 897.03 examples/s]7704 examples [00:08, 897.52 examples/s]7799 examples [00:08, 911.29 examples/s]7892 examples [00:08, 914.44 examples/s]7984 examples [00:08, 913.75 examples/s]8076 examples [00:08, 910.49 examples/s]8169 examples [00:09, 915.92 examples/s]8261 examples [00:09, 902.82 examples/s]8353 examples [00:09, 906.88 examples/s]8444 examples [00:09, 906.67 examples/s]8537 examples [00:09, 913.50 examples/s]8634 examples [00:09, 929.70 examples/s]8732 examples [00:09, 940.15 examples/s]8827 examples [00:09, 919.22 examples/s]8920 examples [00:09, 874.49 examples/s]9009 examples [00:09, 847.06 examples/s]9095 examples [00:10, 843.63 examples/s]9185 examples [00:10, 857.25 examples/s]9277 examples [00:10, 874.52 examples/s]9375 examples [00:10, 903.57 examples/s]9475 examples [00:10, 928.31 examples/s]9569 examples [00:10, 891.22 examples/s]9664 examples [00:10, 907.89 examples/s]9756 examples [00:10, 903.55 examples/s]9849 examples [00:10, 910.66 examples/s]9946 examples [00:10, 925.73 examples/s]10039 examples [00:11, 865.60 examples/s]10135 examples [00:11, 889.96 examples/s]10226 examples [00:11, 893.15 examples/s]10316 examples [00:11, 892.92 examples/s]10416 examples [00:11, 922.49 examples/s]10509 examples [00:11, 899.04 examples/s]10600 examples [00:11, 895.74 examples/s]10690 examples [00:11, 879.78 examples/s]10780 examples [00:11, 883.50 examples/s]10872 examples [00:12, 891.62 examples/s]10962 examples [00:12, 890.38 examples/s]11052 examples [00:12, 891.98 examples/s]11142 examples [00:12, 839.15 examples/s]11227 examples [00:12, 819.96 examples/s]11310 examples [00:12, 795.72 examples/s]11408 examples [00:12, 841.54 examples/s]11494 examples [00:12, 828.56 examples/s]11589 examples [00:12, 860.91 examples/s]11680 examples [00:12, 874.05 examples/s]11769 examples [00:13, 874.75 examples/s]11857 examples [00:13, 855.91 examples/s]11948 examples [00:13, 869.62 examples/s]12036 examples [00:13, 866.77 examples/s]12127 examples [00:13, 877.97 examples/s]12216 examples [00:13, 863.24 examples/s]12303 examples [00:13, 855.80 examples/s]12390 examples [00:13, 859.31 examples/s]12478 examples [00:13, 865.22 examples/s]12572 examples [00:14, 885.14 examples/s]12661 examples [00:14, 876.01 examples/s]12749 examples [00:14, 856.38 examples/s]12839 examples [00:14, 864.85 examples/s]12933 examples [00:14, 883.29 examples/s]13026 examples [00:14, 895.76 examples/s]13117 examples [00:14, 898.65 examples/s]13214 examples [00:14, 916.43 examples/s]13309 examples [00:14, 926.22 examples/s]13404 examples [00:14, 930.95 examples/s]13498 examples [00:15, 928.89 examples/s]13591 examples [00:15, 919.77 examples/s]13684 examples [00:15, 917.11 examples/s]13776 examples [00:15, 908.71 examples/s]13868 examples [00:15, 909.82 examples/s]13965 examples [00:15, 925.67 examples/s]14060 examples [00:15, 932.76 examples/s]14158 examples [00:15, 943.98 examples/s]14253 examples [00:15, 932.30 examples/s]14347 examples [00:15, 927.24 examples/s]14440 examples [00:16, 918.19 examples/s]14532 examples [00:16, 897.99 examples/s]14622 examples [00:16, 892.36 examples/s]14712 examples [00:16, 882.17 examples/s]14807 examples [00:16, 899.20 examples/s]14900 examples [00:16, 906.95 examples/s]14991 examples [00:16, 896.71 examples/s]15085 examples [00:16, 908.73 examples/s]15177 examples [00:16, 910.89 examples/s]15272 examples [00:16, 920.69 examples/s]15368 examples [00:17, 928.96 examples/s]15461 examples [00:17, 889.70 examples/s]15558 examples [00:17, 910.88 examples/s]15650 examples [00:17, 902.56 examples/s]15741 examples [00:17, 877.59 examples/s]15830 examples [00:17, 855.03 examples/s]15916 examples [00:17, 850.93 examples/s]16008 examples [00:17, 868.70 examples/s]16096 examples [00:17, 867.64 examples/s]16187 examples [00:18, 878.78 examples/s]16279 examples [00:18, 883.51 examples/s]16371 examples [00:18, 891.58 examples/s]16465 examples [00:18, 905.55 examples/s]16556 examples [00:18, 905.28 examples/s]16647 examples [00:18, 900.01 examples/s]16747 examples [00:18, 927.51 examples/s]16841 examples [00:18, 917.35 examples/s]16937 examples [00:18, 928.41 examples/s]17031 examples [00:18, 923.75 examples/s]17124 examples [00:19, 907.09 examples/s]17215 examples [00:19, 905.70 examples/s]17306 examples [00:19, 875.44 examples/s]17396 examples [00:19, 882.47 examples/s]17488 examples [00:19, 890.35 examples/s]17578 examples [00:19, 886.84 examples/s]17669 examples [00:19, 893.51 examples/s]17759 examples [00:19, 870.58 examples/s]17852 examples [00:19, 886.35 examples/s]17941 examples [00:19, 884.37 examples/s]18031 examples [00:20, 887.70 examples/s]18129 examples [00:20, 913.25 examples/s]18221 examples [00:20, 911.47 examples/s]18313 examples [00:20, 905.32 examples/s]18404 examples [00:20, 871.77 examples/s]18496 examples [00:20, 885.61 examples/s]18592 examples [00:20, 904.07 examples/s]18683 examples [00:20, 897.49 examples/s]18773 examples [00:20, 893.47 examples/s]18867 examples [00:20, 904.60 examples/s]18963 examples [00:21, 918.97 examples/s]19056 examples [00:21, 911.79 examples/s]19148 examples [00:21, 907.64 examples/s]19239 examples [00:21, 907.38 examples/s]19331 examples [00:21, 908.53 examples/s]19428 examples [00:21, 925.71 examples/s]19524 examples [00:21, 934.49 examples/s]19618 examples [00:21, 926.67 examples/s]19711 examples [00:21, 916.64 examples/s]19803 examples [00:22, 908.83 examples/s]19895 examples [00:22, 910.52 examples/s]19987 examples [00:22, 899.90 examples/s]20078 examples [00:22, 866.13 examples/s]20175 examples [00:22, 893.90 examples/s]20265 examples [00:22, 890.37 examples/s]20358 examples [00:22, 899.92 examples/s]20449 examples [00:22, 880.45 examples/s]20542 examples [00:22, 893.94 examples/s]20636 examples [00:22, 906.71 examples/s]20728 examples [00:23, 909.47 examples/s]20832 examples [00:23, 944.52 examples/s]20927 examples [00:23, 925.05 examples/s]21020 examples [00:23, 913.66 examples/s]21115 examples [00:23, 921.14 examples/s]21211 examples [00:23, 930.99 examples/s]21309 examples [00:23, 944.26 examples/s]21404 examples [00:23, 937.38 examples/s]21499 examples [00:23, 940.12 examples/s]21594 examples [00:23, 917.17 examples/s]21686 examples [00:24, 888.24 examples/s]21776 examples [00:24, 875.16 examples/s]21864 examples [00:24, 840.81 examples/s]21951 examples [00:24, 848.91 examples/s]22041 examples [00:24, 863.49 examples/s]22133 examples [00:24, 879.26 examples/s]22227 examples [00:24, 893.59 examples/s]22317 examples [00:24, 871.92 examples/s]22414 examples [00:24, 898.44 examples/s]22505 examples [00:25, 889.13 examples/s]22595 examples [00:25, 866.89 examples/s]22683 examples [00:25, 866.51 examples/s]22772 examples [00:25, 871.83 examples/s]22861 examples [00:25, 876.92 examples/s]22951 examples [00:25, 881.29 examples/s]23040 examples [00:25, 860.95 examples/s]23132 examples [00:25, 876.99 examples/s]23220 examples [00:25, 862.72 examples/s]23311 examples [00:25, 876.15 examples/s]23401 examples [00:26, 881.78 examples/s]23497 examples [00:26, 901.59 examples/s]23588 examples [00:26, 900.48 examples/s]23683 examples [00:26, 912.56 examples/s]23778 examples [00:26, 922.32 examples/s]23871 examples [00:26, 919.40 examples/s]23967 examples [00:26, 930.35 examples/s]24063 examples [00:26, 935.91 examples/s]24157 examples [00:26, 933.40 examples/s]24251 examples [00:26, 918.20 examples/s]24346 examples [00:27, 925.96 examples/s]24442 examples [00:27, 934.42 examples/s]24536 examples [00:27, 931.04 examples/s]24630 examples [00:27, 933.41 examples/s]24724 examples [00:27, 929.32 examples/s]24817 examples [00:27, 916.38 examples/s]24909 examples [00:27, 915.85 examples/s]25001 examples [00:27, 901.46 examples/s]25092 examples [00:27, 891.30 examples/s]25187 examples [00:27, 907.18 examples/s]25284 examples [00:28, 923.73 examples/s]25377 examples [00:28, 924.06 examples/s]25470 examples [00:28, 915.90 examples/s]25562 examples [00:28, 915.09 examples/s]25658 examples [00:28, 926.28 examples/s]25760 examples [00:28, 951.67 examples/s]25856 examples [00:28, 952.25 examples/s]25952 examples [00:28, 937.77 examples/s]26046 examples [00:28, 919.42 examples/s]26139 examples [00:29, 889.06 examples/s]26229 examples [00:29, 883.60 examples/s]26325 examples [00:29, 903.79 examples/s]26416 examples [00:29, 903.34 examples/s]26508 examples [00:29, 908.12 examples/s]26599 examples [00:29, 907.40 examples/s]26690 examples [00:29, 871.31 examples/s]26786 examples [00:29, 894.26 examples/s]26885 examples [00:29, 919.78 examples/s]26987 examples [00:29, 947.61 examples/s]27083 examples [00:30, 930.04 examples/s]27177 examples [00:30, 928.28 examples/s]27274 examples [00:30, 938.58 examples/s]27369 examples [00:30, 920.95 examples/s]27462 examples [00:30, 904.24 examples/s]27553 examples [00:30, 904.20 examples/s]27644 examples [00:30, 885.27 examples/s]27737 examples [00:30, 895.36 examples/s]27827 examples [00:30, 847.71 examples/s]27919 examples [00:30, 867.40 examples/s]28013 examples [00:31, 885.71 examples/s]28105 examples [00:31, 895.33 examples/s]28195 examples [00:31, 883.74 examples/s]28284 examples [00:31, 870.84 examples/s]28372 examples [00:31, 865.81 examples/s]28464 examples [00:31, 879.89 examples/s]28559 examples [00:31, 897.82 examples/s]28649 examples [00:31, 896.62 examples/s]28739 examples [00:31, 862.42 examples/s]28826 examples [00:32, 856.70 examples/s]28919 examples [00:32, 875.01 examples/s]29016 examples [00:32, 900.85 examples/s]29107 examples [00:32, 883.72 examples/s]29196 examples [00:32, 871.77 examples/s]29291 examples [00:32, 891.10 examples/s]29381 examples [00:32, 893.29 examples/s]29471 examples [00:32, 889.16 examples/s]29563 examples [00:32, 896.20 examples/s]29653 examples [00:32, 889.78 examples/s]29747 examples [00:33, 901.72 examples/s]29838 examples [00:33, 885.12 examples/s]29930 examples [00:33, 894.20 examples/s]30020 examples [00:33, 844.07 examples/s]30109 examples [00:33, 855.48 examples/s]30196 examples [00:33, 850.87 examples/s]30294 examples [00:33, 883.97 examples/s]30385 examples [00:33, 889.96 examples/s]30477 examples [00:33, 897.60 examples/s]30568 examples [00:33, 879.33 examples/s]30657 examples [00:34, 847.35 examples/s]30753 examples [00:34, 876.13 examples/s]30842 examples [00:34, 866.16 examples/s]30933 examples [00:34, 878.42 examples/s]31022 examples [00:34, 872.32 examples/s]31119 examples [00:34, 899.18 examples/s]31213 examples [00:34, 909.09 examples/s]31312 examples [00:34, 930.82 examples/s]31418 examples [00:34, 964.43 examples/s]31515 examples [00:35, 938.04 examples/s]31610 examples [00:35, 934.32 examples/s]31709 examples [00:35, 948.43 examples/s]31807 examples [00:35, 957.64 examples/s]31908 examples [00:35, 972.18 examples/s]32006 examples [00:35, 964.83 examples/s]32103 examples [00:35, 927.37 examples/s]32197 examples [00:35, 894.52 examples/s]32288 examples [00:35, 893.21 examples/s]32378 examples [00:35, 890.44 examples/s]32469 examples [00:36, 895.01 examples/s]32568 examples [00:36, 921.08 examples/s]32666 examples [00:36, 936.98 examples/s]32768 examples [00:36, 960.06 examples/s]32865 examples [00:36, 957.42 examples/s]32962 examples [00:36, 960.56 examples/s]33068 examples [00:36, 986.96 examples/s]33168 examples [00:36, 988.82 examples/s]33273 examples [00:36, 1005.28 examples/s]33376 examples [00:36, 1010.21 examples/s]33478 examples [00:37, 1005.09 examples/s]33579 examples [00:37, 990.74 examples/s] 33679 examples [00:37, 974.54 examples/s]33782 examples [00:37, 989.54 examples/s]33882 examples [00:37, 986.67 examples/s]33981 examples [00:37, 980.93 examples/s]34083 examples [00:37, 990.24 examples/s]34183 examples [00:37, 987.22 examples/s]34282 examples [00:37, 976.35 examples/s]34381 examples [00:37, 978.05 examples/s]34479 examples [00:38, 946.08 examples/s]34574 examples [00:38, 905.64 examples/s]34672 examples [00:38, 925.95 examples/s]34766 examples [00:38, 923.93 examples/s]34860 examples [00:38, 925.36 examples/s]34959 examples [00:38, 941.38 examples/s]35055 examples [00:38, 945.87 examples/s]35150 examples [00:38, 933.01 examples/s]35247 examples [00:38, 942.82 examples/s]35342 examples [00:39, 931.32 examples/s]35436 examples [00:39, 932.06 examples/s]35530 examples [00:39, 928.80 examples/s]35624 examples [00:39, 931.70 examples/s]35718 examples [00:39, 927.13 examples/s]35811 examples [00:39, 913.60 examples/s]35914 examples [00:39, 945.56 examples/s]36010 examples [00:39, 947.23 examples/s]36111 examples [00:39, 964.16 examples/s]36214 examples [00:39, 981.19 examples/s]36313 examples [00:40, 938.03 examples/s]36412 examples [00:40, 951.54 examples/s]36508 examples [00:40, 930.41 examples/s]36610 examples [00:40, 953.21 examples/s]36710 examples [00:40, 964.18 examples/s]36807 examples [00:40, 935.45 examples/s]36901 examples [00:40, 918.41 examples/s]36994 examples [00:40, 897.46 examples/s]37085 examples [00:40, 898.49 examples/s]37176 examples [00:41, 883.35 examples/s]37265 examples [00:41, 871.45 examples/s]37353 examples [00:41, 861.55 examples/s]37440 examples [00:41, 858.84 examples/s]37539 examples [00:41, 891.94 examples/s]37636 examples [00:41, 911.21 examples/s]37734 examples [00:41, 929.81 examples/s]37836 examples [00:41, 952.37 examples/s]37936 examples [00:41, 965.24 examples/s]38034 examples [00:41, 968.73 examples/s]38132 examples [00:42, 969.78 examples/s]38230 examples [00:42, 952.19 examples/s]38326 examples [00:42, 924.14 examples/s]38419 examples [00:42, 887.75 examples/s]38511 examples [00:42, 894.06 examples/s]38601 examples [00:42, 887.82 examples/s]38692 examples [00:42, 894.34 examples/s]38784 examples [00:42, 900.46 examples/s]38878 examples [00:42, 911.05 examples/s]38970 examples [00:42, 895.33 examples/s]39066 examples [00:43, 912.97 examples/s]39158 examples [00:43, 890.27 examples/s]39261 examples [00:43, 927.96 examples/s]39364 examples [00:43, 955.42 examples/s]39463 examples [00:43, 965.38 examples/s]39560 examples [00:43, 937.84 examples/s]39661 examples [00:43, 956.36 examples/s]39758 examples [00:43, 942.65 examples/s]39854 examples [00:43, 947.11 examples/s]39949 examples [00:44, 942.65 examples/s]40044 examples [00:44, 873.60 examples/s]40133 examples [00:44, 869.77 examples/s]40227 examples [00:44, 887.79 examples/s]40319 examples [00:44, 896.95 examples/s]40410 examples [00:44, 879.42 examples/s]40499 examples [00:44, 840.82 examples/s]40594 examples [00:44, 868.78 examples/s]40689 examples [00:44, 889.07 examples/s]40779 examples [00:44, 865.38 examples/s]40879 examples [00:45, 898.79 examples/s]40975 examples [00:45, 914.25 examples/s]41067 examples [00:45, 913.95 examples/s]41175 examples [00:45, 957.72 examples/s]41279 examples [00:45, 980.91 examples/s]41378 examples [00:45, 925.60 examples/s]41472 examples [00:45, 924.69 examples/s]41567 examples [00:45, 932.08 examples/s]41665 examples [00:45, 944.11 examples/s]41762 examples [00:45, 949.39 examples/s]41858 examples [00:46, 935.55 examples/s]41958 examples [00:46, 951.52 examples/s]42067 examples [00:46, 988.41 examples/s]42167 examples [00:46, 971.66 examples/s]42265 examples [00:46, 941.65 examples/s]42360 examples [00:46, 935.69 examples/s]42454 examples [00:46, 930.81 examples/s]42548 examples [00:46, 932.87 examples/s]42646 examples [00:46, 944.27 examples/s]42742 examples [00:47, 946.48 examples/s]42837 examples [00:47, 937.14 examples/s]42931 examples [00:47, 911.42 examples/s]43023 examples [00:47, 899.37 examples/s]43118 examples [00:47, 913.73 examples/s]43211 examples [00:47, 917.50 examples/s]43303 examples [00:47, 903.05 examples/s]43396 examples [00:47, 908.73 examples/s]43489 examples [00:47, 914.53 examples/s]43581 examples [00:47, 906.14 examples/s]43672 examples [00:48, 905.49 examples/s]43763 examples [00:48, 876.70 examples/s]43858 examples [00:48, 896.91 examples/s]43957 examples [00:48, 921.22 examples/s]44050 examples [00:48, 919.72 examples/s]44143 examples [00:48, 912.06 examples/s]44236 examples [00:48, 914.54 examples/s]44328 examples [00:48, 914.21 examples/s]44420 examples [00:48, 913.99 examples/s]44517 examples [00:48, 928.22 examples/s]44613 examples [00:49, 935.90 examples/s]44707 examples [00:49, 909.97 examples/s]44799 examples [00:49, 907.12 examples/s]44890 examples [00:49, 904.81 examples/s]44988 examples [00:49, 925.21 examples/s]45081 examples [00:49, 919.21 examples/s]45175 examples [00:49, 925.24 examples/s]45270 examples [00:49, 931.25 examples/s]45364 examples [00:49, 928.72 examples/s]45457 examples [00:50, 917.35 examples/s]45549 examples [00:50, 898.20 examples/s]45639 examples [00:50, 883.12 examples/s]45728 examples [00:50, 861.93 examples/s]45820 examples [00:50, 876.34 examples/s]45908 examples [00:50, 861.62 examples/s]46002 examples [00:50, 883.25 examples/s]46091 examples [00:50, 878.02 examples/s]46183 examples [00:50, 887.65 examples/s]46276 examples [00:50, 899.20 examples/s]46367 examples [00:51, 891.09 examples/s]46459 examples [00:51, 896.22 examples/s]46550 examples [00:51, 897.95 examples/s]46644 examples [00:51, 908.10 examples/s]46740 examples [00:51, 922.32 examples/s]46834 examples [00:51, 925.85 examples/s]46927 examples [00:51, 917.08 examples/s]47019 examples [00:51, 915.53 examples/s]47111 examples [00:51, 903.49 examples/s]47206 examples [00:51, 914.87 examples/s]47303 examples [00:52, 929.88 examples/s]47398 examples [00:52, 935.13 examples/s]47492 examples [00:52, 909.95 examples/s]47588 examples [00:52, 922.42 examples/s]47681 examples [00:52, 921.14 examples/s]47774 examples [00:52, 914.42 examples/s]47866 examples [00:52, 911.63 examples/s]47959 examples [00:52, 915.76 examples/s]48054 examples [00:52, 923.45 examples/s]48147 examples [00:52, 901.48 examples/s]48240 examples [00:53, 908.35 examples/s]48334 examples [00:53, 915.17 examples/s]48426 examples [00:53, 900.92 examples/s]48517 examples [00:53, 900.89 examples/s]48618 examples [00:53, 929.22 examples/s]48715 examples [00:53, 938.14 examples/s]48810 examples [00:53, 915.22 examples/s]48902 examples [00:53, 912.46 examples/s]48996 examples [00:53, 919.32 examples/s]49089 examples [00:54, 883.57 examples/s]49179 examples [00:54, 887.04 examples/s]49270 examples [00:54, 893.70 examples/s]49363 examples [00:54, 902.32 examples/s]49454 examples [00:54, 900.35 examples/s]49551 examples [00:54, 919.87 examples/s]49645 examples [00:54, 924.34 examples/s]49740 examples [00:54, 929.02 examples/s]49834 examples [00:54, 925.07 examples/s]49927 examples [00:54, 919.42 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6500/50000 [00:00<00:00, 64997.89 examples/s] 36%|███▌      | 17869/50000 [00:00<00:00, 74579.93 examples/s] 57%|█████▋    | 28281/50000 [00:00<00:00, 81517.43 examples/s] 79%|███████▉  | 39555/50000 [00:00<00:00, 88902.69 examples/s]                                                               0 examples [00:00, ? examples/s]65 examples [00:00, 648.43 examples/s]158 examples [00:00, 712.29 examples/s]250 examples [00:00, 762.35 examples/s]341 examples [00:00, 800.67 examples/s]428 examples [00:00, 819.80 examples/s]523 examples [00:00, 853.15 examples/s]615 examples [00:00, 871.29 examples/s]706 examples [00:00, 881.37 examples/s]792 examples [00:00, 872.68 examples/s]882 examples [00:01, 879.91 examples/s]975 examples [00:01, 892.49 examples/s]1067 examples [00:01, 900.00 examples/s]1157 examples [00:01, 893.66 examples/s]1250 examples [00:01, 902.75 examples/s]1341 examples [00:01, 899.03 examples/s]1431 examples [00:01, 726.93 examples/s]1514 examples [00:01, 754.48 examples/s]1614 examples [00:01, 814.06 examples/s]1709 examples [00:01, 850.28 examples/s]1802 examples [00:02, 870.68 examples/s]1893 examples [00:02, 882.04 examples/s]1983 examples [00:02, 880.69 examples/s]2076 examples [00:02, 892.40 examples/s]2169 examples [00:02, 901.26 examples/s]2260 examples [00:02, 899.64 examples/s]2351 examples [00:02, 885.36 examples/s]2440 examples [00:02, 870.12 examples/s]2528 examples [00:02, 830.40 examples/s]2612 examples [00:03, 813.71 examples/s]2702 examples [00:03, 836.87 examples/s]2787 examples [00:03, 840.09 examples/s]2872 examples [00:03, 841.43 examples/s]2959 examples [00:03, 848.61 examples/s]3049 examples [00:03, 861.43 examples/s]3137 examples [00:03, 865.41 examples/s]3234 examples [00:03, 891.82 examples/s]3324 examples [00:03, 879.67 examples/s]3413 examples [00:03, 860.61 examples/s]3507 examples [00:04, 882.64 examples/s]3596 examples [00:04, 871.34 examples/s]3689 examples [00:04, 886.77 examples/s]3778 examples [00:04, 855.08 examples/s]3864 examples [00:04, 852.81 examples/s]3950 examples [00:04, 853.04 examples/s]4040 examples [00:04, 865.84 examples/s]4133 examples [00:04, 882.27 examples/s]4222 examples [00:04, 883.27 examples/s]4315 examples [00:04, 895.42 examples/s]4411 examples [00:05, 912.70 examples/s]4510 examples [00:05, 933.27 examples/s]4604 examples [00:05, 926.17 examples/s]4697 examples [00:05, 912.33 examples/s]4789 examples [00:05, 914.26 examples/s]4881 examples [00:05, 912.49 examples/s]4973 examples [00:05, 902.83 examples/s]5064 examples [00:05, 880.71 examples/s]5153 examples [00:05, 866.84 examples/s]5241 examples [00:06, 868.72 examples/s]5328 examples [00:06, 856.60 examples/s]5415 examples [00:06, 858.98 examples/s]5503 examples [00:06, 862.80 examples/s]5597 examples [00:06, 884.42 examples/s]5688 examples [00:06, 889.46 examples/s]5781 examples [00:06, 900.78 examples/s]5874 examples [00:06, 907.10 examples/s]5965 examples [00:06, 904.62 examples/s]6056 examples [00:06, 905.32 examples/s]6147 examples [00:07, 883.57 examples/s]6236 examples [00:07, 884.28 examples/s]6325 examples [00:07, 882.46 examples/s]6416 examples [00:07, 888.29 examples/s]6507 examples [00:07, 894.44 examples/s]6606 examples [00:07, 920.21 examples/s]6701 examples [00:07, 927.00 examples/s]6794 examples [00:07, 926.40 examples/s]6889 examples [00:07, 932.21 examples/s]6983 examples [00:07, 934.53 examples/s]7083 examples [00:08, 951.49 examples/s]7179 examples [00:08, 953.04 examples/s]7275 examples [00:08, 938.38 examples/s]7369 examples [00:08, 928.11 examples/s]7462 examples [00:08, 926.88 examples/s]7560 examples [00:08, 940.07 examples/s]7658 examples [00:08, 948.93 examples/s]7753 examples [00:08, 931.71 examples/s]7847 examples [00:08, 887.63 examples/s]7937 examples [00:08, 879.06 examples/s]8026 examples [00:09, 880.50 examples/s]8118 examples [00:09, 891.98 examples/s]8210 examples [00:09, 898.58 examples/s]8301 examples [00:09, 889.93 examples/s]8393 examples [00:09, 897.80 examples/s]8486 examples [00:09, 905.87 examples/s]8581 examples [00:09, 916.44 examples/s]8673 examples [00:09, 901.31 examples/s]8764 examples [00:09, 900.07 examples/s]8855 examples [00:09, 899.14 examples/s]8945 examples [00:10, 876.90 examples/s]9034 examples [00:10, 878.26 examples/s]9122 examples [00:10, 871.26 examples/s]9210 examples [00:10, 862.95 examples/s]9297 examples [00:10, 860.59 examples/s]9384 examples [00:10, 835.05 examples/s]9473 examples [00:10, 850.49 examples/s]9567 examples [00:10, 873.47 examples/s]9658 examples [00:10, 883.08 examples/s]9758 examples [00:11, 915.02 examples/s]9858 examples [00:11, 937.93 examples/s]9956 examples [00:11, 949.78 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9HCXAQ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9HCXAQ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58f3b288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58f3b288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f58f3b288c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f595ffa2e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f58d50fe5f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f58f3b288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f58f3b288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f58f3b288c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f58d50b9400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f58cbb9bf28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f58d408a7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f58d408a7b8> 

  function with postional parmater data_info <function split_train_valid at 0x7f58d408a7b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f58d408a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f58d408a8c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f58d408a8c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f0c31da67f4c541c5053bd0edea3fdd4bb6d72553fa420fb7f0985a75a2eeed8
  Stored in directory: /tmp/pip-ephem-wheel-cache-1lzitkk3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<131:24:09, 1.82kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:04<91:57:14, 2.60kB/s]  .vector_cache/glove.6B.zip:   1%|          | 5.80M/862M [00:04<63:58:10, 3.72kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:04<44:25:28, 5.31kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:04<30:53:03, 7.59kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:05<21:22:45, 10.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.1M/862M [00:05<14:47:59, 15.5kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.5M/862M [00:05<10:14:32, 22.1kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:05<7:07:28, 31.6kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:06<4:58:35, 45.0kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:07<12:20:22, 18.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<8:37:41, 25.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:08<10:55:19, 20.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:08<7:38:07, 29.2kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:09<11:01:23, 20.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:11<7:42:30, 28.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:11<5:24:39, 41.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:11<3:47:28, 58.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:12<7:37:03, 29.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:12<5:19:40, 41.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:13<9:21:28, 23.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:13<6:32:36, 33.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:14<10:02:22, 21.9kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:16<7:01:21, 31.2kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:16<4:55:51, 44.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:16<3:27:20, 63.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:17<7:17:57, 29.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:19<5:06:49, 42.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:19<3:36:06, 60.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:19<2:31:34, 85.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:20<6:38:12, 32.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:22<4:39:07, 46.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:22<3:16:24, 65.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<2:17:52, 93.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:23<6:20:04, 33.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:23<4:25:54, 48.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:24<8:32:12, 25.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:24<5:57:03, 35.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:26<4:20:13, 49.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:26<3:03:05, 69.8kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:26<2:08:34, 99.1kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:27<6:12:30, 34.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<4:21:10, 48.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<3:03:48, 68.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<2:09:01, 98.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<6:14:19, 33.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<4:21:51, 48.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<8:23:35, 25.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:31<5:52:07, 35.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<9:18:10, 22.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<6:30:23, 32.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<4:34:09, 45.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<3:12:13, 64.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<6:33:59, 31.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<4:35:34, 45.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<8:28:29, 24.5kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<5:55:31, 34.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<9:15:45, 22.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<6:28:28, 31.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<9:47:18, 21.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<6:50:36, 30.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<9:32:21, 21.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:39<6:40:04, 30.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:40<9:48:26, 20.9kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<6:51:21, 29.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<4:48:45, 42.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<3:22:18, 60.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<6:55:13, 29.4kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<4:50:23, 41.9kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:44<8:24:31, 24.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<5:52:47, 34.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<8:47:40, 23.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<6:08:57, 32.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<8:54:55, 22.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<6:12:43, 32.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<4:32:14, 44.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<3:11:26, 62.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<2:14:18, 89.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<6:02:37, 33.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<4:13:44, 47.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<7:34:35, 26.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<5:17:50, 37.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<8:37:31, 23.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<6:01:48, 32.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<8:53:42, 22.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<6:13:02, 31.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<9:14:39, 21.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<6:26:45, 30.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<4:35:59, 42.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<3:14:16, 60.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<2:16:15, 86.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<5:50:05, 33.5kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<4:04:52, 47.8kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<7:41:15, 25.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<5:22:32, 36.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<8:14:05, 23.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<5:45:22, 33.7kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<8:50:03, 21.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<6:09:08, 31.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<4:33:51, 42.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<3:12:48, 59.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<2:15:13, 85.2kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<5:41:11, 33.8kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<3:58:38, 48.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<7:32:16, 25.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<5:15:06, 36.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<3:49:18, 49.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<2:41:24, 70.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<1:53:24, 100kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<5:07:05, 37.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<3:33:53, 52.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:08<2:44:47, 68.7kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<1:56:13, 97.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:08<1:21:44, 138kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:09<5:09:40, 36.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<3:35:50, 52.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<2:37:53, 71.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<1:51:21, 101kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<1:18:21, 143kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<4:58:45, 37.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<3:28:02, 53.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<2:44:09, 67.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<1:55:38, 96.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<1:21:21, 136kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<5:00:07, 36.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<3:29:13, 52.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<2:31:55, 72.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:17<1:47:17, 102kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<1:15:34, 145kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<4:30:48, 40.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<3:08:38, 57.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<2:22:33, 76.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<1:40:39, 108kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<1:10:50, 153kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<4:52:48, 37.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<3:23:55, 53.0kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<2:33:52, 70.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<1:48:27, 99.4kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<1:16:18, 141kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<4:49:27, 37.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<3:21:31, 53.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<2:36:38, 68.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<1:50:27, 96.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<1:17:40, 137kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<4:46:45, 37.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<3:19:39, 53.0kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<2:32:26, 69.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<1:47:32, 98.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<1:15:38, 139kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<4:40:58, 37.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<3:15:40, 53.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<2:27:04, 71.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<1:43:41, 101kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<1:12:58, 143kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<4:35:27, 37.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<3:11:59, 54.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<2:19:39, 74.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<1:38:32, 105kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<1:09:22, 149kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<4:28:55, 38.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<3:07:26, 54.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<2:16:14, 75.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<1:36:33, 106kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:38<1:07:54, 151kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<4:34:46, 37.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<3:11:56, 53.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<2:16:20, 74.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<1:36:47, 105kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<1:08:02, 149kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<4:32:13, 37.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<3:09:42, 53.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<2:17:46, 73.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<1:37:12, 103kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<1:08:23, 147kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<4:27:20, 37.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<3:06:16, 53.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<2:15:41, 73.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<1:35:41, 104kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<1:07:18, 147kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<4:31:58, 36.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<3:09:27, 52.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<2:18:10, 71.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<1:37:27, 101kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<1:08:33, 143kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<4:23:06, 37.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<3:03:14, 53.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<2:14:33, 72.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<1:34:56, 103kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<1:06:46, 145kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<4:22:10, 37.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<3:02:44, 52.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<2:11:48, 73.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<1:32:59, 104kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<1:05:24, 147kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<4:17:24, 37.3kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:57<2:59:20, 53.3kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<2:09:56, 73.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:59<1:31:41, 104kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<1:04:29, 147kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<4:16:02, 37.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<2:58:35, 53.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<2:07:40, 73.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<1:30:41, 104kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<1:03:44, 147kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<4:08:35, 37.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<2:53:48, 53.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<5:52:25, 26.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<4:05:45, 37.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<2:54:32, 53.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<2:03:26, 75.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<1:26:34, 107kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<4:24:15, 35.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<3:04:01, 50.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<2:13:45, 68.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<1:34:20, 97.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<1:06:20, 138kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<4:02:09, 37.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<2:49:21, 53.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<5:26:34, 27.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<3:48:12, 39.8kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<6:08:56, 24.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<4:16:57, 35.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<3:03:48, 49.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<2:09:20, 69.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<1:30:43, 98.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<4:22:49, 34.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<3:03:41, 48.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<5:46:04, 25.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<4:01:43, 36.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<6:29:47, 22.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<4:32:12, 32.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<6:46:46, 21.8kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<4:43:06, 31.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<3:23:10, 43.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<2:22:49, 61.6kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<1:40:07, 87.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<4:23:13, 33.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<3:03:14, 47.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:23<2:12:57, 65.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<1:33:59, 92.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<1:06:01, 131kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<3:52:46, 37.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<2:42:41, 53.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:25<5:25:52, 26.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<3:47:35, 37.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<6:10:45, 23.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<4:18:55, 33.0kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<6:18:09, 22.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<4:24:01, 32.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<6:35:48, 21.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<4:36:23, 30.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<6:28:23, 21.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<4:31:09, 31.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<6:35:20, 21.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<4:35:59, 30.5kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<6:41:41, 20.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<4:40:25, 29.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<6:40:10, 20.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<4:39:20, 29.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<6:41:06, 20.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<4:39:20, 29.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<3:18:18, 41.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<2:19:17, 59.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<1:37:41, 84.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<3:55:24, 35.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<2:43:47, 50.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<1:59:30, 68.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<1:24:15, 96.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<59:12, 137kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<3:42:23, 36.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<2:35:23, 52.1kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<5:08:03, 26.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<3:35:05, 37.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<5:50:29, 23.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<4:04:43, 32.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<5:56:08, 22.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<4:08:38, 32.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<5:58:17, 22.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<4:09:12, 31.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<2:58:54, 44.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<2:05:47, 62.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<1:28:12, 89.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<3:43:44, 35.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<2:36:18, 50.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<5:00:45, 26.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<3:29:08, 37.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<2:31:12, 51.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<1:46:22, 73.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<1:14:36, 104kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:50<3:45:16, 34.4kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<2:36:42, 49.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:52<1:53:38, 67.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<1:20:07, 95.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<56:22, 136kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<3:08:49, 40.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<2:11:19, 57.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<1:36:23, 78.6kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<1:07:59, 111kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<47:49, 158kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<3:23:36, 37.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<2:21:41, 52.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<1:42:20, 73.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<1:12:04, 104kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<50:41, 147kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<3:19:30, 37.3kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<2:18:43, 53.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<1:41:28, 72.6kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<1:11:34, 103kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<50:20, 146kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<3:07:38, 39.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<2:10:31, 55.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:04<1:34:41, 76.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<1:06:49, 109kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<47:01, 154kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<3:02:47, 39.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<2:07:07, 56.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<1:32:18, 77.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<1:05:37, 109kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<46:06, 154kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<3:01:54, 39.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<2:06:30, 55.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<1:31:57, 76.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<1:05:01, 108kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<45:41, 154kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<3:06:50, 37.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<2:10:00, 53.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<1:33:34, 74.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:13<1:06:32, 104kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<46:42, 148kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<3:03:10, 37.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<2:07:22, 53.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<1:32:15, 74.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<1:05:05, 105kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<45:46, 149kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<2:53:50, 39.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<2:00:49, 55.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<1:28:03, 76.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<1:02:07, 108kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<43:42, 153kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<2:49:09, 39.6kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<1:58:06, 56.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<4:07:05, 27.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<2:51:40, 38.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<2:03:37, 53.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:23<1:26:58, 75.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<1:00:58, 108kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<3:09:48, 34.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<2:12:28, 49.3kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<4:10:44, 26.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<2:54:09, 37.2kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<2:05:27, 51.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<1:28:14, 73.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<1:01:49, 104kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<3:07:24, 34.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<2:10:46, 48.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<4:07:18, 25.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<2:51:48, 36.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<2:03:07, 51.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<1:26:36, 72.9kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<1:00:45, 103kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<2:45:21, 38.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<1:54:51, 54.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<1:23:31, 74.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<58:54, 105kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<41:23, 149kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<2:41:25, 38.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<1:52:22, 54.6kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<1:20:08, 76.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<56:30, 108kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<39:42, 153kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<2:39:31, 38.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<1:51:20, 54.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<3:36:03, 27.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<2:30:39, 39.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<4:09:53, 24.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:40<2:53:51, 34.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<2:02:57, 48.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<1:26:25, 68.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<1:00:31, 97.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<2:50:33, 34.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<1:58:38, 49.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<1:24:31, 68.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<59:38, 97.6kB/s]  .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<41:51, 138kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<2:32:37, 37.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:46<1:46:05, 54.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<1:16:04, 75.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<53:38, 107kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<37:40, 151kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<2:30:00, 37.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<1:44:19, 54.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<1:14:29, 75.4kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<52:47, 106kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<37:03, 151kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<2:21:18, 39.5kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<1:38:35, 56.3kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<3:16:04, 28.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<2:16:25, 40.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<1:36:31, 56.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<1:08:17, 80.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<47:48, 114kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<2:30:59, 36.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<1:44:53, 51.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<1:15:06, 71.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<52:52, 101kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<37:08, 144kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<2:19:40, 38.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<1:37:03, 54.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<1:09:26, 75.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<48:58, 107kB/s]   .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<34:22, 152kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<2:20:10, 37.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<1:37:24, 53.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<1:09:32, 74.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<49:02, 105kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<34:25, 149kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<2:14:41, 38.1kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<1:33:29, 54.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<1:07:09, 75.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<47:18, 107kB/s]   .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<33:02, 152kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<25:42, 195kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<2:00:21, 41.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<1:23:34, 59.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<1:00:04, 82.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<42:40, 116kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<29:58, 164kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:11<1:58:17, 41.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<1:22:10, 59.4kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:13<58:47, 82.5kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:13<41:35, 116kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:13<29:11, 165kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:14<2:02:47, 39.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<1:25:13, 56.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<1:01:08, 77.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<43:07, 110kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<30:15, 156kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<2:06:47, 37.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<1:28:06, 53.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<1:02:36, 74.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<44:08, 105kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<30:58, 149kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<1:58:51, 38.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<1:22:25, 55.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<59:13, 76.6kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:22<41:45, 108kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<29:17, 154kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<1:57:54, 38.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<1:21:52, 54.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<58:20, 75.9kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<41:27, 107kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<29:01, 151kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<1:58:05, 37.2kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<1:21:54, 53.1kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<58:30, 73.9kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<41:15, 105kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<28:55, 148kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<1:51:03, 38.6kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:29<1:16:57, 55.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<55:15, 76.3kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<39:04, 108kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<27:24, 153kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<1:40:43, 41.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<1:09:51, 59.3kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<49:58, 82.3kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<35:15, 116kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<24:44, 165kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<1:45:59, 38.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<1:13:28, 54.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<52:26, 76.5kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<36:57, 108kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<25:54, 153kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<1:43:23, 38.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<1:11:55, 54.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<2:26:10, 26.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<1:41:36, 38.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<2:46:19, 23.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<1:55:34, 33.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<2:55:42, 22.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<2:01:54, 31.4kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<1:25:39, 44.4kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<1:00:09, 63.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:43<42:00, 89.6kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<1:50:33, 34.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<1:16:52, 48.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<2:22:57, 26.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<1:38:52, 37.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<1:10:15, 52.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<49:40, 73.6kB/s]  .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<34:33, 105kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<24:49, 145kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<17:38, 203kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<12:25, 286kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<1:30:03, 39.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:50<1:02:24, 56.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<44:21, 78.6kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<31:18, 111kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<21:47, 158kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<20:42, 167kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<1:18:34, 43.9kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<54:20, 62.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<38:59, 86.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<27:41, 122kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<19:10, 174kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<14:40, 226kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<10:31, 314kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<07:28, 438kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<1:20:56, 40.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<56:02, 57.8kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<39:50, 80.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<28:18, 113kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<19:46, 160kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<1:24:14, 37.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<58:11, 53.7kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<41:31, 74.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<29:22, 105kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:03<20:17, 150kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<15:35, 194kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<11:08, 271kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<07:52, 380kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<1:13:47, 40.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<51:02, 58.0kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<36:14, 80.8kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<25:44, 114kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<17:58, 161kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<1:16:14, 37.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<52:33, 54.1kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<37:34, 75.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<26:25, 106kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<18:21, 152kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<13:27, 205kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<09:37, 285kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<06:48, 399kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<1:04:46, 41.9kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<44:37, 59.9kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<31:57, 82.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<22:31, 117kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<15:28, 167kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<13:35, 190kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<09:42, 265kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<06:52, 370kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<58:18, 43.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<40:11, 62.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<28:38, 86.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<20:19, 121kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<14:00, 173kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<10:41, 225kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<07:48, 307kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<05:31, 428kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:24<53:05, 44.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<36:30, 63.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<26:06, 88.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<18:24, 124kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<12:40, 177kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<09:47, 228kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<06:58, 318kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<04:56, 443kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<52:11, 42.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<35:52, 59.9kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<25:30, 83.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<18:05, 117kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<12:24, 167kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<09:35, 214kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<06:59, 293kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<04:55, 410kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<48:20, 41.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<33:02, 59.6kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<23:50, 81.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:36<16:47, 116kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<11:41, 164kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<49:35, 38.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<33:51, 55.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<24:16, 75.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<17:04, 108kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<11:51, 152kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<48:49, 37.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<33:24, 52.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<23:38, 73.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<16:36, 104kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<11:31, 148kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<45:13, 37.6kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:43<30:44, 53.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<22:06, 73.9kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<15:32, 105kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<10:46, 148kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<43:00, 37.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<29:21, 53.0kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<20:41, 73.9kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<14:32, 105kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<10:04, 149kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<07:10, 203kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<05:06, 284kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<03:35, 395kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<31:33, 45.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<21:22, 64.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<15:17, 88.5kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<10:45, 125kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<07:15, 179kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<06:02, 212kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<04:23, 292kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<03:03, 408kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<30:06, 41.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<20:11, 59.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<14:37, 80.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<10:23, 113kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<07:02, 161kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<05:12, 213kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<03:43, 297kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<02:35, 414kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<25:57, 41.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<17:15, 59.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<12:31, 80.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<08:51, 113kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<05:54, 161kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<04:35, 204kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<03:15, 285kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<02:15, 398kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<22:16, 40.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<14:53, 57.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<10:21, 80.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<07:21, 112kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:08<04:59, 159kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:09<20:23, 38.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<13:15, 55.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<09:41, 74.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:11<06:46, 106kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<04:22, 151kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<04:01, 163kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<02:50, 228kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<01:55, 320kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<15:14, 40.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<09:55, 58.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<06:50, 80.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<04:49, 113kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<03:06, 161kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<02:17, 209kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<01:37, 292kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<01:05, 408kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<10:41, 41.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<06:30, 59.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<04:45, 78.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<03:17, 112kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<01:58, 156kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<01:22, 218kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:52, 307kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<06:49, 39.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<03:40, 56.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<02:46, 72.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<01:53, 102kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<01:08, 145kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<04:02, 40.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<01:50, 58.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<01:14, 77.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:48, 110kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:23, 156kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<01:30, 39.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:07, 56.6kB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:28:03,  6.02it/s]  0%|          | 726/400000 [00:00<12:54:30,  8.59it/s]  0%|          | 1453/400000 [00:00<9:01:26, 12.27it/s]  1%|          | 2220/400000 [00:00<6:18:32, 17.51it/s]  1%|          | 2973/400000 [00:00<4:24:44, 24.99it/s]  1%|          | 3739/400000 [00:00<3:05:13, 35.66it/s]  1%|          | 4506/400000 [00:00<2:09:39, 50.84it/s]  1%|▏         | 5302/400000 [00:00<1:30:49, 72.43it/s]  2%|▏         | 6078/400000 [00:00<1:03:42, 103.05it/s]  2%|▏         | 6843/400000 [00:01<44:45, 146.37it/s]    2%|▏         | 7589/400000 [00:01<31:32, 207.36it/s]  2%|▏         | 8345/400000 [00:01<22:17, 292.79it/s]  2%|▏         | 9138/400000 [00:01<15:49, 411.74it/s]  2%|▏         | 9898/400000 [00:01<11:18, 574.76it/s]  3%|▎         | 10674/400000 [00:01<08:09, 795.81it/s]  3%|▎         | 11466/400000 [00:01<05:56, 1089.89it/s]  3%|▎         | 12238/400000 [00:01<04:25, 1460.13it/s]  3%|▎         | 13004/400000 [00:01<03:20, 1928.23it/s]  3%|▎         | 13786/400000 [00:01<02:35, 2490.94it/s]  4%|▎         | 14612/400000 [00:02<02:02, 3151.21it/s]  4%|▍         | 15394/400000 [00:02<01:40, 3829.21it/s]  4%|▍         | 16173/400000 [00:02<01:27, 4395.44it/s]  4%|▍         | 16988/400000 [00:02<01:15, 5100.22it/s]  4%|▍         | 17755/400000 [00:02<01:07, 5660.00it/s]  5%|▍         | 18579/400000 [00:02<01:01, 6245.40it/s]  5%|▍         | 19401/400000 [00:02<00:56, 6730.08it/s]  5%|▌         | 20211/400000 [00:02<00:53, 7089.78it/s]  5%|▌         | 21010/400000 [00:02<00:52, 7277.42it/s]  5%|▌         | 21802/400000 [00:03<00:52, 7227.91it/s]  6%|▌         | 22576/400000 [00:03<00:51, 7374.12it/s]  6%|▌         | 23366/400000 [00:03<00:50, 7522.27it/s]  6%|▌         | 24158/400000 [00:03<00:49, 7635.32it/s]  6%|▌         | 24939/400000 [00:03<00:49, 7635.82it/s]  6%|▋         | 25720/400000 [00:03<00:48, 7686.34it/s]  7%|▋         | 26545/400000 [00:03<00:47, 7846.91it/s]  7%|▋         | 27357/400000 [00:03<00:47, 7926.83it/s]  7%|▋         | 28159/400000 [00:03<00:46, 7952.47it/s]  7%|▋         | 28962/400000 [00:03<00:46, 7972.85it/s]  7%|▋         | 29762/400000 [00:04<00:47, 7767.77it/s]  8%|▊         | 30542/400000 [00:04<00:48, 7625.47it/s]  8%|▊         | 31339/400000 [00:04<00:47, 7722.99it/s]  8%|▊         | 32188/400000 [00:04<00:46, 7936.37it/s]  8%|▊         | 33035/400000 [00:04<00:45, 8088.91it/s]  8%|▊         | 33847/400000 [00:04<00:45, 7993.06it/s]  9%|▊         | 34649/400000 [00:04<00:46, 7887.93it/s]  9%|▉         | 35465/400000 [00:04<00:45, 7966.84it/s]  9%|▉         | 36264/400000 [00:04<00:45, 7968.10it/s]  9%|▉         | 37064/400000 [00:04<00:45, 7974.92it/s]  9%|▉         | 37863/400000 [00:05<00:46, 7716.46it/s] 10%|▉         | 38637/400000 [00:05<00:47, 7627.08it/s] 10%|▉         | 39477/400000 [00:05<00:45, 7841.98it/s] 10%|█         | 40265/400000 [00:05<00:45, 7822.60it/s] 10%|█         | 41090/400000 [00:05<00:45, 7944.59it/s] 10%|█         | 41887/400000 [00:05<00:47, 7616.62it/s] 11%|█         | 42701/400000 [00:05<00:46, 7765.15it/s] 11%|█         | 43488/400000 [00:05<00:45, 7796.03it/s] 11%|█         | 44271/400000 [00:05<00:45, 7794.25it/s] 11%|█▏        | 45053/400000 [00:05<00:45, 7798.70it/s] 11%|█▏        | 45883/400000 [00:06<00:44, 7939.35it/s] 12%|█▏        | 46679/400000 [00:06<00:44, 7922.85it/s] 12%|█▏        | 47473/400000 [00:06<00:44, 7914.62it/s] 12%|█▏        | 48298/400000 [00:06<00:43, 8009.84it/s] 12%|█▏        | 49100/400000 [00:06<00:44, 7868.07it/s] 12%|█▏        | 49888/400000 [00:06<00:44, 7836.85it/s] 13%|█▎        | 50688/400000 [00:06<00:44, 7882.84it/s] 13%|█▎        | 51477/400000 [00:06<00:44, 7879.71it/s] 13%|█▎        | 52280/400000 [00:06<00:43, 7922.24it/s] 13%|█▎        | 53073/400000 [00:06<00:44, 7866.52it/s] 13%|█▎        | 53860/400000 [00:07<00:44, 7796.84it/s] 14%|█▎        | 54667/400000 [00:07<00:43, 7876.50it/s] 14%|█▍        | 55456/400000 [00:07<00:44, 7682.38it/s] 14%|█▍        | 56231/400000 [00:07<00:44, 7701.58it/s] 14%|█▍        | 57007/400000 [00:07<00:44, 7717.69it/s] 14%|█▍        | 57821/400000 [00:07<00:43, 7839.46it/s] 15%|█▍        | 58624/400000 [00:07<00:43, 7895.42it/s] 15%|█▍        | 59415/400000 [00:07<00:43, 7788.13it/s] 15%|█▌        | 60261/400000 [00:07<00:42, 7976.21it/s] 15%|█▌        | 61061/400000 [00:08<00:42, 7975.81it/s] 15%|█▌        | 61882/400000 [00:08<00:42, 8042.63it/s] 16%|█▌        | 62688/400000 [00:08<00:42, 7965.85it/s] 16%|█▌        | 63486/400000 [00:08<00:42, 7862.52it/s] 16%|█▌        | 64274/400000 [00:08<00:42, 7821.89it/s] 16%|█▋        | 65057/400000 [00:08<00:43, 7777.22it/s] 16%|█▋        | 65836/400000 [00:08<00:43, 7770.38it/s] 17%|█▋        | 66672/400000 [00:08<00:41, 7937.19it/s] 17%|█▋        | 67472/400000 [00:08<00:41, 7954.71it/s] 17%|█▋        | 68269/400000 [00:08<00:42, 7836.37it/s] 17%|█▋        | 69061/400000 [00:09<00:42, 7858.45it/s] 17%|█▋        | 69865/400000 [00:09<00:41, 7909.75it/s] 18%|█▊        | 70669/400000 [00:09<00:41, 7946.24it/s] 18%|█▊        | 71465/400000 [00:09<00:41, 7833.65it/s] 18%|█▊        | 72250/400000 [00:09<00:43, 7547.90it/s] 18%|█▊        | 73008/400000 [00:09<00:43, 7532.63it/s] 18%|█▊        | 73795/400000 [00:09<00:42, 7628.38it/s] 19%|█▊        | 74569/400000 [00:09<00:42, 7661.48it/s] 19%|█▉        | 75337/400000 [00:09<00:42, 7646.36it/s] 19%|█▉        | 76103/400000 [00:09<00:42, 7609.83it/s] 19%|█▉        | 76865/400000 [00:10<00:43, 7472.10it/s] 19%|█▉        | 77621/400000 [00:10<00:43, 7495.58it/s] 20%|█▉        | 78372/400000 [00:10<00:43, 7406.07it/s] 20%|█▉        | 79114/400000 [00:10<00:43, 7369.78it/s] 20%|█▉        | 79852/400000 [00:10<00:43, 7359.85it/s] 20%|██        | 80589/400000 [00:10<00:43, 7339.95it/s] 20%|██        | 81324/400000 [00:10<00:43, 7299.58it/s] 21%|██        | 82057/400000 [00:10<00:43, 7307.76it/s] 21%|██        | 82807/400000 [00:10<00:43, 7361.93it/s] 21%|██        | 83544/400000 [00:10<00:43, 7354.86it/s] 21%|██        | 84280/400000 [00:11<00:43, 7301.99it/s] 21%|██▏       | 85011/400000 [00:11<00:43, 7288.10it/s] 21%|██▏       | 85740/400000 [00:11<00:44, 7135.97it/s] 22%|██▏       | 86461/400000 [00:11<00:43, 7156.10it/s] 22%|██▏       | 87199/400000 [00:11<00:43, 7221.74it/s] 22%|██▏       | 87927/400000 [00:11<00:43, 7236.61it/s] 22%|██▏       | 88685/400000 [00:11<00:42, 7334.60it/s] 22%|██▏       | 89420/400000 [00:11<00:42, 7275.86it/s] 23%|██▎       | 90149/400000 [00:11<00:43, 7104.56it/s] 23%|██▎       | 90894/400000 [00:11<00:42, 7202.44it/s] 23%|██▎       | 91616/400000 [00:12<00:43, 7142.81it/s] 23%|██▎       | 92345/400000 [00:12<00:42, 7185.77it/s] 23%|██▎       | 93087/400000 [00:12<00:42, 7251.42it/s] 23%|██▎       | 93833/400000 [00:12<00:41, 7311.17it/s] 24%|██▎       | 94565/400000 [00:12<00:42, 7181.32it/s] 24%|██▍       | 95298/400000 [00:12<00:42, 7223.97it/s] 24%|██▍       | 96076/400000 [00:12<00:41, 7382.03it/s] 24%|██▍       | 96838/400000 [00:12<00:40, 7451.74it/s] 24%|██▍       | 97587/400000 [00:12<00:40, 7462.00it/s] 25%|██▍       | 98336/400000 [00:12<00:40, 7469.63it/s] 25%|██▍       | 99092/400000 [00:13<00:40, 7493.98it/s] 25%|██▍       | 99872/400000 [00:13<00:39, 7581.02it/s] 25%|██▌       | 100665/400000 [00:13<00:38, 7681.86it/s] 25%|██▌       | 101446/400000 [00:13<00:38, 7717.29it/s] 26%|██▌       | 102226/400000 [00:13<00:38, 7739.54it/s] 26%|██▌       | 103001/400000 [00:13<00:38, 7626.64it/s] 26%|██▌       | 103772/400000 [00:13<00:38, 7648.96it/s] 26%|██▌       | 104538/400000 [00:13<00:39, 7544.87it/s] 26%|██▋       | 105294/400000 [00:13<00:39, 7531.94it/s] 27%|██▋       | 106063/400000 [00:14<00:38, 7576.38it/s] 27%|██▋       | 106822/400000 [00:14<00:39, 7479.67it/s] 27%|██▋       | 107571/400000 [00:14<00:39, 7411.28it/s] 27%|██▋       | 108313/400000 [00:14<00:39, 7304.25it/s] 27%|██▋       | 109081/400000 [00:14<00:39, 7412.64it/s] 27%|██▋       | 109871/400000 [00:14<00:38, 7551.15it/s] 28%|██▊       | 110636/400000 [00:14<00:38, 7578.77it/s] 28%|██▊       | 111395/400000 [00:14<00:38, 7423.15it/s] 28%|██▊       | 112168/400000 [00:14<00:38, 7511.14it/s] 28%|██▊       | 112921/400000 [00:14<00:38, 7447.00it/s] 28%|██▊       | 113667/400000 [00:15<00:38, 7364.95it/s] 29%|██▊       | 114405/400000 [00:15<00:38, 7342.39it/s] 29%|██▉       | 115140/400000 [00:15<00:39, 7129.34it/s] 29%|██▉       | 115855/400000 [00:15<00:40, 7088.50it/s] 29%|██▉       | 116652/400000 [00:15<00:38, 7329.11it/s] 29%|██▉       | 117426/400000 [00:15<00:37, 7445.74it/s] 30%|██▉       | 118187/400000 [00:15<00:37, 7492.57it/s] 30%|██▉       | 118955/400000 [00:15<00:37, 7547.07it/s] 30%|██▉       | 119712/400000 [00:15<00:37, 7461.11it/s] 30%|███       | 120490/400000 [00:15<00:37, 7551.77it/s] 30%|███       | 121247/400000 [00:16<00:36, 7550.51it/s] 31%|███       | 122043/400000 [00:16<00:36, 7668.54it/s] 31%|███       | 122857/400000 [00:16<00:35, 7801.93it/s] 31%|███       | 123670/400000 [00:16<00:34, 7895.92it/s] 31%|███       | 124533/400000 [00:16<00:34, 8100.98it/s] 31%|███▏      | 125351/400000 [00:16<00:33, 8121.13it/s] 32%|███▏      | 126165/400000 [00:16<00:36, 7569.36it/s] 32%|███▏      | 126931/400000 [00:16<00:36, 7419.00it/s] 32%|███▏      | 127680/400000 [00:16<00:36, 7404.64it/s] 32%|███▏      | 128426/400000 [00:16<00:37, 7329.97it/s] 32%|███▏      | 129201/400000 [00:17<00:36, 7448.44it/s] 32%|███▏      | 129999/400000 [00:17<00:35, 7599.04it/s] 33%|███▎      | 130821/400000 [00:17<00:35, 7662.22it/s] 33%|███▎      | 131590/400000 [00:17<00:35, 7627.59it/s] 33%|███▎      | 132404/400000 [00:17<00:34, 7773.46it/s] 33%|███▎      | 133187/400000 [00:17<00:34, 7790.27it/s] 34%|███▎      | 134010/400000 [00:17<00:33, 7915.49it/s] 34%|███▎      | 134807/400000 [00:17<00:33, 7930.29it/s] 34%|███▍      | 135625/400000 [00:17<00:33, 8000.29it/s] 34%|███▍      | 136456/400000 [00:17<00:32, 8088.08it/s] 34%|███▍      | 137266/400000 [00:18<00:32, 8008.23it/s] 35%|███▍      | 138068/400000 [00:18<00:32, 8002.15it/s] 35%|███▍      | 138869/400000 [00:18<00:33, 7860.07it/s] 35%|███▍      | 139721/400000 [00:18<00:32, 8045.13it/s] 35%|███▌      | 140528/400000 [00:18<00:32, 8017.46it/s] 35%|███▌      | 141331/400000 [00:18<00:33, 7695.96it/s] 36%|███▌      | 142105/400000 [00:18<00:34, 7567.82it/s] 36%|███▌      | 142964/400000 [00:18<00:32, 7847.54it/s] 36%|███▌      | 143795/400000 [00:18<00:32, 7977.22it/s] 36%|███▌      | 144642/400000 [00:19<00:31, 8115.86it/s] 36%|███▋      | 145471/400000 [00:19<00:31, 8166.99it/s] 37%|███▋      | 146291/400000 [00:19<00:31, 7985.09it/s] 37%|███▋      | 147101/400000 [00:19<00:31, 8018.61it/s] 37%|███▋      | 147937/400000 [00:19<00:31, 8117.04it/s] 37%|███▋      | 148802/400000 [00:19<00:30, 8268.63it/s] 37%|███▋      | 149631/400000 [00:19<00:30, 8194.21it/s] 38%|███▊      | 150452/400000 [00:19<00:30, 8179.56it/s] 38%|███▊      | 151271/400000 [00:19<00:30, 8102.95it/s] 38%|███▊      | 152083/400000 [00:19<00:30, 8101.91it/s] 38%|███▊      | 152894/400000 [00:20<00:30, 8039.51it/s] 38%|███▊      | 153699/400000 [00:20<00:30, 8026.52it/s] 39%|███▊      | 154503/400000 [00:20<00:31, 7677.03it/s] 39%|███▉      | 155286/400000 [00:20<00:31, 7721.47it/s] 39%|███▉      | 156120/400000 [00:20<00:30, 7897.14it/s] 39%|███▉      | 156913/400000 [00:20<00:30, 7870.71it/s] 39%|███▉      | 157703/400000 [00:20<00:30, 7838.49it/s] 40%|███▉      | 158489/400000 [00:20<00:31, 7686.44it/s] 40%|███▉      | 159285/400000 [00:20<00:30, 7765.10it/s] 40%|████      | 160064/400000 [00:20<00:30, 7772.39it/s] 40%|████      | 160877/400000 [00:21<00:30, 7875.94it/s] 40%|████      | 161724/400000 [00:21<00:29, 8045.01it/s] 41%|████      | 162531/400000 [00:21<00:30, 7905.91it/s] 41%|████      | 163370/400000 [00:21<00:29, 8044.92it/s] 41%|████      | 164177/400000 [00:21<00:29, 8009.78it/s] 41%|████      | 164980/400000 [00:21<00:29, 8009.11it/s] 41%|████▏     | 165837/400000 [00:21<00:28, 8168.12it/s] 42%|████▏     | 166656/400000 [00:21<00:29, 7995.85it/s] 42%|████▏     | 167458/400000 [00:21<00:29, 7752.30it/s] 42%|████▏     | 168237/400000 [00:21<00:30, 7695.09it/s] 42%|████▏     | 169009/400000 [00:22<00:30, 7698.88it/s] 42%|████▏     | 169781/400000 [00:22<00:30, 7651.29it/s] 43%|████▎     | 170576/400000 [00:22<00:29, 7738.28it/s] 43%|████▎     | 171351/400000 [00:22<00:29, 7639.80it/s] 43%|████▎     | 172212/400000 [00:22<00:28, 7906.79it/s] 43%|████▎     | 173108/400000 [00:22<00:27, 8194.78it/s] 43%|████▎     | 173951/400000 [00:22<00:27, 8262.35it/s] 44%|████▎     | 174784/400000 [00:22<00:27, 8280.05it/s] 44%|████▍     | 175619/400000 [00:22<00:27, 8297.87it/s] 44%|████▍     | 176451/400000 [00:23<00:27, 8184.69it/s] 44%|████▍     | 177272/400000 [00:23<00:27, 8141.88it/s] 45%|████▍     | 178088/400000 [00:23<00:27, 8116.87it/s] 45%|████▍     | 178901/400000 [00:23<00:28, 7698.41it/s] 45%|████▍     | 179732/400000 [00:23<00:27, 7871.84it/s] 45%|████▌     | 180578/400000 [00:23<00:27, 8039.35it/s] 45%|████▌     | 181386/400000 [00:23<00:28, 7788.69it/s] 46%|████▌     | 182170/400000 [00:23<00:27, 7787.57it/s] 46%|████▌     | 182952/400000 [00:23<00:28, 7708.72it/s] 46%|████▌     | 183738/400000 [00:23<00:27, 7751.77it/s] 46%|████▌     | 184521/400000 [00:24<00:27, 7773.73it/s] 46%|████▋     | 185347/400000 [00:24<00:27, 7910.05it/s] 47%|████▋     | 186140/400000 [00:24<00:27, 7863.48it/s] 47%|████▋     | 186990/400000 [00:24<00:26, 8042.87it/s] 47%|████▋     | 187802/400000 [00:24<00:26, 8063.16it/s] 47%|████▋     | 188663/400000 [00:24<00:25, 8218.98it/s] 47%|████▋     | 189487/400000 [00:24<00:25, 8221.27it/s] 48%|████▊     | 190311/400000 [00:24<00:26, 7979.47it/s] 48%|████▊     | 191126/400000 [00:24<00:26, 8028.00it/s] 48%|████▊     | 191978/400000 [00:24<00:25, 8165.78it/s] 48%|████▊     | 192832/400000 [00:25<00:25, 8274.12it/s] 48%|████▊     | 193686/400000 [00:25<00:24, 8349.78it/s] 49%|████▊     | 194523/400000 [00:25<00:24, 8294.42it/s] 49%|████▉     | 195354/400000 [00:25<00:25, 8169.05it/s] 49%|████▉     | 196211/400000 [00:25<00:24, 8284.91it/s] 49%|████▉     | 197043/400000 [00:25<00:24, 8294.94it/s] 49%|████▉     | 197903/400000 [00:25<00:24, 8381.48it/s] 50%|████▉     | 198742/400000 [00:25<00:24, 8305.04it/s] 50%|████▉     | 199574/400000 [00:25<00:24, 8223.25it/s] 50%|█████     | 200398/400000 [00:25<00:24, 8165.11it/s] 50%|█████     | 201216/400000 [00:26<00:24, 7992.75it/s] 51%|█████     | 202017/400000 [00:26<00:24, 7977.80it/s] 51%|█████     | 202816/400000 [00:26<00:24, 7907.85it/s] 51%|█████     | 203611/400000 [00:26<00:24, 7918.50it/s] 51%|█████     | 204471/400000 [00:26<00:24, 8108.77it/s] 51%|█████▏    | 205302/400000 [00:26<00:23, 8166.62it/s] 52%|█████▏    | 206149/400000 [00:26<00:23, 8253.64it/s] 52%|█████▏    | 206976/400000 [00:26<00:23, 8134.80it/s] 52%|█████▏    | 207791/400000 [00:26<00:24, 7801.46it/s] 52%|█████▏    | 208594/400000 [00:27<00:24, 7866.23it/s] 52%|█████▏    | 209424/400000 [00:27<00:23, 7990.46it/s] 53%|█████▎    | 210226/400000 [00:27<00:24, 7824.69it/s] 53%|█████▎    | 211011/400000 [00:27<00:24, 7795.87it/s] 53%|█████▎    | 211809/400000 [00:27<00:23, 7846.86it/s] 53%|█████▎    | 212682/400000 [00:27<00:23, 8092.31it/s] 53%|█████▎    | 213540/400000 [00:27<00:22, 8232.02it/s] 54%|█████▎    | 214414/400000 [00:27<00:22, 8375.55it/s] 54%|█████▍    | 215254/400000 [00:27<00:22, 8075.42it/s] 54%|█████▍    | 216066/400000 [00:27<00:23, 7730.84it/s] 54%|█████▍    | 216846/400000 [00:28<00:23, 7699.70it/s] 54%|█████▍    | 217631/400000 [00:28<00:23, 7741.51it/s] 55%|█████▍    | 218409/400000 [00:28<00:23, 7670.52it/s] 55%|█████▍    | 219180/400000 [00:28<00:23, 7681.68it/s] 55%|█████▍    | 219960/400000 [00:28<00:23, 7714.48it/s] 55%|█████▌    | 220825/400000 [00:28<00:22, 7973.14it/s] 55%|█████▌    | 221626/400000 [00:28<00:22, 7899.70it/s] 56%|█████▌    | 222435/400000 [00:28<00:22, 7954.25it/s] 56%|█████▌    | 223233/400000 [00:28<00:22, 7692.79it/s] 56%|█████▌    | 224048/400000 [00:28<00:22, 7823.06it/s] 56%|█████▌    | 224842/400000 [00:29<00:22, 7857.31it/s] 56%|█████▋    | 225671/400000 [00:29<00:21, 7981.63it/s] 57%|█████▋    | 226471/400000 [00:29<00:21, 7920.43it/s] 57%|█████▋    | 227265/400000 [00:29<00:22, 7840.44it/s] 57%|█████▋    | 228063/400000 [00:29<00:21, 7879.63it/s] 57%|█████▋    | 228852/400000 [00:29<00:21, 7864.92it/s] 57%|█████▋    | 229640/400000 [00:29<00:21, 7804.53it/s] 58%|█████▊    | 230478/400000 [00:29<00:21, 7968.49it/s] 58%|█████▊    | 231277/400000 [00:29<00:21, 7835.76it/s] 58%|█████▊    | 232062/400000 [00:29<00:21, 7837.47it/s] 58%|█████▊    | 232847/400000 [00:30<00:21, 7825.03it/s] 58%|█████▊    | 233692/400000 [00:30<00:20, 8000.83it/s] 59%|█████▊    | 234494/400000 [00:30<00:21, 7868.26it/s] 59%|█████▉    | 235283/400000 [00:30<00:20, 7871.23it/s] 59%|█████▉    | 236072/400000 [00:30<00:20, 7844.29it/s] 59%|█████▉    | 236858/400000 [00:30<00:20, 7818.50it/s] 59%|█████▉    | 237660/400000 [00:30<00:20, 7876.78it/s] 60%|█████▉    | 238482/400000 [00:30<00:20, 7974.63it/s] 60%|█████▉    | 239281/400000 [00:30<00:20, 7802.49it/s] 60%|██████    | 240119/400000 [00:31<00:20, 7964.23it/s] 60%|██████    | 240923/400000 [00:31<00:19, 7985.99it/s] 60%|██████    | 241723/400000 [00:31<00:19, 7971.76it/s] 61%|██████    | 242522/400000 [00:31<00:20, 7857.81it/s] 61%|██████    | 243309/400000 [00:31<00:20, 7697.17it/s] 61%|██████    | 244081/400000 [00:31<00:20, 7649.13it/s] 61%|██████    | 244847/400000 [00:31<00:20, 7623.74it/s] 61%|██████▏   | 245629/400000 [00:31<00:20, 7679.34it/s] 62%|██████▏   | 246423/400000 [00:31<00:19, 7754.11it/s] 62%|██████▏   | 247200/400000 [00:31<00:19, 7675.87it/s] 62%|██████▏   | 248000/400000 [00:32<00:19, 7769.84it/s] 62%|██████▏   | 248790/400000 [00:32<00:19, 7807.43it/s] 62%|██████▏   | 249572/400000 [00:32<00:19, 7645.98it/s] 63%|██████▎   | 250364/400000 [00:32<00:19, 7724.40it/s] 63%|██████▎   | 251138/400000 [00:32<00:19, 7676.74it/s] 63%|██████▎   | 251953/400000 [00:32<00:18, 7812.05it/s] 63%|██████▎   | 252738/400000 [00:32<00:18, 7821.08it/s] 63%|██████▎   | 253577/400000 [00:32<00:18, 7982.14it/s] 64%|██████▎   | 254419/400000 [00:32<00:17, 8105.19it/s] 64%|██████▍   | 255231/400000 [00:32<00:18, 7972.21it/s] 64%|██████▍   | 256030/400000 [00:33<00:18, 7904.95it/s] 64%|██████▍   | 256822/400000 [00:33<00:18, 7877.73it/s] 64%|██████▍   | 257611/400000 [00:33<00:18, 7730.05it/s] 65%|██████▍   | 258386/400000 [00:33<00:18, 7728.43it/s] 65%|██████▍   | 259160/400000 [00:33<00:18, 7597.85it/s] 65%|██████▌   | 260014/400000 [00:33<00:17, 7857.11it/s] 65%|██████▌   | 260847/400000 [00:33<00:17, 7991.62it/s] 65%|██████▌   | 261676/400000 [00:33<00:17, 8076.84it/s] 66%|██████▌   | 262486/400000 [00:33<00:17, 8064.20it/s] 66%|██████▌   | 263294/400000 [00:33<00:17, 7887.52it/s] 66%|██████▌   | 264085/400000 [00:34<00:17, 7779.76it/s] 66%|██████▌   | 264958/400000 [00:34<00:16, 8041.22it/s] 66%|██████▋   | 265766/400000 [00:34<00:16, 8029.69it/s] 67%|██████▋   | 266640/400000 [00:34<00:16, 8228.85it/s] 67%|██████▋   | 267466/400000 [00:34<00:16, 8200.47it/s] 67%|██████▋   | 268333/400000 [00:34<00:15, 8333.09it/s] 67%|██████▋   | 269169/400000 [00:34<00:15, 8228.68it/s] 67%|██████▋   | 269994/400000 [00:34<00:16, 8065.29it/s] 68%|██████▊   | 270803/400000 [00:34<00:16, 7984.99it/s] 68%|██████▊   | 271604/400000 [00:34<00:16, 7959.08it/s] 68%|██████▊   | 272430/400000 [00:35<00:15, 8045.69it/s] 68%|██████▊   | 273236/400000 [00:35<00:15, 8044.68it/s] 69%|██████▊   | 274042/400000 [00:35<00:15, 8015.10it/s] 69%|██████▊   | 274845/400000 [00:35<00:15, 7967.76it/s] 69%|██████▉   | 275655/400000 [00:35<00:15, 8004.14it/s] 69%|██████▉   | 276484/400000 [00:35<00:15, 8085.83it/s] 69%|██████▉   | 277341/400000 [00:35<00:14, 8223.65it/s] 70%|██████▉   | 278212/400000 [00:35<00:14, 8362.89it/s] 70%|██████▉   | 279050/400000 [00:35<00:14, 8171.42it/s] 70%|██████▉   | 279869/400000 [00:36<00:15, 7941.96it/s] 70%|███████   | 280670/400000 [00:36<00:14, 7959.60it/s] 70%|███████   | 281489/400000 [00:36<00:14, 8023.90it/s] 71%|███████   | 282293/400000 [00:36<00:15, 7649.58it/s] 71%|███████   | 283063/400000 [00:36<00:15, 7543.36it/s] 71%|███████   | 283829/400000 [00:36<00:15, 7577.60it/s] 71%|███████   | 284594/400000 [00:36<00:15, 7598.86it/s] 71%|███████▏  | 285359/400000 [00:36<00:15, 7613.74it/s] 72%|███████▏  | 286129/400000 [00:36<00:14, 7638.23it/s] 72%|███████▏  | 286910/400000 [00:36<00:14, 7686.84it/s] 72%|███████▏  | 287680/400000 [00:37<00:14, 7598.72it/s] 72%|███████▏  | 288490/400000 [00:37<00:14, 7741.24it/s] 72%|███████▏  | 289284/400000 [00:37<00:14, 7798.99it/s] 73%|███████▎  | 290066/400000 [00:37<00:14, 7803.50it/s] 73%|███████▎  | 290847/400000 [00:37<00:14, 7638.83it/s] 73%|███████▎  | 291668/400000 [00:37<00:13, 7800.91it/s] 73%|███████▎  | 292513/400000 [00:37<00:13, 7983.01it/s] 73%|███████▎  | 293314/400000 [00:37<00:13, 7886.16it/s] 74%|███████▎  | 294105/400000 [00:37<00:13, 7864.36it/s] 74%|███████▎  | 294893/400000 [00:37<00:13, 7727.41it/s] 74%|███████▍  | 295668/400000 [00:38<00:13, 7489.32it/s] 74%|███████▍  | 296420/400000 [00:38<00:13, 7493.95it/s] 74%|███████▍  | 297172/400000 [00:38<00:14, 7257.90it/s] 74%|███████▍  | 297937/400000 [00:38<00:13, 7368.58it/s] 75%|███████▍  | 298693/400000 [00:38<00:13, 7424.98it/s] 75%|███████▍  | 299456/400000 [00:38<00:13, 7483.10it/s] 75%|███████▌  | 300220/400000 [00:38<00:13, 7529.29it/s] 75%|███████▌  | 301096/400000 [00:38<00:12, 7860.04it/s] 75%|███████▌  | 301914/400000 [00:38<00:12, 7952.60it/s] 76%|███████▌  | 302760/400000 [00:38<00:12, 8095.99it/s] 76%|███████▌  | 303573/400000 [00:39<00:12, 7875.99it/s] 76%|███████▌  | 304394/400000 [00:39<00:11, 7973.13it/s] 76%|███████▋  | 305264/400000 [00:39<00:11, 8177.67it/s] 77%|███████▋  | 306138/400000 [00:39<00:11, 8336.94it/s] 77%|███████▋  | 306975/400000 [00:39<00:11, 8256.09it/s] 77%|███████▋  | 307803/400000 [00:39<00:11, 8210.48it/s] 77%|███████▋  | 308626/400000 [00:39<00:11, 8036.45it/s] 77%|███████▋  | 309458/400000 [00:39<00:11, 8117.40it/s] 78%|███████▊  | 310298/400000 [00:39<00:10, 8199.32it/s] 78%|███████▊  | 311120/400000 [00:40<00:11, 8074.67it/s] 78%|███████▊  | 311929/400000 [00:40<00:11, 7923.85it/s] 78%|███████▊  | 312723/400000 [00:40<00:11, 7872.12it/s] 78%|███████▊  | 313512/400000 [00:40<00:11, 7803.86it/s] 79%|███████▊  | 314355/400000 [00:40<00:10, 7980.84it/s] 79%|███████▉  | 315225/400000 [00:40<00:10, 8183.68it/s] 79%|███████▉  | 316046/400000 [00:40<00:10, 8073.14it/s] 79%|███████▉  | 316856/400000 [00:40<00:10, 7978.28it/s] 79%|███████▉  | 317726/400000 [00:40<00:10, 8181.40it/s] 80%|███████▉  | 318550/400000 [00:40<00:09, 8197.30it/s] 80%|███████▉  | 319372/400000 [00:41<00:09, 8202.21it/s] 80%|████████  | 320194/400000 [00:41<00:10, 7955.81it/s] 80%|████████  | 320996/400000 [00:41<00:09, 7973.46it/s] 80%|████████  | 321843/400000 [00:41<00:09, 8113.45it/s] 81%|████████  | 322657/400000 [00:41<00:09, 7822.56it/s] 81%|████████  | 323443/400000 [00:41<00:09, 7781.88it/s] 81%|████████  | 324224/400000 [00:41<00:09, 7697.82it/s] 81%|████████  | 324996/400000 [00:41<00:09, 7669.06it/s] 81%|████████▏ | 325765/400000 [00:41<00:09, 7481.92it/s] 82%|████████▏ | 326617/400000 [00:41<00:09, 7763.79it/s] 82%|████████▏ | 327409/400000 [00:42<00:09, 7808.11it/s] 82%|████████▏ | 328237/400000 [00:42<00:09, 7943.30it/s] 82%|████████▏ | 329046/400000 [00:42<00:08, 7985.01it/s] 82%|████████▏ | 329847/400000 [00:42<00:08, 7965.44it/s] 83%|████████▎ | 330645/400000 [00:42<00:08, 7932.91it/s] 83%|████████▎ | 331447/400000 [00:42<00:08, 7954.71it/s] 83%|████████▎ | 332244/400000 [00:42<00:08, 7945.22it/s] 83%|████████▎ | 333064/400000 [00:42<00:08, 8017.60it/s] 83%|████████▎ | 333867/400000 [00:42<00:08, 7885.13it/s] 84%|████████▎ | 334657/400000 [00:42<00:08, 7833.65it/s] 84%|████████▍ | 335442/400000 [00:43<00:08, 7759.92it/s] 84%|████████▍ | 336219/400000 [00:43<00:08, 7530.92it/s] 84%|████████▍ | 337042/400000 [00:43<00:08, 7726.36it/s] 84%|████████▍ | 337818/400000 [00:43<00:08, 7392.05it/s] 85%|████████▍ | 338563/400000 [00:43<00:08, 7409.30it/s] 85%|████████▍ | 339308/400000 [00:43<00:08, 7336.73it/s] 85%|████████▌ | 340116/400000 [00:43<00:07, 7543.30it/s] 85%|████████▌ | 340916/400000 [00:43<00:07, 7673.01it/s] 85%|████████▌ | 341718/400000 [00:43<00:07, 7771.41it/s] 86%|████████▌ | 342498/400000 [00:44<00:07, 7724.24it/s] 86%|████████▌ | 343273/400000 [00:44<00:07, 7703.96it/s] 86%|████████▌ | 344045/400000 [00:44<00:07, 7678.32it/s] 86%|████████▌ | 344879/400000 [00:44<00:07, 7863.85it/s] 86%|████████▋ | 345684/400000 [00:44<00:06, 7917.43it/s] 87%|████████▋ | 346478/400000 [00:44<00:06, 7883.46it/s] 87%|████████▋ | 347288/400000 [00:44<00:06, 7946.00it/s] 87%|████████▋ | 348140/400000 [00:44<00:06, 8107.04it/s] 87%|████████▋ | 348960/400000 [00:44<00:06, 8131.53it/s] 87%|████████▋ | 349775/400000 [00:44<00:06, 7806.15it/s] 88%|████████▊ | 350592/400000 [00:45<00:06, 7910.03it/s] 88%|████████▊ | 351386/400000 [00:45<00:06, 7743.53it/s] 88%|████████▊ | 352173/400000 [00:45<00:06, 7780.72it/s] 88%|████████▊ | 352995/400000 [00:45<00:05, 7905.92it/s] 88%|████████▊ | 353831/400000 [00:45<00:05, 8035.31it/s] 89%|████████▊ | 354654/400000 [00:45<00:05, 8091.43it/s] 89%|████████▉ | 355465/400000 [00:45<00:05, 8032.34it/s] 89%|████████▉ | 356270/400000 [00:45<00:05, 7963.77it/s] 89%|████████▉ | 357136/400000 [00:45<00:05, 8159.38it/s] 90%|████████▉ | 358013/400000 [00:45<00:05, 8332.98it/s] 90%|████████▉ | 358849/400000 [00:46<00:04, 8242.54it/s] 90%|████████▉ | 359675/400000 [00:46<00:04, 8241.58it/s] 90%|█████████ | 360501/400000 [00:46<00:04, 7987.29it/s] 90%|█████████ | 361432/400000 [00:46<00:04, 8341.59it/s] 91%|█████████ | 362307/400000 [00:46<00:04, 8459.00it/s] 91%|█████████ | 363158/400000 [00:46<00:04, 8460.01it/s] 91%|█████████ | 364008/400000 [00:46<00:04, 8283.78it/s] 91%|█████████ | 364840/400000 [00:46<00:04, 7886.26it/s] 91%|█████████▏| 365635/400000 [00:46<00:04, 7736.69it/s] 92%|█████████▏| 366415/400000 [00:46<00:04, 7754.07it/s] 92%|█████████▏| 367201/400000 [00:47<00:04, 7783.19it/s] 92%|█████████▏| 367982/400000 [00:47<00:04, 7709.13it/s] 92%|█████████▏| 368803/400000 [00:47<00:03, 7850.69it/s] 92%|█████████▏| 369623/400000 [00:47<00:03, 7949.62it/s] 93%|█████████▎| 370420/400000 [00:47<00:03, 7904.43it/s] 93%|█████████▎| 371299/400000 [00:47<00:03, 8149.65it/s] 93%|█████████▎| 372117/400000 [00:47<00:03, 8040.71it/s] 93%|█████████▎| 372924/400000 [00:47<00:03, 8014.21it/s] 93%|█████████▎| 373742/400000 [00:47<00:03, 8062.70it/s] 94%|█████████▎| 374550/400000 [00:48<00:03, 8056.15it/s] 94%|█████████▍| 375357/400000 [00:48<00:03, 8022.71it/s] 94%|█████████▍| 376160/400000 [00:48<00:02, 7989.45it/s] 94%|█████████▍| 376960/400000 [00:48<00:02, 7829.90it/s] 94%|█████████▍| 377766/400000 [00:48<00:02, 7897.02it/s] 95%|█████████▍| 378557/400000 [00:48<00:02, 7826.80it/s] 95%|█████████▍| 379359/400000 [00:48<00:02, 7879.86it/s] 95%|█████████▌| 380148/400000 [00:48<00:02, 7786.94it/s] 95%|█████████▌| 380948/400000 [00:48<00:02, 7848.80it/s] 95%|█████████▌| 381795/400000 [00:48<00:02, 8025.27it/s] 96%|█████████▌| 382627/400000 [00:49<00:02, 8109.64it/s] 96%|█████████▌| 383462/400000 [00:49<00:02, 8178.04it/s] 96%|█████████▌| 384281/400000 [00:49<00:01, 8163.04it/s] 96%|█████████▋| 385135/400000 [00:49<00:01, 8270.96it/s] 96%|█████████▋| 385966/400000 [00:49<00:01, 8280.30it/s] 97%|█████████▋| 386795/400000 [00:49<00:01, 8250.47it/s] 97%|█████████▋| 387664/400000 [00:49<00:01, 8376.09it/s] 97%|█████████▋| 388503/400000 [00:49<00:01, 8271.60it/s] 97%|█████████▋| 389331/400000 [00:49<00:01, 8077.61it/s] 98%|█████████▊| 390141/400000 [00:49<00:01, 7988.86it/s] 98%|█████████▊| 390960/400000 [00:50<00:01, 8047.40it/s] 98%|█████████▊| 391766/400000 [00:50<00:01, 8049.30it/s] 98%|█████████▊| 392572/400000 [00:50<00:00, 7820.01it/s] 98%|█████████▊| 393356/400000 [00:50<00:00, 7634.67it/s] 99%|█████████▊| 394230/400000 [00:50<00:00, 7935.63it/s] 99%|█████████▉| 395029/400000 [00:50<00:00, 7907.55it/s] 99%|█████████▉| 395825/400000 [00:50<00:00, 7918.58it/s] 99%|█████████▉| 396620/400000 [00:50<00:00, 7800.90it/s] 99%|█████████▉| 397416/400000 [00:50<00:00, 7847.76it/s]100%|█████████▉| 398211/400000 [00:50<00:00, 7876.86it/s]100%|█████████▉| 399000/400000 [00:51<00:00, 7815.67it/s]100%|█████████▉| 399783/400000 [00:51<00:00, 7748.78it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7810.80it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f58d40b50f0>, <torchtext.data.dataset.TabularDataset object at 0x7f58d40b5240>, <torchtext.vocab.Vocab object at 0x7f58d40b5160>), {}) 

