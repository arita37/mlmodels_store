
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '31d2b88611b4231ac14a6179e72e5d4a4ee459ac', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/31d2b88611b4231ac14a6179e72e5d4a4ee459ac

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1544f72378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1544f72378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f15b6fc20d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f15b6fc20d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f154e7569d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f154e7569d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f15648a68c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f15648a68c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f15648a68c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 84%|████████▍ | 8355840/9912422 [00:00<00:00, 82609152.94it/s]9920512it [00:00, 37963651.99it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 278470.74it/s]           
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 943689.60it/s]1654784it [00:00, 12387744.01it/s]                         
0it [00:00, ?it/s]8192it [00:00, 136118.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f154de5c940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1545efe048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f15648a6510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f15648a6510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f15648a6510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:12,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:12,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:11,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:11,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:10,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:48,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:48,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:47,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:47,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:46,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:32,  4.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:30,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:20,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:20,  6.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 11.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 27.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 35.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 43.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 52.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 61.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 70.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 83.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.17 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.17 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.32 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ url]
0 examples [00:00, ? examples/s]2020-06-05 06:11:41.389580: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-05 06:11:41.509048: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-05 06:11:41.509349: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b802e34190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-05 06:11:41.509376: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.41 examples/s]103 examples [00:00,  3.44 examples/s]203 examples [00:00,  4.91 examples/s]299 examples [00:00,  7.00 examples/s]409 examples [00:00,  9.97 examples/s]521 examples [00:00, 14.19 examples/s]630 examples [00:01, 20.15 examples/s]737 examples [00:01, 28.56 examples/s]841 examples [00:01, 40.32 examples/s]950 examples [00:01, 56.69 examples/s]1054 examples [00:01, 79.13 examples/s]1165 examples [00:01, 109.68 examples/s]1275 examples [00:01, 150.23 examples/s]1385 examples [00:01, 202.70 examples/s]1496 examples [00:01, 268.43 examples/s]1607 examples [00:01, 347.31 examples/s]1719 examples [00:02, 437.59 examples/s]1829 examples [00:02, 532.11 examples/s]1940 examples [00:02, 630.28 examples/s]2052 examples [00:02, 724.52 examples/s]2164 examples [00:02, 809.74 examples/s]2275 examples [00:02, 870.50 examples/s]2386 examples [00:02, 928.90 examples/s]2496 examples [00:02, 966.02 examples/s]2607 examples [00:02, 1002.65 examples/s]2717 examples [00:02, 1029.19 examples/s]2827 examples [00:03, 1011.76 examples/s]2939 examples [00:03, 1041.19 examples/s]3048 examples [00:03, 1054.30 examples/s]3159 examples [00:03, 1069.69 examples/s]3271 examples [00:03, 1083.42 examples/s]3383 examples [00:03, 1093.71 examples/s]3495 examples [00:03, 1100.87 examples/s]3608 examples [00:03, 1108.67 examples/s]3720 examples [00:03, 1110.15 examples/s]3834 examples [00:03, 1117.47 examples/s]3948 examples [00:04, 1122.15 examples/s]4061 examples [00:04, 1117.80 examples/s]4174 examples [00:04, 1119.01 examples/s]4286 examples [00:04, 1114.48 examples/s]4399 examples [00:04, 1117.18 examples/s]4511 examples [00:04, 1116.20 examples/s]4623 examples [00:04, 1106.81 examples/s]4735 examples [00:04, 1110.65 examples/s]4847 examples [00:04, 1106.12 examples/s]4958 examples [00:04, 1106.63 examples/s]5069 examples [00:05, 1107.40 examples/s]5180 examples [00:05, 1091.79 examples/s]5291 examples [00:05, 1094.76 examples/s]5401 examples [00:05, 1053.38 examples/s]5508 examples [00:05, 1056.18 examples/s]5619 examples [00:05, 1070.85 examples/s]5728 examples [00:05, 1075.67 examples/s]5836 examples [00:05, 1055.94 examples/s]5947 examples [00:05, 1070.33 examples/s]6055 examples [00:05, 1071.70 examples/s]6163 examples [00:06, 1039.78 examples/s]6272 examples [00:06, 1054.09 examples/s]6382 examples [00:06, 1066.63 examples/s]6493 examples [00:06, 1077.54 examples/s]6605 examples [00:06, 1089.85 examples/s]6715 examples [00:06, 1092.47 examples/s]6826 examples [00:06, 1096.29 examples/s]6936 examples [00:06, 1075.10 examples/s]7047 examples [00:06, 1083.31 examples/s]7159 examples [00:07, 1092.10 examples/s]7271 examples [00:07, 1099.59 examples/s]7382 examples [00:07, 1099.45 examples/s]7493 examples [00:07, 1098.01 examples/s]7603 examples [00:07, 1098.57 examples/s]7716 examples [00:07, 1105.90 examples/s]7827 examples [00:07, 1106.48 examples/s]7938 examples [00:07, 1093.50 examples/s]8048 examples [00:07, 1046.31 examples/s]8159 examples [00:07, 1063.55 examples/s]8270 examples [00:08, 1075.11 examples/s]8381 examples [00:08, 1081.38 examples/s]8490 examples [00:08, 1082.76 examples/s]8599 examples [00:08, 1082.02 examples/s]8709 examples [00:08, 1085.98 examples/s]8819 examples [00:08, 1087.61 examples/s]8929 examples [00:08, 1090.59 examples/s]9039 examples [00:08, 1092.69 examples/s]9149 examples [00:08, 1093.36 examples/s]9259 examples [00:08, 1079.96 examples/s]9370 examples [00:09, 1088.15 examples/s]9479 examples [00:09, 1061.81 examples/s]9591 examples [00:09, 1076.02 examples/s]9701 examples [00:09, 1082.90 examples/s]9812 examples [00:09, 1089.42 examples/s]9922 examples [00:09, 1091.77 examples/s]10032 examples [00:09, 1042.41 examples/s]10141 examples [00:09, 1055.29 examples/s]10250 examples [00:09, 1063.90 examples/s]10361 examples [00:09, 1077.27 examples/s]10473 examples [00:10, 1088.83 examples/s]10584 examples [00:10, 1094.43 examples/s]10695 examples [00:10, 1096.88 examples/s]10805 examples [00:10, 1091.94 examples/s]10915 examples [00:10, 1092.32 examples/s]11025 examples [00:10, 1082.72 examples/s]11135 examples [00:10, 1087.51 examples/s]11244 examples [00:10, 1086.14 examples/s]11353 examples [00:10, 1083.83 examples/s]11462 examples [00:10, 1082.52 examples/s]11571 examples [00:11, 1066.68 examples/s]11679 examples [00:11, 1068.46 examples/s]11789 examples [00:11, 1076.68 examples/s]11897 examples [00:11, 1063.87 examples/s]12008 examples [00:11, 1074.62 examples/s]12118 examples [00:11, 1080.21 examples/s]12229 examples [00:11, 1087.92 examples/s]12338 examples [00:11, 1086.03 examples/s]12447 examples [00:11, 1064.85 examples/s]12557 examples [00:12, 1074.57 examples/s]12668 examples [00:12, 1084.22 examples/s]12777 examples [00:12, 1043.72 examples/s]12886 examples [00:12, 1055.46 examples/s]12993 examples [00:12, 1059.13 examples/s]13101 examples [00:12, 1063.64 examples/s]13209 examples [00:12, 1068.31 examples/s]13316 examples [00:12, 1060.47 examples/s]13426 examples [00:12, 1071.67 examples/s]13534 examples [00:12, 1074.02 examples/s]13643 examples [00:13, 1076.12 examples/s]13751 examples [00:13, 1073.80 examples/s]13859 examples [00:13, 1062.66 examples/s]13969 examples [00:13, 1071.76 examples/s]14077 examples [00:13, 1070.95 examples/s]14185 examples [00:13, 1072.43 examples/s]14297 examples [00:13, 1084.41 examples/s]14408 examples [00:13, 1090.67 examples/s]14522 examples [00:13, 1102.17 examples/s]14634 examples [00:13, 1105.00 examples/s]14745 examples [00:14, 1104.67 examples/s]14856 examples [00:14, 1105.11 examples/s]14967 examples [00:14, 1101.74 examples/s]15078 examples [00:14, 1102.87 examples/s]15189 examples [00:14, 1097.20 examples/s]15299 examples [00:14, 1095.71 examples/s]15409 examples [00:14, 1096.24 examples/s]15520 examples [00:14, 1099.54 examples/s]15630 examples [00:14, 1098.68 examples/s]15740 examples [00:14, 1085.99 examples/s]15849 examples [00:15, 1086.88 examples/s]15958 examples [00:15, 1084.15 examples/s]16067 examples [00:15, 1047.15 examples/s]16176 examples [00:15, 1057.88 examples/s]16286 examples [00:15, 1067.38 examples/s]16397 examples [00:15, 1078.00 examples/s]16508 examples [00:15, 1084.79 examples/s]16619 examples [00:15, 1091.14 examples/s]16729 examples [00:15, 1068.91 examples/s]16837 examples [00:15, 1040.08 examples/s]16942 examples [00:16, 1034.47 examples/s]17051 examples [00:16, 1048.77 examples/s]17162 examples [00:16, 1065.44 examples/s]17273 examples [00:16, 1076.32 examples/s]17382 examples [00:16, 1079.15 examples/s]17492 examples [00:16, 1085.21 examples/s]17603 examples [00:16, 1091.30 examples/s]17714 examples [00:16, 1096.27 examples/s]17826 examples [00:16, 1101.85 examples/s]17937 examples [00:16, 1099.79 examples/s]18049 examples [00:17, 1104.17 examples/s]18160 examples [00:17, 1097.18 examples/s]18270 examples [00:17, 1094.37 examples/s]18380 examples [00:17, 1089.04 examples/s]18489 examples [00:17, 1047.00 examples/s]18599 examples [00:17, 1061.07 examples/s]18706 examples [00:17, 1032.97 examples/s]18811 examples [00:17, 1037.52 examples/s]18916 examples [00:17, 1028.59 examples/s]19020 examples [00:18, 1008.50 examples/s]19131 examples [00:18, 1036.80 examples/s]19242 examples [00:18, 1055.11 examples/s]19348 examples [00:18, 1032.54 examples/s]19460 examples [00:18, 1057.15 examples/s]19570 examples [00:18, 1066.83 examples/s]19680 examples [00:18, 1075.50 examples/s]19788 examples [00:18, 1073.77 examples/s]19896 examples [00:18, 1071.33 examples/s]20004 examples [00:18, 1016.36 examples/s]20108 examples [00:19, 1022.60 examples/s]20217 examples [00:19, 1040.76 examples/s]20328 examples [00:19, 1058.42 examples/s]20441 examples [00:19, 1076.37 examples/s]20553 examples [00:19, 1087.76 examples/s]20663 examples [00:19, 1087.60 examples/s]20772 examples [00:19, 1044.77 examples/s]20883 examples [00:19, 1062.83 examples/s]20991 examples [00:19, 1066.09 examples/s]21103 examples [00:19, 1080.38 examples/s]21212 examples [00:20, 1077.46 examples/s]21323 examples [00:20, 1084.37 examples/s]21434 examples [00:20, 1089.82 examples/s]21544 examples [00:20, 1092.76 examples/s]21654 examples [00:20, 1090.31 examples/s]21764 examples [00:20, 1062.76 examples/s]21871 examples [00:20, 1064.09 examples/s]21978 examples [00:20, 1053.93 examples/s]22088 examples [00:20, 1064.71 examples/s]22198 examples [00:20, 1073.42 examples/s]22306 examples [00:21, 1049.19 examples/s]22414 examples [00:21, 1056.90 examples/s]22523 examples [00:21, 1064.66 examples/s]22630 examples [00:21, 1038.89 examples/s]22740 examples [00:21, 1055.15 examples/s]22846 examples [00:21, 1054.92 examples/s]22957 examples [00:21, 1070.15 examples/s]23065 examples [00:21, 1058.87 examples/s]23177 examples [00:21, 1075.88 examples/s]23289 examples [00:22, 1087.30 examples/s]23400 examples [00:22, 1093.91 examples/s]23510 examples [00:22, 1089.59 examples/s]23620 examples [00:22, 1089.92 examples/s]23730 examples [00:22, 1066.26 examples/s]23839 examples [00:22, 1071.32 examples/s]23947 examples [00:22, 1071.63 examples/s]24055 examples [00:22, 1071.26 examples/s]24166 examples [00:22, 1081.25 examples/s]24275 examples [00:22, 1080.92 examples/s]24384 examples [00:23, 1082.68 examples/s]24493 examples [00:23, 1082.92 examples/s]24602 examples [00:23, 1054.23 examples/s]24708 examples [00:23, 1034.78 examples/s]24815 examples [00:23, 1044.90 examples/s]24923 examples [00:23, 1054.19 examples/s]25032 examples [00:23, 1062.23 examples/s]25140 examples [00:23, 1067.11 examples/s]25248 examples [00:23, 1070.73 examples/s]25356 examples [00:23, 1071.01 examples/s]25464 examples [00:24, 1073.41 examples/s]25572 examples [00:24, 1069.64 examples/s]25679 examples [00:24, 1030.07 examples/s]25783 examples [00:24, 1030.78 examples/s]25889 examples [00:24, 1036.47 examples/s]25996 examples [00:24, 1044.77 examples/s]26105 examples [00:24, 1056.26 examples/s]26214 examples [00:24, 1065.65 examples/s]26322 examples [00:24, 1067.54 examples/s]26432 examples [00:24, 1074.49 examples/s]26541 examples [00:25, 1076.50 examples/s]26649 examples [00:25, 1075.95 examples/s]26757 examples [00:25, 1076.21 examples/s]26868 examples [00:25, 1085.03 examples/s]26977 examples [00:25, 1074.73 examples/s]27085 examples [00:25, 1074.61 examples/s]27193 examples [00:25, 1075.14 examples/s]27301 examples [00:25, 1071.91 examples/s]27412 examples [00:25, 1080.51 examples/s]27521 examples [00:25, 1077.77 examples/s]27629 examples [00:26, 1049.11 examples/s]27737 examples [00:26, 1056.25 examples/s]27847 examples [00:26, 1066.35 examples/s]27957 examples [00:26, 1075.71 examples/s]28068 examples [00:26, 1084.45 examples/s]28177 examples [00:26, 1076.65 examples/s]28287 examples [00:26, 1081.91 examples/s]28396 examples [00:26, 1083.01 examples/s]28507 examples [00:26, 1090.09 examples/s]28617 examples [00:27, 1071.15 examples/s]28725 examples [00:27, 1072.35 examples/s]28833 examples [00:27, 1058.75 examples/s]28941 examples [00:27, 1064.32 examples/s]29050 examples [00:27, 1071.77 examples/s]29158 examples [00:27, 1054.86 examples/s]29264 examples [00:27, 1048.51 examples/s]29369 examples [00:27, 1035.41 examples/s]29478 examples [00:27, 1050.42 examples/s]29585 examples [00:27, 1055.24 examples/s]29696 examples [00:28, 1068.88 examples/s]29805 examples [00:28, 1073.23 examples/s]29913 examples [00:28, 1061.99 examples/s]30020 examples [00:28, 1002.94 examples/s]30130 examples [00:28, 1028.14 examples/s]30235 examples [00:28, 1034.47 examples/s]30344 examples [00:28, 1049.06 examples/s]30453 examples [00:28, 1059.52 examples/s]30563 examples [00:28, 1071.12 examples/s]30671 examples [00:28, 1071.96 examples/s]30781 examples [00:29, 1079.92 examples/s]30891 examples [00:29, 1084.13 examples/s]31000 examples [00:29, 1084.72 examples/s]31110 examples [00:29, 1087.54 examples/s]31220 examples [00:29, 1090.36 examples/s]31330 examples [00:29, 1092.00 examples/s]31440 examples [00:29, 1080.88 examples/s]31549 examples [00:29, 1079.72 examples/s]31658 examples [00:29, 1080.22 examples/s]31769 examples [00:29, 1088.04 examples/s]31881 examples [00:30, 1095.28 examples/s]31991 examples [00:30, 1094.54 examples/s]32101 examples [00:30, 1093.86 examples/s]32211 examples [00:30, 1075.94 examples/s]32319 examples [00:30, 1075.91 examples/s]32429 examples [00:30, 1082.15 examples/s]32538 examples [00:30, 1079.79 examples/s]32647 examples [00:30, 1050.66 examples/s]32758 examples [00:30, 1066.72 examples/s]32870 examples [00:30, 1081.83 examples/s]32981 examples [00:31, 1088.03 examples/s]33090 examples [00:31, 1086.85 examples/s]33200 examples [00:31, 1088.24 examples/s]33309 examples [00:31, 1062.00 examples/s]33420 examples [00:31, 1074.17 examples/s]33528 examples [00:31, 1075.73 examples/s]33636 examples [00:31, 1068.25 examples/s]33749 examples [00:31, 1084.80 examples/s]33861 examples [00:31, 1093.97 examples/s]33971 examples [00:32, 1085.71 examples/s]34084 examples [00:32, 1097.45 examples/s]34194 examples [00:32, 1082.31 examples/s]34307 examples [00:32, 1093.42 examples/s]34417 examples [00:32, 1044.91 examples/s]34523 examples [00:32, 1046.21 examples/s]34630 examples [00:32, 1052.44 examples/s]34736 examples [00:32, 1023.87 examples/s]34847 examples [00:32, 1046.52 examples/s]34957 examples [00:32, 1059.89 examples/s]35066 examples [00:33, 1068.08 examples/s]35178 examples [00:33, 1083.09 examples/s]35287 examples [00:33, 1078.42 examples/s]35397 examples [00:33, 1082.76 examples/s]35508 examples [00:33, 1090.15 examples/s]35618 examples [00:33, 1092.29 examples/s]35728 examples [00:33, 1093.66 examples/s]35838 examples [00:33, 1075.22 examples/s]35946 examples [00:33, 1055.73 examples/s]36053 examples [00:33, 1059.10 examples/s]36165 examples [00:34, 1075.77 examples/s]36278 examples [00:34, 1090.63 examples/s]36388 examples [00:34, 1088.25 examples/s]36497 examples [00:34, 1087.02 examples/s]36608 examples [00:34, 1092.15 examples/s]36720 examples [00:34, 1098.29 examples/s]36830 examples [00:34, 1086.23 examples/s]36939 examples [00:34, 1078.44 examples/s]37048 examples [00:34, 1079.18 examples/s]37157 examples [00:34, 1081.59 examples/s]37266 examples [00:35, 1083.59 examples/s]37378 examples [00:35, 1091.80 examples/s]37489 examples [00:35, 1095.06 examples/s]37599 examples [00:35, 1091.03 examples/s]37709 examples [00:35, 1083.90 examples/s]37818 examples [00:35, 1084.72 examples/s]37927 examples [00:35, 1064.18 examples/s]38036 examples [00:35, 1069.04 examples/s]38143 examples [00:35, 1067.78 examples/s]38252 examples [00:35, 1072.31 examples/s]38360 examples [00:36, 1062.26 examples/s]38469 examples [00:36, 1069.13 examples/s]38578 examples [00:36, 1073.92 examples/s]38687 examples [00:36, 1075.91 examples/s]38795 examples [00:36, 1056.88 examples/s]38905 examples [00:36, 1069.17 examples/s]39015 examples [00:36, 1076.20 examples/s]39123 examples [00:36, 1062.64 examples/s]39230 examples [00:36, 1035.48 examples/s]39340 examples [00:37, 1053.06 examples/s]39450 examples [00:37, 1064.75 examples/s]39559 examples [00:37, 1070.60 examples/s]39667 examples [00:37, 1069.52 examples/s]39775 examples [00:37, 1061.86 examples/s]39884 examples [00:37, 1070.11 examples/s]39994 examples [00:37, 1077.72 examples/s]40102 examples [00:37, 1014.47 examples/s]40211 examples [00:37, 1034.67 examples/s]40324 examples [00:37, 1059.35 examples/s]40431 examples [00:38, 1056.60 examples/s]40540 examples [00:38, 1065.49 examples/s]40651 examples [00:38, 1075.97 examples/s]40759 examples [00:38, 1064.85 examples/s]40869 examples [00:38, 1074.32 examples/s]40978 examples [00:38, 1076.90 examples/s]41089 examples [00:38, 1085.39 examples/s]41202 examples [00:38, 1095.85 examples/s]41312 examples [00:38, 1088.28 examples/s]41421 examples [00:38, 1084.86 examples/s]41530 examples [00:39, 1077.48 examples/s]41639 examples [00:39, 1079.08 examples/s]41750 examples [00:39, 1085.45 examples/s]41859 examples [00:39, 1079.96 examples/s]41968 examples [00:39, 1070.38 examples/s]42076 examples [00:39, 1070.58 examples/s]42184 examples [00:39, 1053.61 examples/s]42292 examples [00:39, 1060.01 examples/s]42399 examples [00:39, 1001.38 examples/s]42507 examples [00:39, 1022.63 examples/s]42617 examples [00:40, 1043.55 examples/s]42725 examples [00:40, 1053.85 examples/s]42831 examples [00:40, 993.34 examples/s] 42938 examples [00:40, 1015.08 examples/s]43045 examples [00:40, 1028.28 examples/s]43153 examples [00:40, 1041.33 examples/s]43258 examples [00:40, 1017.65 examples/s]43361 examples [00:40, 1007.20 examples/s]43463 examples [00:40, 937.41 examples/s] 43563 examples [00:41, 953.92 examples/s]43668 examples [00:41, 978.56 examples/s]43770 examples [00:41, 988.23 examples/s]43870 examples [00:41, 973.58 examples/s]43968 examples [00:41, 952.83 examples/s]44064 examples [00:41, 949.93 examples/s]44170 examples [00:41, 978.88 examples/s]44280 examples [00:41, 1011.20 examples/s]44384 examples [00:41, 1018.20 examples/s]44487 examples [00:41, 1010.42 examples/s]44589 examples [00:42, 993.79 examples/s] 44699 examples [00:42, 1022.47 examples/s]44809 examples [00:42, 1042.53 examples/s]44919 examples [00:42, 1057.89 examples/s]45028 examples [00:42, 1066.52 examples/s]45135 examples [00:42, 1048.70 examples/s]45244 examples [00:42, 1059.60 examples/s]45351 examples [00:42, 1056.56 examples/s]45457 examples [00:42, 1055.81 examples/s]45563 examples [00:43, 1020.65 examples/s]45674 examples [00:43, 1044.97 examples/s]45782 examples [00:43, 1054.94 examples/s]45892 examples [00:43, 1065.51 examples/s]45999 examples [00:43, 1058.54 examples/s]46106 examples [00:43, 1036.60 examples/s]46210 examples [00:43, 1033.09 examples/s]46323 examples [00:43, 1058.04 examples/s]46435 examples [00:43, 1075.80 examples/s]46547 examples [00:43, 1086.67 examples/s]46659 examples [00:44, 1094.08 examples/s]46771 examples [00:44, 1099.93 examples/s]46882 examples [00:44, 1086.49 examples/s]46992 examples [00:44, 1088.26 examples/s]47102 examples [00:44, 1091.04 examples/s]47213 examples [00:44, 1096.62 examples/s]47324 examples [00:44, 1099.45 examples/s]47434 examples [00:44, 1098.51 examples/s]47546 examples [00:44, 1102.52 examples/s]47657 examples [00:44, 1103.17 examples/s]47768 examples [00:45, 1097.13 examples/s]47881 examples [00:45, 1104.81 examples/s]47992 examples [00:45, 1105.94 examples/s]48104 examples [00:45, 1109.88 examples/s]48216 examples [00:45, 1112.55 examples/s]48328 examples [00:45, 1093.14 examples/s]48438 examples [00:45, 1081.88 examples/s]48548 examples [00:45, 1084.64 examples/s]48657 examples [00:45, 1069.29 examples/s]48766 examples [00:45, 1074.14 examples/s]48874 examples [00:46, 1034.68 examples/s]48978 examples [00:46, 1032.71 examples/s]49082 examples [00:46, 1027.08 examples/s]49191 examples [00:46, 1044.14 examples/s]49296 examples [00:46, 1036.25 examples/s]49405 examples [00:46, 1050.51 examples/s]49517 examples [00:46, 1068.01 examples/s]49627 examples [00:46, 1076.00 examples/s]49738 examples [00:46, 1084.55 examples/s]49847 examples [00:46, 1084.63 examples/s]49956 examples [00:47, 1084.06 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7636/50000 [00:00<00:00, 76359.34 examples/s] 42%|████▏     | 21216/50000 [00:00<00:00, 87901.39 examples/s] 68%|██████▊   | 34152/50000 [00:00<00:00, 97250.79 examples/s] 95%|█████████▌| 47512/50000 [00:00<00:00, 105892.54 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 864.02 examples/s]201 examples [00:00, 930.97 examples/s]307 examples [00:00, 965.80 examples/s]417 examples [00:00, 1002.34 examples/s]526 examples [00:00, 1025.86 examples/s]639 examples [00:00, 1053.29 examples/s]736 examples [00:00, 836.85 examples/s] 847 examples [00:00, 902.15 examples/s]955 examples [00:00, 947.22 examples/s]1066 examples [00:01, 989.26 examples/s]1177 examples [00:01, 1021.79 examples/s]1286 examples [00:01, 1039.27 examples/s]1396 examples [00:01, 1056.32 examples/s]1505 examples [00:01, 1063.58 examples/s]1612 examples [00:01, 1058.81 examples/s]1724 examples [00:01, 1073.67 examples/s]1837 examples [00:01, 1088.14 examples/s]1949 examples [00:01, 1095.95 examples/s]2060 examples [00:01, 1097.35 examples/s]2170 examples [00:02, 1092.36 examples/s]2280 examples [00:02, 1090.32 examples/s]2391 examples [00:02, 1093.99 examples/s]2504 examples [00:02, 1102.15 examples/s]2616 examples [00:02, 1105.68 examples/s]2729 examples [00:02, 1112.12 examples/s]2843 examples [00:02, 1120.31 examples/s]2956 examples [00:02, 1121.04 examples/s]3069 examples [00:02, 1111.93 examples/s]3181 examples [00:02, 1111.14 examples/s]3293 examples [00:03, 1108.74 examples/s]3405 examples [00:03, 1109.54 examples/s]3518 examples [00:03, 1113.24 examples/s]3630 examples [00:03, 1114.87 examples/s]3742 examples [00:03, 1110.69 examples/s]3854 examples [00:03, 1098.31 examples/s]3965 examples [00:03, 1101.76 examples/s]4079 examples [00:03, 1110.21 examples/s]4191 examples [00:03, 1111.62 examples/s]4303 examples [00:04, 1101.93 examples/s]4414 examples [00:04, 1100.85 examples/s]4526 examples [00:04, 1106.22 examples/s]4639 examples [00:04, 1112.63 examples/s]4751 examples [00:04, 1109.07 examples/s]4862 examples [00:04, 1098.59 examples/s]4972 examples [00:04, 1060.59 examples/s]5080 examples [00:04, 1065.68 examples/s]5190 examples [00:04, 1075.44 examples/s]5298 examples [00:04, 1075.81 examples/s]5406 examples [00:05, 1069.08 examples/s]5516 examples [00:05, 1076.35 examples/s]5624 examples [00:05, 1066.26 examples/s]5734 examples [00:05, 1074.13 examples/s]5845 examples [00:05, 1083.91 examples/s]5954 examples [00:05, 1084.84 examples/s]6066 examples [00:05, 1093.95 examples/s]6176 examples [00:05, 1093.54 examples/s]6286 examples [00:05, 1093.74 examples/s]6398 examples [00:05, 1099.71 examples/s]6508 examples [00:06, 1084.73 examples/s]6621 examples [00:06, 1095.39 examples/s]6734 examples [00:06, 1104.55 examples/s]6846 examples [00:06, 1108.90 examples/s]6959 examples [00:06, 1113.88 examples/s]7071 examples [00:06, 1105.12 examples/s]7182 examples [00:06, 1101.57 examples/s]7293 examples [00:06, 1094.16 examples/s]7405 examples [00:06, 1101.25 examples/s]7518 examples [00:06, 1108.18 examples/s]7629 examples [00:07, 1103.03 examples/s]7740 examples [00:07, 1101.23 examples/s]7851 examples [00:07, 1098.08 examples/s]7961 examples [00:07, 1090.13 examples/s]8071 examples [00:07, 1089.13 examples/s]8180 examples [00:07, 1082.95 examples/s]8289 examples [00:07, 1051.76 examples/s]8402 examples [00:07, 1072.74 examples/s]8515 examples [00:07, 1088.73 examples/s]8629 examples [00:07, 1100.98 examples/s]8740 examples [00:08, 1098.14 examples/s]8853 examples [00:08, 1104.93 examples/s]8965 examples [00:08, 1108.75 examples/s]9080 examples [00:08, 1119.79 examples/s]9193 examples [00:08, 1120.76 examples/s]9306 examples [00:08, 1109.37 examples/s]9418 examples [00:08, 1111.14 examples/s]9532 examples [00:08, 1119.46 examples/s]9644 examples [00:08, 1112.72 examples/s]9756 examples [00:08, 1100.73 examples/s]9867 examples [00:09, 1089.92 examples/s]9977 examples [00:09, 1085.81 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUSGFS2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUSGFS2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f15648a68c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f15648a68c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f15648a68c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f15d0d20e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f15455b71d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f15648a68c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f15648a68c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f15648a68c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f154de5c940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1545b67d30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f153c0466a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f153c0466a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f153c0466a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f153c0467b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f153c0467b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f153c0467b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=dbf4f15981e4db48f1d09ba94dce5f0800ca6cdf08e0cdad5d5a5bcdd795277f
  Stored in directory: /tmp/pip-ephem-wheel-cache-47j0wypm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<28:48:02, 8.32kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:23:42, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<14:19:48, 16.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<10:02:14, 23.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<7:00:26, 34.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.35M/862M [00:01<4:52:25, 48.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<3:24:05, 69.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<2:22:03, 99.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:39:11, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:02<1:09:05, 202kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:02<48:26, 286kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:02<33:44, 408kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:02<23:36, 580kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<16:30, 824kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:03<11:45, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:05<10:06, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<08:55, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<06:42, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:07<07:13, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<06:39, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<04:58, 2.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<03:40, 3.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<13:07, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<11:54, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<08:54, 1.49MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<06:23, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<11:06, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<09:09, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<06:44, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<07:47, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:13<08:16, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<06:31, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<04:41, 2.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<21:46, 601kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<16:36, 788kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<11:54, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<11:21, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<09:17, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:17<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<07:50, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<05:03, 2.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<06:35, 1.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<04:29, 2.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<06:09, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<05:37, 2.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<04:12, 3.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<05:58, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<05:28, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<04:09, 3.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:25, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:07, 3.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:51, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:02, 3.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:47, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:21, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:04, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:47, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:03, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:46, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:35, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:09, 2.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:52, 3.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:57, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:28, 2.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:08, 2.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:45, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:39, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:19, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:50, 3.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<44:10, 276kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<32:11, 379kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<22:48, 534kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<18:41, 649kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:34, 779kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:30, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<09:59, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:15, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:05, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:01, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:24, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:47, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:58, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:01, 2.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<05:36, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:10, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:51, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:28, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:49, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:13, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:56, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:20, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:57, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:43, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<05:16, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<06:04, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:50, 2.39MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<05:14, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:52, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:39, 3.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:13, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:00, 1.90MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:47, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<04:50, 2.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:40, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:12, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<04:48, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:39, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:12, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<04:46, 2.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:34, 3.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:09, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:45, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<03:36, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:08, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:44, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<03:35, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:06, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:42, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:34, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:49, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:32, 2.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:19, 3.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<07:06, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:07, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:16, 3.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<5:58:05, 30.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<4:12:33, 42.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<2:57:04, 61.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<2:05:09, 85.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:28:46, 121kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<1:02:17, 172kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<45:43, 234kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<34:17, 311kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<24:27, 436kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<17:14, 617kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<16:13, 654kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<12:27, 851kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:58, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:44, 1.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:17, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:22, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:34, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<1:17:29, 135kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<55:16, 189kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<38:52, 269kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<29:33, 352kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<21:33, 482kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<15:18, 678kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<10:48, 957kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<49:13, 210kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<35:28, 291kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<25:00, 412kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<19:43, 520kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<15:45, 651kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<11:29, 891kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<09:42, 1.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:56, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:48, 1.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:14, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:27, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:56, 2.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:35, 2.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:25, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:16, 1.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:34, 1.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:01, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:44, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:54, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:28, 2.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:23, 2.92MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:38, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:15, 2.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:13, 3.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:10, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:31, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:09, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:06, 2.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:25, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:54, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<02:56, 3.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:11, 4.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<40:07, 239kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<29:02, 330kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<20:31, 465kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<16:32, 575kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<12:34, 756kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<09:01, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:30, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:56, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:05, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:44, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:58, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:43, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:46, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:15, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:05, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:58, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:47, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:41, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:29, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:22, 2.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:30, 2.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:01, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:54, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<02:56, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:24, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:00, 2.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:02, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:13, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:52, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:55, 3.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:08, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<03:48, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:53, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<03:46, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<02:51, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:49, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:42, 2.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:48, 3.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:00, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<03:38, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:37, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:43, 3.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:52, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:35, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<02:43, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:53, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:35, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:42, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:52, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:25, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:26, 2.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<02:32, 3.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:07, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:27, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:17, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:13, 1.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:48, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:50, 2.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:54, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:24, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:26, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:31, 3.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:55, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:16, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:11, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:05, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:34, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:33, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:35, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:18, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:16, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:35, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:23, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:02, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:29, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:41, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:22, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<02:33, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:17, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:29, 3.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:14, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:27, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:29, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:26, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:27, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:11, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:24, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:09, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<02:21, 3.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:23, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:21, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:21, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:05, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:20, 3.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:19, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:19, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:17, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:02, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:17, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:00, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<02:16, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:14, 3.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:11, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:56, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:13, 3.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<02:55, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:12, 3.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:08, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:31, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:45, 2.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:02, 3.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:45, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:19, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:29, 2.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:17, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:51, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:08, 3.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<01:35, 4.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<27:31, 238kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<19:56, 329kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<14:04, 464kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<11:19, 573kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<08:36, 754kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<06:08, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:48, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:43, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:27, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:54, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:23, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:31, 2.51MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:14, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:54, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:11, 2.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:59, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:22, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:40, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:37, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<01:59, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:48, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:34, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:57, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:33, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<01:54, 3.12MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:45, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:55, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<01:52, 3.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:41, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:28, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:51, 3.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:39, 2.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:26, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:50, 3.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:37, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:25, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:49, 3.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:33, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:56, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:19, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:20, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:46, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:52, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:17, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<01:39, 3.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<39:36, 136kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<28:14, 191kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<19:48, 270kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<15:01, 354kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<11:02, 481kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:48, 677kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<06:40, 788kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<05:12, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:43, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:50, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<03:12, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:51, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:31, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:53, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:30, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:16, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:42, 2.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:21, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:09, 2.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<01:38, 3.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:06, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<01:35, 3.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:15, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:04, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:33, 3.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:13, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:32, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:11, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:00, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<01:31, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:58, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<01:29, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:25, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:55, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:03, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:54, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:26, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:02, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:53, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:21, 3.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<11:28, 380kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<08:29, 513kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<06:00, 722kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:10, 829kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:03, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:55, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:02, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:32, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:52, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:17, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:27, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:22, 2.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<4:07:29, 16.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<2:53:56, 23.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<2:01:33, 33.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<1:24:32, 47.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<1:00:16, 66.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<42:27, 94.5kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<29:44, 134kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<20:38, 192kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<35:49, 110kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<25:52, 153kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<18:15, 216kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<12:39, 307kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<40:27, 96.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<28:40, 135kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<20:02, 192kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<14:47, 258kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<11:07, 343kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<07:55, 480kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<05:33, 678kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<05:27, 686kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:13, 887kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:01, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:57, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:26, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:47, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:05, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:10, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:42, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:44, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:35, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:11, 2.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:38, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:53, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:29, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:04, 3.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<25:01, 136kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<17:50, 190kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<12:28, 270kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<09:25, 354kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<06:55, 481kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:53, 675kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<04:09, 784kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<03:14, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:19, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:22, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:58, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:26, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:44, 1.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:32, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:08, 2.72MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:30, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:22, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:01, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:17, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<00:58, 3.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:22, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:15, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<00:56, 3.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:19, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:13, 2.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<00:55, 3.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:17, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:11, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<00:53, 3.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:15, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:09, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:51, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:13, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:23, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:05, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:47, 3.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:44, 938kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:08, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:39, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:40, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:16, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:55, 2.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:35, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:22, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:00, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:14, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:21, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:03, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:44, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<03:49, 602kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:52, 801kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:02, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<01:26, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<09:26, 237kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<07:03, 316kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<05:01, 442kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:47, 572kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:52, 751kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:02, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:53, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:30, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:05, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:13, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:04, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:47, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:00, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:54, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:39, 2.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:54, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:49, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:36, 3.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:51, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:47, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:35, 3.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:49, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:55, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:43, 2.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:46, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:42, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:32, 3.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:44, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:41, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:31, 3.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:49, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:37, 2.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:28, 3.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:39, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:35, 2.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:24, 3.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<09:58, 135kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<07:05, 189kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<04:54, 268kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<03:38, 351kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<02:48, 455kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<02:00, 629kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:32, 784kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<02:37, 459kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:51, 613kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:24, 801kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:59, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:54, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:44, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:31, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:35, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:31, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:22, 2.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:28, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:25, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:18, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:21, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:14, 3.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:39, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:32, 1.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:23, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:22, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:20, 1.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:15, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:11, 3.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:09, 3.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:06, 3.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:05, 3.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:03, 4.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<02:59, 82.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<02:06, 116kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<01:24, 164kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:50, 234kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:36, 292kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:25, 401kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:15, 564kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:09, 678kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 879kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.02MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 806/400000 [00:00<00:49, 8057.68it/s]  0%|          | 1657/400000 [00:00<00:48, 8187.87it/s]  1%|          | 2519/400000 [00:00<00:47, 8310.83it/s]  1%|          | 3363/400000 [00:00<00:47, 8348.36it/s]  1%|          | 4237/400000 [00:00<00:46, 8459.23it/s]  1%|▏         | 5103/400000 [00:00<00:46, 8515.97it/s]  1%|▏         | 5974/400000 [00:00<00:45, 8572.51it/s]  2%|▏         | 6792/400000 [00:00<00:46, 8447.90it/s]  2%|▏         | 7661/400000 [00:00<00:46, 8517.23it/s]  2%|▏         | 8518/400000 [00:01<00:45, 8532.73it/s]  2%|▏         | 9382/400000 [00:01<00:45, 8561.94it/s]  3%|▎         | 10237/400000 [00:01<00:45, 8557.06it/s]  3%|▎         | 11103/400000 [00:01<00:45, 8585.23it/s]  3%|▎         | 11954/400000 [00:01<00:45, 8552.98it/s]  3%|▎         | 12806/400000 [00:01<00:45, 8542.43it/s]  3%|▎         | 13673/400000 [00:01<00:45, 8579.28it/s]  4%|▎         | 14529/400000 [00:01<00:44, 8571.26it/s]  4%|▍         | 15385/400000 [00:01<00:45, 8507.83it/s]  4%|▍         | 16240/400000 [00:01<00:45, 8519.73it/s]  4%|▍         | 17116/400000 [00:02<00:44, 8588.79it/s]  4%|▍         | 17984/400000 [00:02<00:44, 8613.00it/s]  5%|▍         | 18846/400000 [00:02<00:44, 8544.72it/s]  5%|▍         | 19713/400000 [00:02<00:44, 8581.18it/s]  5%|▌         | 20572/400000 [00:02<00:44, 8581.44it/s]  5%|▌         | 21431/400000 [00:02<00:44, 8550.48it/s]  6%|▌         | 22315/400000 [00:02<00:43, 8634.15it/s]  6%|▌         | 23179/400000 [00:02<00:43, 8634.62it/s]  6%|▌         | 24058/400000 [00:02<00:43, 8677.96it/s]  6%|▌         | 24926/400000 [00:02<00:43, 8568.26it/s]  6%|▋         | 25799/400000 [00:03<00:43, 8614.63it/s]  7%|▋         | 26674/400000 [00:03<00:43, 8654.29it/s]  7%|▋         | 27540/400000 [00:03<00:43, 8505.55it/s]  7%|▋         | 28423/400000 [00:03<00:43, 8597.46it/s]  7%|▋         | 29284/400000 [00:03<00:43, 8443.10it/s]  8%|▊         | 30146/400000 [00:03<00:43, 8494.93it/s]  8%|▊         | 30997/400000 [00:03<00:43, 8427.90it/s]  8%|▊         | 31862/400000 [00:03<00:43, 8491.70it/s]  8%|▊         | 32737/400000 [00:03<00:42, 8567.42it/s]  8%|▊         | 33595/400000 [00:03<00:43, 8407.78it/s]  9%|▊         | 34437/400000 [00:04<00:43, 8341.09it/s]  9%|▉         | 35293/400000 [00:04<00:43, 8404.56it/s]  9%|▉         | 36174/400000 [00:04<00:42, 8521.49it/s]  9%|▉         | 37061/400000 [00:04<00:42, 8621.44it/s]  9%|▉         | 37935/400000 [00:04<00:41, 8655.39it/s] 10%|▉         | 38812/400000 [00:04<00:41, 8686.93it/s] 10%|▉         | 39694/400000 [00:04<00:41, 8723.71it/s] 10%|█         | 40567/400000 [00:04<00:41, 8704.55it/s] 10%|█         | 41441/400000 [00:04<00:41, 8713.21it/s] 11%|█         | 42313/400000 [00:04<00:41, 8691.18it/s] 11%|█         | 43195/400000 [00:05<00:40, 8728.57it/s] 11%|█         | 44071/400000 [00:05<00:40, 8735.41it/s] 11%|█         | 44945/400000 [00:05<00:40, 8730.19it/s] 11%|█▏        | 45819/400000 [00:05<00:40, 8707.23it/s] 12%|█▏        | 46690/400000 [00:05<00:40, 8676.63it/s] 12%|█▏        | 47563/400000 [00:05<00:40, 8689.73it/s] 12%|█▏        | 48442/400000 [00:05<00:40, 8717.94it/s] 12%|█▏        | 49314/400000 [00:05<00:40, 8698.55it/s] 13%|█▎        | 50189/400000 [00:05<00:40, 8713.12it/s] 13%|█▎        | 51061/400000 [00:05<00:40, 8684.37it/s] 13%|█▎        | 51945/400000 [00:06<00:39, 8728.49it/s] 13%|█▎        | 52820/400000 [00:06<00:39, 8733.01it/s] 13%|█▎        | 53694/400000 [00:06<00:39, 8669.08it/s] 14%|█▎        | 54562/400000 [00:06<00:40, 8573.89it/s] 14%|█▍        | 55420/400000 [00:06<00:40, 8573.02it/s] 14%|█▍        | 56294/400000 [00:06<00:39, 8621.72it/s] 14%|█▍        | 57157/400000 [00:06<00:39, 8593.27it/s] 15%|█▍        | 58017/400000 [00:06<00:39, 8558.14it/s] 15%|█▍        | 58904/400000 [00:06<00:39, 8647.79it/s] 15%|█▍        | 59772/400000 [00:06<00:39, 8656.74it/s] 15%|█▌        | 60647/400000 [00:07<00:39, 8684.26it/s] 15%|█▌        | 61525/400000 [00:07<00:38, 8712.08it/s] 16%|█▌        | 62397/400000 [00:07<00:39, 8584.63it/s] 16%|█▌        | 63261/400000 [00:07<00:39, 8599.78it/s] 16%|█▌        | 64122/400000 [00:07<00:39, 8527.41it/s] 16%|█▌        | 64987/400000 [00:07<00:39, 8561.37it/s] 16%|█▋        | 65850/400000 [00:07<00:38, 8580.02it/s] 17%|█▋        | 66709/400000 [00:07<00:38, 8556.96it/s] 17%|█▋        | 67570/400000 [00:07<00:38, 8572.60it/s] 17%|█▋        | 68428/400000 [00:07<00:38, 8526.08it/s] 17%|█▋        | 69288/400000 [00:08<00:38, 8545.65it/s] 18%|█▊        | 70151/400000 [00:08<00:38, 8568.10it/s] 18%|█▊        | 71013/400000 [00:08<00:38, 8583.49it/s] 18%|█▊        | 71886/400000 [00:08<00:38, 8625.41it/s] 18%|█▊        | 72749/400000 [00:08<00:39, 8309.75it/s] 18%|█▊        | 73608/400000 [00:08<00:38, 8390.30it/s] 19%|█▊        | 74474/400000 [00:08<00:38, 8467.61it/s] 19%|█▉        | 75330/400000 [00:08<00:38, 8494.61it/s] 19%|█▉        | 76187/400000 [00:08<00:38, 8515.28it/s] 19%|█▉        | 77044/400000 [00:08<00:37, 8530.18it/s] 19%|█▉        | 77914/400000 [00:09<00:37, 8579.52it/s] 20%|█▉        | 78783/400000 [00:09<00:37, 8611.64it/s] 20%|█▉        | 79645/400000 [00:09<00:37, 8613.91it/s] 20%|██        | 80507/400000 [00:09<00:37, 8476.02it/s] 20%|██        | 81377/400000 [00:09<00:37, 8540.46it/s] 21%|██        | 82260/400000 [00:09<00:36, 8623.23it/s] 21%|██        | 83138/400000 [00:09<00:36, 8668.17it/s] 21%|██        | 84006/400000 [00:09<00:36, 8633.23it/s] 21%|██        | 84886/400000 [00:09<00:36, 8679.79it/s] 21%|██▏       | 85755/400000 [00:09<00:36, 8669.25it/s] 22%|██▏       | 86633/400000 [00:10<00:36, 8700.26it/s] 22%|██▏       | 87512/400000 [00:10<00:35, 8725.36it/s] 22%|██▏       | 88385/400000 [00:10<00:35, 8680.66it/s] 22%|██▏       | 89261/400000 [00:10<00:35, 8701.45it/s] 23%|██▎       | 90132/400000 [00:10<00:35, 8682.10it/s] 23%|██▎       | 91015/400000 [00:10<00:35, 8725.10it/s] 23%|██▎       | 91888/400000 [00:10<00:35, 8724.56it/s] 23%|██▎       | 92761/400000 [00:10<00:35, 8580.29it/s] 23%|██▎       | 93625/400000 [00:10<00:35, 8596.64it/s] 24%|██▎       | 94486/400000 [00:10<00:35, 8588.43it/s] 24%|██▍       | 95346/400000 [00:11<00:35, 8569.67it/s] 24%|██▍       | 96212/400000 [00:11<00:35, 8595.80it/s] 24%|██▍       | 97072/400000 [00:11<00:35, 8471.39it/s] 24%|██▍       | 97943/400000 [00:11<00:35, 8539.70it/s] 25%|██▍       | 98804/400000 [00:11<00:35, 8559.64it/s] 25%|██▍       | 99678/400000 [00:11<00:34, 8612.75it/s] 25%|██▌       | 100554/400000 [00:11<00:34, 8655.23it/s] 25%|██▌       | 101420/400000 [00:11<00:34, 8597.50it/s] 26%|██▌       | 102295/400000 [00:11<00:34, 8641.52it/s] 26%|██▌       | 103160/400000 [00:12<00:34, 8634.14it/s] 26%|██▌       | 104037/400000 [00:12<00:34, 8674.33it/s] 26%|██▌       | 104914/400000 [00:12<00:33, 8701.80it/s] 26%|██▋       | 105785/400000 [00:12<00:34, 8574.01it/s] 27%|██▋       | 106643/400000 [00:12<00:34, 8456.25it/s] 27%|██▋       | 107509/400000 [00:12<00:34, 8515.12it/s] 27%|██▋       | 108392/400000 [00:12<00:33, 8606.76it/s] 27%|██▋       | 109274/400000 [00:12<00:33, 8669.64it/s] 28%|██▊       | 110142/400000 [00:12<00:33, 8645.49it/s] 28%|██▊       | 111007/400000 [00:12<00:33, 8634.13it/s] 28%|██▊       | 111878/400000 [00:13<00:33, 8656.64it/s] 28%|██▊       | 112753/400000 [00:13<00:33, 8683.78it/s] 28%|██▊       | 113629/400000 [00:13<00:32, 8705.85it/s] 29%|██▊       | 114500/400000 [00:13<00:32, 8679.45it/s] 29%|██▉       | 115369/400000 [00:13<00:33, 8578.47it/s] 29%|██▉       | 116228/400000 [00:13<00:33, 8543.46it/s] 29%|██▉       | 117101/400000 [00:13<00:32, 8598.25it/s] 29%|██▉       | 117976/400000 [00:13<00:32, 8642.15it/s] 30%|██▉       | 118841/400000 [00:13<00:32, 8577.44it/s] 30%|██▉       | 119700/400000 [00:13<00:33, 8489.87it/s] 30%|███       | 120572/400000 [00:14<00:32, 8554.89it/s] 30%|███       | 121444/400000 [00:14<00:32, 8603.47it/s] 31%|███       | 122305/400000 [00:14<00:32, 8590.96it/s] 31%|███       | 123167/400000 [00:14<00:32, 8598.35it/s] 31%|███       | 124028/400000 [00:14<00:32, 8491.07it/s] 31%|███       | 124878/400000 [00:14<00:32, 8371.31it/s] 31%|███▏      | 125740/400000 [00:14<00:32, 8442.54it/s] 32%|███▏      | 126603/400000 [00:14<00:32, 8496.07it/s] 32%|███▏      | 127454/400000 [00:14<00:32, 8474.98it/s] 32%|███▏      | 128307/400000 [00:14<00:32, 8489.43it/s] 32%|███▏      | 129177/400000 [00:15<00:31, 8549.27it/s] 33%|███▎      | 130033/400000 [00:15<00:31, 8490.36it/s] 33%|███▎      | 130907/400000 [00:15<00:31, 8561.53it/s] 33%|███▎      | 131764/400000 [00:15<00:31, 8393.64it/s] 33%|███▎      | 132605/400000 [00:15<00:32, 8333.13it/s] 33%|███▎      | 133485/400000 [00:15<00:31, 8465.64it/s] 34%|███▎      | 134359/400000 [00:15<00:31, 8543.90it/s] 34%|███▍      | 135235/400000 [00:15<00:30, 8607.06it/s] 34%|███▍      | 136097/400000 [00:15<00:31, 8463.56it/s] 34%|███▍      | 136955/400000 [00:15<00:30, 8496.10it/s] 34%|███▍      | 137815/400000 [00:16<00:30, 8525.83it/s] 35%|███▍      | 138699/400000 [00:16<00:30, 8617.02it/s] 35%|███▍      | 139588/400000 [00:16<00:29, 8696.92it/s] 35%|███▌      | 140459/400000 [00:16<00:29, 8656.82it/s] 35%|███▌      | 141326/400000 [00:16<00:30, 8533.25it/s] 36%|███▌      | 142199/400000 [00:16<00:30, 8591.19it/s] 36%|███▌      | 143059/400000 [00:16<00:29, 8590.33it/s] 36%|███▌      | 143919/400000 [00:16<00:29, 8539.45it/s] 36%|███▌      | 144788/400000 [00:16<00:29, 8583.13it/s] 36%|███▋      | 145647/400000 [00:16<00:30, 8412.82it/s] 37%|███▋      | 146490/400000 [00:17<00:30, 8343.44it/s] 37%|███▋      | 147352/400000 [00:17<00:29, 8422.34it/s] 37%|███▋      | 148232/400000 [00:17<00:29, 8529.94it/s] 37%|███▋      | 149097/400000 [00:17<00:29, 8565.21it/s] 37%|███▋      | 149968/400000 [00:17<00:29, 8607.20it/s] 38%|███▊      | 150854/400000 [00:17<00:28, 8679.00it/s] 38%|███▊      | 151737/400000 [00:17<00:28, 8722.89it/s] 38%|███▊      | 152610/400000 [00:17<00:28, 8710.02it/s] 38%|███▊      | 153482/400000 [00:17<00:28, 8701.83it/s] 39%|███▊      | 154353/400000 [00:17<00:28, 8660.91it/s] 39%|███▉      | 155242/400000 [00:18<00:28, 8726.56it/s] 39%|███▉      | 156121/400000 [00:18<00:27, 8742.76it/s] 39%|███▉      | 157013/400000 [00:18<00:27, 8793.51it/s] 39%|███▉      | 157893/400000 [00:18<00:27, 8789.52it/s] 40%|███▉      | 158773/400000 [00:18<00:27, 8783.13it/s] 40%|███▉      | 159655/400000 [00:18<00:27, 8791.18it/s] 40%|████      | 160535/400000 [00:18<00:27, 8788.75it/s] 40%|████      | 161418/400000 [00:18<00:27, 8799.94it/s] 41%|████      | 162299/400000 [00:18<00:28, 8458.17it/s] 41%|████      | 163148/400000 [00:19<00:29, 8163.06it/s] 41%|████      | 163969/400000 [00:19<00:28, 8148.12it/s] 41%|████      | 164787/400000 [00:19<00:29, 8034.59it/s] 41%|████▏     | 165659/400000 [00:19<00:28, 8228.08it/s] 42%|████▏     | 166527/400000 [00:19<00:27, 8356.37it/s] 42%|████▏     | 167381/400000 [00:19<00:27, 8408.20it/s] 42%|████▏     | 168229/400000 [00:19<00:27, 8429.47it/s] 42%|████▏     | 169091/400000 [00:19<00:27, 8485.60it/s] 42%|████▏     | 169965/400000 [00:19<00:26, 8558.39it/s] 43%|████▎     | 170822/400000 [00:19<00:26, 8538.59it/s] 43%|████▎     | 171677/400000 [00:20<00:26, 8522.63it/s] 43%|████▎     | 172549/400000 [00:20<00:26, 8578.26it/s] 43%|████▎     | 173412/400000 [00:20<00:26, 8592.50it/s] 44%|████▎     | 174287/400000 [00:20<00:26, 8636.62it/s] 44%|████▍     | 175151/400000 [00:20<00:26, 8522.27it/s] 44%|████▍     | 176005/400000 [00:20<00:26, 8527.36it/s] 44%|████▍     | 176883/400000 [00:20<00:25, 8601.42it/s] 44%|████▍     | 177772/400000 [00:20<00:25, 8685.45it/s] 45%|████▍     | 178642/400000 [00:20<00:25, 8667.87it/s] 45%|████▍     | 179510/400000 [00:20<00:25, 8646.20it/s] 45%|████▌     | 180375/400000 [00:21<00:25, 8589.10it/s] 45%|████▌     | 181247/400000 [00:21<00:25, 8625.64it/s] 46%|████▌     | 182117/400000 [00:21<00:25, 8645.54it/s] 46%|████▌     | 182989/400000 [00:21<00:25, 8666.02it/s] 46%|████▌     | 183856/400000 [00:21<00:24, 8657.77it/s] 46%|████▌     | 184722/400000 [00:21<00:24, 8629.77it/s] 46%|████▋     | 185590/400000 [00:21<00:24, 8642.18it/s] 47%|████▋     | 186455/400000 [00:21<00:24, 8637.57it/s] 47%|████▋     | 187331/400000 [00:21<00:24, 8671.26it/s] 47%|████▋     | 188205/400000 [00:21<00:24, 8690.86it/s] 47%|████▋     | 189075/400000 [00:22<00:24, 8639.02it/s] 47%|████▋     | 189942/400000 [00:22<00:24, 8646.29it/s] 48%|████▊     | 190808/400000 [00:22<00:24, 8649.83it/s] 48%|████▊     | 191687/400000 [00:22<00:23, 8689.05it/s] 48%|████▊     | 192568/400000 [00:22<00:23, 8723.60it/s] 48%|████▊     | 193441/400000 [00:22<00:23, 8643.10it/s] 49%|████▊     | 194325/400000 [00:22<00:23, 8700.71it/s] 49%|████▉     | 195215/400000 [00:22<00:23, 8756.96it/s] 49%|████▉     | 196094/400000 [00:22<00:23, 8766.00it/s] 49%|████▉     | 196971/400000 [00:22<00:23, 8712.14it/s] 49%|████▉     | 197843/400000 [00:23<00:23, 8549.69it/s] 50%|████▉     | 198721/400000 [00:23<00:23, 8615.03it/s] 50%|████▉     | 199605/400000 [00:23<00:23, 8680.57it/s] 50%|█████     | 200488/400000 [00:23<00:22, 8723.62it/s] 50%|█████     | 201361/400000 [00:23<00:23, 8635.38it/s] 51%|█████     | 202229/400000 [00:23<00:22, 8647.91it/s] 51%|█████     | 203110/400000 [00:23<00:22, 8693.96it/s] 51%|█████     | 203992/400000 [00:23<00:22, 8729.10it/s] 51%|█████     | 204870/400000 [00:23<00:22, 8741.55it/s] 51%|█████▏    | 205745/400000 [00:23<00:22, 8694.62it/s] 52%|█████▏    | 206615/400000 [00:24<00:22, 8669.79it/s] 52%|█████▏    | 207491/400000 [00:24<00:22, 8695.54it/s] 52%|█████▏    | 208361/400000 [00:24<00:22, 8474.66it/s] 52%|█████▏    | 209233/400000 [00:24<00:22, 8545.26it/s] 53%|█████▎    | 210103/400000 [00:24<00:22, 8590.76it/s] 53%|█████▎    | 210963/400000 [00:24<00:22, 8571.03it/s] 53%|█████▎    | 211836/400000 [00:24<00:21, 8617.83it/s] 53%|█████▎    | 212707/400000 [00:24<00:21, 8642.85it/s] 53%|█████▎    | 213586/400000 [00:24<00:21, 8685.58it/s] 54%|█████▎    | 214455/400000 [00:24<00:21, 8639.83it/s] 54%|█████▍    | 215320/400000 [00:25<00:21, 8419.67it/s] 54%|█████▍    | 216189/400000 [00:25<00:21, 8498.72it/s] 54%|█████▍    | 217068/400000 [00:25<00:21, 8582.44it/s] 54%|█████▍    | 217945/400000 [00:25<00:21, 8637.58it/s] 55%|█████▍    | 218810/400000 [00:25<00:21, 8590.78it/s] 55%|█████▍    | 219670/400000 [00:25<00:20, 8590.76it/s] 55%|█████▌    | 220539/400000 [00:25<00:20, 8619.29it/s] 55%|█████▌    | 221402/400000 [00:25<00:20, 8610.24it/s] 56%|█████▌    | 222278/400000 [00:25<00:20, 8654.55it/s] 56%|█████▌    | 223144/400000 [00:25<00:20, 8627.08it/s] 56%|█████▌    | 224007/400000 [00:26<00:21, 8357.21it/s] 56%|█████▌    | 224888/400000 [00:26<00:20, 8486.17it/s] 56%|█████▋    | 225760/400000 [00:26<00:20, 8553.41it/s] 57%|█████▋    | 226626/400000 [00:26<00:20, 8583.89it/s] 57%|█████▋    | 227486/400000 [00:26<00:20, 8537.74it/s] 57%|█████▋    | 228341/400000 [00:26<00:20, 8440.91it/s] 57%|█████▋    | 229207/400000 [00:26<00:20, 8505.20it/s] 58%|█████▊    | 230074/400000 [00:26<00:19, 8552.51it/s] 58%|█████▊    | 230930/400000 [00:26<00:20, 8258.66it/s] 58%|█████▊    | 231791/400000 [00:27<00:20, 8360.99it/s] 58%|█████▊    | 232662/400000 [00:27<00:19, 8462.47it/s] 58%|█████▊    | 233552/400000 [00:27<00:19, 8586.38it/s] 59%|█████▊    | 234430/400000 [00:27<00:19, 8641.07it/s] 59%|█████▉    | 235296/400000 [00:27<00:19, 8508.56it/s] 59%|█████▉    | 236149/400000 [00:27<00:20, 8073.93it/s] 59%|█████▉    | 237034/400000 [00:27<00:19, 8290.38it/s] 59%|█████▉    | 237913/400000 [00:27<00:19, 8433.21it/s] 60%|█████▉    | 238802/400000 [00:27<00:18, 8564.76it/s] 60%|█████▉    | 239689/400000 [00:27<00:18, 8653.91it/s] 60%|██████    | 240558/400000 [00:28<00:18, 8623.07it/s] 60%|██████    | 241423/400000 [00:28<00:18, 8578.48it/s] 61%|██████    | 242283/400000 [00:28<00:18, 8421.44it/s] 61%|██████    | 243162/400000 [00:28<00:18, 8526.76it/s] 61%|██████    | 244041/400000 [00:28<00:18, 8601.50it/s] 61%|██████    | 244903/400000 [00:28<00:18, 8441.56it/s] 61%|██████▏   | 245749/400000 [00:28<00:18, 8401.19it/s] 62%|██████▏   | 246623/400000 [00:28<00:18, 8497.94it/s] 62%|██████▏   | 247502/400000 [00:28<00:17, 8583.42it/s] 62%|██████▏   | 248366/400000 [00:28<00:17, 8599.37it/s] 62%|██████▏   | 249230/400000 [00:29<00:17, 8609.77it/s] 63%|██████▎   | 250111/400000 [00:29<00:17, 8667.05it/s] 63%|██████▎   | 250991/400000 [00:29<00:17, 8706.08it/s] 63%|██████▎   | 251862/400000 [00:29<00:17, 8705.52it/s] 63%|██████▎   | 252733/400000 [00:29<00:16, 8670.49it/s] 63%|██████▎   | 253601/400000 [00:29<00:16, 8645.34it/s] 64%|██████▎   | 254478/400000 [00:29<00:16, 8680.73it/s] 64%|██████▍   | 255350/400000 [00:29<00:16, 8691.73it/s] 64%|██████▍   | 256220/400000 [00:29<00:16, 8673.10it/s] 64%|██████▍   | 257088/400000 [00:29<00:16, 8495.76it/s] 64%|██████▍   | 257944/400000 [00:30<00:16, 8514.45it/s] 65%|██████▍   | 258828/400000 [00:30<00:16, 8609.23it/s] 65%|██████▍   | 259716/400000 [00:30<00:16, 8687.89it/s] 65%|██████▌   | 260586/400000 [00:30<00:16, 8656.58it/s] 65%|██████▌   | 261453/400000 [00:30<00:16, 8636.77it/s] 66%|██████▌   | 262319/400000 [00:30<00:15, 8643.25it/s] 66%|██████▌   | 263209/400000 [00:30<00:15, 8715.93it/s] 66%|██████▌   | 264095/400000 [00:30<00:15, 8758.52it/s] 66%|██████▌   | 264972/400000 [00:30<00:15, 8745.39it/s] 66%|██████▋   | 265853/400000 [00:30<00:15, 8761.78it/s] 67%|██████▋   | 266730/400000 [00:31<00:15, 8712.26it/s] 67%|██████▋   | 267602/400000 [00:31<00:15, 8615.78it/s] 67%|██████▋   | 268481/400000 [00:31<00:15, 8666.19it/s] 67%|██████▋   | 269358/400000 [00:31<00:15, 8694.82it/s] 68%|██████▊   | 270228/400000 [00:31<00:15, 8592.07it/s] 68%|██████▊   | 271088/400000 [00:31<00:15, 8557.92it/s] 68%|██████▊   | 271968/400000 [00:31<00:14, 8627.03it/s] 68%|██████▊   | 272838/400000 [00:31<00:14, 8646.88it/s] 68%|██████▊   | 273720/400000 [00:31<00:14, 8696.81it/s] 69%|██████▊   | 274590/400000 [00:31<00:14, 8669.31it/s] 69%|██████▉   | 275458/400000 [00:32<00:14, 8627.80it/s] 69%|██████▉   | 276329/400000 [00:32<00:14, 8650.35it/s] 69%|██████▉   | 277202/400000 [00:32<00:14, 8671.89it/s] 70%|██████▉   | 278082/400000 [00:32<00:13, 8709.72it/s] 70%|██████▉   | 278954/400000 [00:32<00:13, 8697.59it/s] 70%|██████▉   | 279826/400000 [00:32<00:13, 8703.88it/s] 70%|███████   | 280698/400000 [00:32<00:13, 8708.21it/s] 70%|███████   | 281572/400000 [00:32<00:13, 8715.35it/s] 71%|███████   | 282444/400000 [00:32<00:13, 8609.93it/s] 71%|███████   | 283317/400000 [00:32<00:13, 8645.57it/s] 71%|███████   | 284182/400000 [00:33<00:13, 8610.53it/s] 71%|███████▏  | 285047/400000 [00:33<00:13, 8621.99it/s] 71%|███████▏  | 285932/400000 [00:33<00:13, 8686.85it/s] 72%|███████▏  | 286807/400000 [00:33<00:13, 8702.76it/s] 72%|███████▏  | 287681/400000 [00:33<00:12, 8712.75it/s] 72%|███████▏  | 288553/400000 [00:33<00:12, 8708.47it/s] 72%|███████▏  | 289432/400000 [00:33<00:12, 8731.67it/s] 73%|███████▎  | 290323/400000 [00:33<00:12, 8781.56it/s] 73%|███████▎  | 291202/400000 [00:33<00:12, 8720.75it/s] 73%|███████▎  | 292075/400000 [00:33<00:12, 8713.38it/s] 73%|███████▎  | 292947/400000 [00:34<00:12, 8531.10it/s] 73%|███████▎  | 293802/400000 [00:34<00:12, 8496.37it/s] 74%|███████▎  | 294668/400000 [00:34<00:12, 8544.28it/s] 74%|███████▍  | 295547/400000 [00:34<00:12, 8614.14it/s] 74%|███████▍  | 296426/400000 [00:34<00:11, 8665.70it/s] 74%|███████▍  | 297310/400000 [00:34<00:11, 8715.54it/s] 75%|███████▍  | 298182/400000 [00:34<00:11, 8645.92it/s] 75%|███████▍  | 299061/400000 [00:34<00:11, 8687.98it/s] 75%|███████▍  | 299931/400000 [00:34<00:11, 8664.78it/s] 75%|███████▌  | 300805/400000 [00:34<00:11, 8686.90it/s] 75%|███████▌  | 301680/400000 [00:35<00:11, 8704.59it/s] 76%|███████▌  | 302551/400000 [00:35<00:11, 8634.74it/s] 76%|███████▌  | 303424/400000 [00:35<00:11, 8660.58it/s] 76%|███████▌  | 304291/400000 [00:35<00:11, 8582.51it/s] 76%|███████▋  | 305168/400000 [00:35<00:10, 8635.16it/s] 77%|███████▋  | 306041/400000 [00:35<00:10, 8663.08it/s] 77%|███████▋  | 306908/400000 [00:35<00:10, 8633.16it/s] 77%|███████▋  | 307772/400000 [00:35<00:10, 8604.29it/s] 77%|███████▋  | 308633/400000 [00:35<00:10, 8536.73it/s] 77%|███████▋  | 309498/400000 [00:36<00:10, 8568.79it/s] 78%|███████▊  | 310366/400000 [00:36<00:10, 8599.62it/s] 78%|███████▊  | 311227/400000 [00:36<00:10, 8599.44it/s] 78%|███████▊  | 312108/400000 [00:36<00:10, 8659.05it/s] 78%|███████▊  | 312975/400000 [00:36<00:10, 8595.52it/s] 78%|███████▊  | 313855/400000 [00:36<00:09, 8655.21it/s] 79%|███████▊  | 314728/400000 [00:36<00:09, 8675.18it/s] 79%|███████▉  | 315596/400000 [00:36<00:09, 8636.66it/s] 79%|███████▉  | 316471/400000 [00:36<00:09, 8668.43it/s] 79%|███████▉  | 317338/400000 [00:36<00:09, 8590.07it/s] 80%|███████▉  | 318223/400000 [00:37<00:09, 8665.80it/s] 80%|███████▉  | 319090/400000 [00:37<00:09, 8647.20it/s] 80%|███████▉  | 319955/400000 [00:37<00:09, 8506.39it/s] 80%|████████  | 320815/400000 [00:37<00:09, 8532.87it/s] 80%|████████  | 321684/400000 [00:37<00:09, 8578.82it/s] 81%|████████  | 322565/400000 [00:37<00:08, 8646.35it/s] 81%|████████  | 323446/400000 [00:37<00:08, 8692.79it/s] 81%|████████  | 324323/400000 [00:37<00:08, 8715.36it/s] 81%|████████▏ | 325195/400000 [00:37<00:08, 8688.95it/s] 82%|████████▏ | 326070/400000 [00:37<00:08, 8704.34it/s] 82%|████████▏ | 326951/400000 [00:38<00:08, 8733.79it/s] 82%|████████▏ | 327825/400000 [00:38<00:08, 8727.24it/s] 82%|████████▏ | 328703/400000 [00:38<00:08, 8742.79it/s] 82%|████████▏ | 329578/400000 [00:38<00:08, 8675.57it/s] 83%|████████▎ | 330446/400000 [00:38<00:08, 8483.66it/s] 83%|████████▎ | 331298/400000 [00:38<00:08, 8491.59it/s] 83%|████████▎ | 332173/400000 [00:38<00:07, 8567.17it/s] 83%|████████▎ | 333031/400000 [00:38<00:07, 8381.08it/s] 83%|████████▎ | 333890/400000 [00:38<00:07, 8440.98it/s] 84%|████████▎ | 334755/400000 [00:38<00:07, 8472.85it/s] 84%|████████▍ | 335631/400000 [00:39<00:07, 8556.24it/s] 84%|████████▍ | 336505/400000 [00:39<00:07, 8608.21it/s] 84%|████████▍ | 337384/400000 [00:39<00:07, 8659.39it/s] 85%|████████▍ | 338251/400000 [00:39<00:07, 8626.62it/s] 85%|████████▍ | 339137/400000 [00:39<00:07, 8694.28it/s] 85%|████████▌ | 340023/400000 [00:39<00:06, 8741.92it/s] 85%|████████▌ | 340909/400000 [00:39<00:06, 8773.72it/s] 85%|████████▌ | 341787/400000 [00:39<00:06, 8764.79it/s] 86%|████████▌ | 342664/400000 [00:39<00:06, 8716.57it/s] 86%|████████▌ | 343547/400000 [00:39<00:06, 8748.72it/s] 86%|████████▌ | 344439/400000 [00:40<00:06, 8798.65it/s] 86%|████████▋ | 345320/400000 [00:40<00:06, 8799.16it/s] 87%|████████▋ | 346209/400000 [00:40<00:06, 8823.47it/s] 87%|████████▋ | 347092/400000 [00:40<00:06, 8696.46it/s] 87%|████████▋ | 347967/400000 [00:40<00:05, 8710.00it/s] 87%|████████▋ | 348855/400000 [00:40<00:05, 8757.86it/s] 87%|████████▋ | 349741/400000 [00:40<00:05, 8786.08it/s] 88%|████████▊ | 350620/400000 [00:40<00:05, 8712.62it/s] 88%|████████▊ | 351492/400000 [00:40<00:05, 8588.49it/s] 88%|████████▊ | 352377/400000 [00:40<00:05, 8663.70it/s] 88%|████████▊ | 353273/400000 [00:41<00:05, 8748.62it/s] 89%|████████▊ | 354149/400000 [00:41<00:05, 8743.76it/s] 89%|████████▉ | 355025/400000 [00:41<00:05, 8746.96it/s] 89%|████████▉ | 355900/400000 [00:41<00:05, 8734.27it/s] 89%|████████▉ | 356785/400000 [00:41<00:04, 8767.31it/s] 89%|████████▉ | 357662/400000 [00:41<00:04, 8688.01it/s] 90%|████████▉ | 358544/400000 [00:41<00:04, 8725.31it/s] 90%|████████▉ | 359418/400000 [00:41<00:04, 8727.99it/s] 90%|█████████ | 360291/400000 [00:41<00:04, 8719.74it/s] 90%|█████████ | 361167/400000 [00:41<00:04, 8731.38it/s] 91%|█████████ | 362041/400000 [00:42<00:04, 8724.04it/s] 91%|█████████ | 362916/400000 [00:42<00:04, 8730.91it/s] 91%|█████████ | 363790/400000 [00:42<00:04, 8669.35it/s] 91%|█████████ | 364658/400000 [00:42<00:04, 8658.27it/s] 91%|█████████▏| 365535/400000 [00:42<00:03, 8690.11it/s] 92%|█████████▏| 366429/400000 [00:42<00:03, 8762.20it/s] 92%|█████████▏| 367316/400000 [00:42<00:03, 8792.04it/s] 92%|█████████▏| 368196/400000 [00:42<00:03, 8776.26it/s] 92%|█████████▏| 369074/400000 [00:42<00:03, 8455.18it/s] 92%|█████████▏| 369952/400000 [00:42<00:03, 8548.50it/s] 93%|█████████▎| 370841/400000 [00:43<00:03, 8647.87it/s] 93%|█████████▎| 371722/400000 [00:43<00:03, 8693.64it/s] 93%|█████████▎| 372593/400000 [00:43<00:03, 8662.25it/s] 93%|█████████▎| 373461/400000 [00:43<00:03, 8657.07it/s] 94%|█████████▎| 374328/400000 [00:43<00:02, 8652.30it/s] 94%|█████████▍| 375211/400000 [00:43<00:02, 8703.23it/s] 94%|█████████▍| 376087/400000 [00:43<00:02, 8720.00it/s] 94%|█████████▍| 376960/400000 [00:43<00:02, 8670.60it/s] 94%|█████████▍| 377828/400000 [00:43<00:02, 8639.34it/s] 95%|█████████▍| 378704/400000 [00:43<00:02, 8675.08it/s] 95%|█████████▍| 379572/400000 [00:44<00:02, 8673.35it/s] 95%|█████████▌| 380446/400000 [00:44<00:02, 8693.07it/s] 95%|█████████▌| 381327/400000 [00:44<00:02, 8726.44it/s] 96%|█████████▌| 382200/400000 [00:44<00:02, 8704.27it/s] 96%|█████████▌| 383078/400000 [00:44<00:01, 8725.35it/s] 96%|█████████▌| 383961/400000 [00:44<00:01, 8754.06it/s] 96%|█████████▌| 384837/400000 [00:44<00:01, 8752.31it/s] 96%|█████████▋| 385719/400000 [00:44<00:01, 8770.39it/s] 97%|█████████▋| 386597/400000 [00:44<00:01, 8760.03it/s] 97%|█████████▋| 387484/400000 [00:44<00:01, 8792.43it/s] 97%|█████████▋| 388364/400000 [00:45<00:01, 8778.48it/s] 97%|█████████▋| 389242/400000 [00:45<00:01, 8718.77it/s] 98%|█████████▊| 390118/400000 [00:45<00:01, 8729.68it/s] 98%|█████████▊| 390992/400000 [00:45<00:01, 8692.57it/s] 98%|█████████▊| 391869/400000 [00:45<00:00, 8714.81it/s] 98%|█████████▊| 392766/400000 [00:45<00:00, 8787.39it/s] 98%|█████████▊| 393645/400000 [00:45<00:00, 8714.88it/s] 99%|█████████▊| 394517/400000 [00:45<00:00, 8714.58it/s] 99%|█████████▉| 395389/400000 [00:45<00:00, 8687.38it/s] 99%|█████████▉| 396275/400000 [00:45<00:00, 8737.93it/s] 99%|█████████▉| 397149/400000 [00:46<00:00, 8707.50it/s]100%|█████████▉| 398020/400000 [00:46<00:00, 8669.59it/s]100%|█████████▉| 398900/400000 [00:46<00:00, 8708.02it/s]100%|█████████▉| 399771/400000 [00:46<00:00, 8579.17it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8615.02it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f153c0b30f0>, <torchtext.data.dataset.TabularDataset object at 0x7f153c0b3240>, <torchtext.vocab.Vocab object at 0x7f153c0b3160>), {}) 

