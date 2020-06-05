
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fed09042378> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fed09042378> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fed7b0950d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fed7b0950d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fed1282a9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fed1282a9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fed289798c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fed289798c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fed289798c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 78%|███████▊  | 7700480/9912422 [00:00<00:00, 67938616.30it/s]9920512it [00:00, 36288544.95it/s]                             
0it [00:00, ?it/s]32768it [00:00, 441763.96it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s] 69%|██████▉   | 1138688/1648877 [00:00<00:00, 11384176.66it/s]1654784it [00:00, 6645646.95it/s]                              
0it [00:00, ?it/s]8192it [00:00, 202250.55it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed0c012978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed0c012cf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fed28979510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fed28979510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fed28979510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:43,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:43,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:42,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:42,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:29,  5.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:29,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:28,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:20,  7.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:20,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:20,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13,  9.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12,  9.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.46 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:06, 18.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:06, 18.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:04, 23.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:04, 23.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:00<00:03, 30.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 60.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 66.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 66.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 71.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 75.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 75.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 77.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 77.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 80.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.15s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.49 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.15s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.49 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.72 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ url]
0 examples [00:00, ? examples/s]2020-06-05 18:09:03.429787: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-05 18:09:03.443091: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-05 18:09:03.443274: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55754ed7d4b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-05 18:09:03.443295: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
28 examples [00:00, 278.91 examples/s]131 examples [00:00, 356.90 examples/s]234 examples [00:00, 443.74 examples/s]333 examples [00:00, 531.22 examples/s]429 examples [00:00, 612.55 examples/s]526 examples [00:00, 687.51 examples/s]627 examples [00:00, 760.18 examples/s]730 examples [00:00, 823.36 examples/s]829 examples [00:00, 866.02 examples/s]932 examples [00:01, 908.04 examples/s]1035 examples [00:01, 940.16 examples/s]1136 examples [00:01, 958.36 examples/s]1237 examples [00:01, 972.29 examples/s]1337 examples [00:01, 972.33 examples/s]1438 examples [00:01, 981.72 examples/s]1540 examples [00:01, 992.51 examples/s]1643 examples [00:01, 1002.02 examples/s]1746 examples [00:01, 1008.81 examples/s]1848 examples [00:01, 1004.28 examples/s]1952 examples [00:02, 1012.67 examples/s]2055 examples [00:02, 1016.54 examples/s]2157 examples [00:02, 1012.57 examples/s]2259 examples [00:02, 1012.83 examples/s]2361 examples [00:02, 969.87 examples/s] 2462 examples [00:02, 980.00 examples/s]2561 examples [00:02, 982.44 examples/s]2661 examples [00:02, 985.50 examples/s]2761 examples [00:02, 988.78 examples/s]2860 examples [00:02, 916.40 examples/s]2953 examples [00:03, 904.80 examples/s]3045 examples [00:03, 869.43 examples/s]3133 examples [00:03, 871.00 examples/s]3222 examples [00:03, 876.18 examples/s]3312 examples [00:03, 881.08 examples/s]3405 examples [00:03, 893.79 examples/s]3500 examples [00:03, 909.07 examples/s]3592 examples [00:03, 856.61 examples/s]3679 examples [00:03, 846.59 examples/s]3774 examples [00:04, 875.01 examples/s]3863 examples [00:04, 874.53 examples/s]3951 examples [00:04, 870.85 examples/s]4039 examples [00:04, 870.63 examples/s]4134 examples [00:04, 891.95 examples/s]4235 examples [00:04, 923.08 examples/s]4335 examples [00:04, 942.10 examples/s]4436 examples [00:04, 959.54 examples/s]4534 examples [00:04, 964.49 examples/s]4631 examples [00:04, 940.92 examples/s]4726 examples [00:05, 934.18 examples/s]4827 examples [00:05, 954.87 examples/s]4930 examples [00:05, 973.96 examples/s]5033 examples [00:05, 987.74 examples/s]5133 examples [00:05, 942.33 examples/s]5228 examples [00:05, 937.32 examples/s]5328 examples [00:05, 953.50 examples/s]5428 examples [00:05, 965.29 examples/s]5525 examples [00:05, 963.30 examples/s]5623 examples [00:05, 966.02 examples/s]5723 examples [00:06, 975.07 examples/s]5821 examples [00:06, 976.23 examples/s]5922 examples [00:06, 984.67 examples/s]6021 examples [00:06, 984.10 examples/s]6121 examples [00:06, 986.71 examples/s]6221 examples [00:06, 988.97 examples/s]6320 examples [00:06, 972.52 examples/s]6421 examples [00:06, 980.60 examples/s]6520 examples [00:06, 974.05 examples/s]6619 examples [00:06, 978.35 examples/s]6717 examples [00:07, 973.53 examples/s]6815 examples [00:07, 970.07 examples/s]6913 examples [00:07, 970.85 examples/s]7013 examples [00:07, 965.78 examples/s]7112 examples [00:07, 972.86 examples/s]7210 examples [00:07, 973.63 examples/s]7311 examples [00:07, 982.25 examples/s]7414 examples [00:07, 994.12 examples/s]7515 examples [00:07, 996.30 examples/s]7615 examples [00:07, 994.23 examples/s]7715 examples [00:08, 990.92 examples/s]7818 examples [00:08, 1000.77 examples/s]7921 examples [00:08, 1007.46 examples/s]8023 examples [00:08, 1008.66 examples/s]8124 examples [00:08, 1002.92 examples/s]8225 examples [00:08, 967.52 examples/s] 8325 examples [00:08, 974.79 examples/s]8423 examples [00:08, 973.62 examples/s]8525 examples [00:08, 982.18 examples/s]8625 examples [00:09, 987.08 examples/s]8727 examples [00:09, 996.62 examples/s]8830 examples [00:09, 1005.40 examples/s]8933 examples [00:09, 1010.57 examples/s]9035 examples [00:09, 1010.35 examples/s]9137 examples [00:09, 1005.54 examples/s]9238 examples [00:09, 984.17 examples/s] 9340 examples [00:09, 993.19 examples/s]9443 examples [00:09, 1001.51 examples/s]9545 examples [00:09, 1004.00 examples/s]9646 examples [00:10, 999.14 examples/s] 9746 examples [00:10, 996.06 examples/s]9849 examples [00:10, 1004.74 examples/s]9950 examples [00:10, 979.40 examples/s] 10049 examples [00:10, 922.06 examples/s]10150 examples [00:10, 945.40 examples/s]10248 examples [00:10, 954.94 examples/s]10348 examples [00:10, 967.84 examples/s]10449 examples [00:10, 977.59 examples/s]10548 examples [00:10, 976.32 examples/s]10649 examples [00:11, 983.82 examples/s]10749 examples [00:11, 987.65 examples/s]10850 examples [00:11, 992.08 examples/s]10951 examples [00:11, 996.74 examples/s]11051 examples [00:11, 997.07 examples/s]11152 examples [00:11, 998.81 examples/s]11252 examples [00:11, 980.69 examples/s]11352 examples [00:11, 984.89 examples/s]11453 examples [00:11, 990.41 examples/s]11553 examples [00:11, 973.07 examples/s]11652 examples [00:12, 977.57 examples/s]11752 examples [00:12, 983.47 examples/s]11853 examples [00:12, 991.14 examples/s]11953 examples [00:12, 966.01 examples/s]12053 examples [00:12, 974.33 examples/s]12151 examples [00:12, 974.81 examples/s]12250 examples [00:12, 979.05 examples/s]12349 examples [00:12, 981.99 examples/s]12450 examples [00:12, 988.43 examples/s]12549 examples [00:12, 987.98 examples/s]12650 examples [00:13, 991.19 examples/s]12750 examples [00:13, 990.02 examples/s]12851 examples [00:13, 994.74 examples/s]12951 examples [00:13, 992.21 examples/s]13051 examples [00:13, 932.74 examples/s]13146 examples [00:13, 928.22 examples/s]13242 examples [00:13, 936.23 examples/s]13343 examples [00:13, 956.19 examples/s]13445 examples [00:13, 971.73 examples/s]13543 examples [00:14, 960.58 examples/s]13644 examples [00:14, 974.86 examples/s]13745 examples [00:14, 982.87 examples/s]13844 examples [00:14, 966.17 examples/s]13945 examples [00:14, 978.32 examples/s]14044 examples [00:14, 979.39 examples/s]14146 examples [00:14, 987.00 examples/s]14245 examples [00:14, 953.80 examples/s]14347 examples [00:14, 970.59 examples/s]14449 examples [00:14, 984.83 examples/s]14548 examples [00:15, 973.20 examples/s]14651 examples [00:15, 987.47 examples/s]14750 examples [00:15, 984.82 examples/s]14849 examples [00:15, 974.90 examples/s]14950 examples [00:15, 982.34 examples/s]15050 examples [00:15, 985.11 examples/s]15151 examples [00:15, 990.00 examples/s]15252 examples [00:15, 993.24 examples/s]15353 examples [00:15, 996.47 examples/s]15453 examples [00:15, 991.38 examples/s]15553 examples [00:16, 986.38 examples/s]15654 examples [00:16, 991.30 examples/s]15754 examples [00:16, 988.99 examples/s]15855 examples [00:16, 994.22 examples/s]15955 examples [00:16, 988.14 examples/s]16054 examples [00:16, 984.17 examples/s]16153 examples [00:16, 965.22 examples/s]16252 examples [00:16, 971.41 examples/s]16352 examples [00:16, 978.74 examples/s]16452 examples [00:16, 984.72 examples/s]16551 examples [00:17, 985.08 examples/s]16654 examples [00:17, 995.92 examples/s]16756 examples [00:17, 1002.27 examples/s]16859 examples [00:17, 1008.88 examples/s]16961 examples [00:17, 1011.18 examples/s]17063 examples [00:17, 997.55 examples/s] 17164 examples [00:17, 999.04 examples/s]17264 examples [00:17, 973.95 examples/s]17362 examples [00:17, 974.49 examples/s]17465 examples [00:18, 988.51 examples/s]17565 examples [00:18, 991.92 examples/s]17668 examples [00:18, 1002.36 examples/s]17771 examples [00:18, 1009.23 examples/s]17873 examples [00:18, 1009.79 examples/s]17975 examples [00:18, 1010.20 examples/s]18077 examples [00:18, 1004.56 examples/s]18178 examples [00:18, 1002.74 examples/s]18279 examples [00:18, 1002.97 examples/s]18381 examples [00:18, 1006.01 examples/s]18482 examples [00:19, 999.36 examples/s] 18582 examples [00:19, 997.22 examples/s]18684 examples [00:19, 1001.28 examples/s]18785 examples [00:19, 1000.04 examples/s]18886 examples [00:19, 999.16 examples/s] 18986 examples [00:19, 996.99 examples/s]19086 examples [00:19, 988.72 examples/s]19188 examples [00:19, 997.69 examples/s]19289 examples [00:19, 1001.20 examples/s]19392 examples [00:19, 1007.32 examples/s]19493 examples [00:20, 1003.79 examples/s]19594 examples [00:20, 991.72 examples/s] 19694 examples [00:20, 967.73 examples/s]19795 examples [00:20, 978.54 examples/s]19895 examples [00:20, 983.70 examples/s]19996 examples [00:20, 991.01 examples/s]20096 examples [00:20, 941.74 examples/s]20191 examples [00:20, 940.45 examples/s]20290 examples [00:20, 953.32 examples/s]20386 examples [00:20, 932.70 examples/s]20484 examples [00:21, 944.89 examples/s]20583 examples [00:21, 955.22 examples/s]20685 examples [00:21, 973.16 examples/s]20783 examples [00:21, 953.39 examples/s]20884 examples [00:21, 967.44 examples/s]20984 examples [00:21, 974.19 examples/s]21082 examples [00:21, 973.03 examples/s]21180 examples [00:21, 951.70 examples/s]21279 examples [00:21, 962.75 examples/s]21382 examples [00:21, 979.66 examples/s]21483 examples [00:22, 987.57 examples/s]21582 examples [00:22, 987.80 examples/s]21683 examples [00:22, 993.23 examples/s]21783 examples [00:22, 990.74 examples/s]21883 examples [00:22, 989.24 examples/s]21982 examples [00:22, 983.82 examples/s]22081 examples [00:22, 960.23 examples/s]22181 examples [00:22, 970.92 examples/s]22282 examples [00:22, 980.41 examples/s]22381 examples [00:23, 980.13 examples/s]22481 examples [00:23, 985.34 examples/s]22580 examples [00:23, 985.03 examples/s]22680 examples [00:23, 989.02 examples/s]22779 examples [00:23, 962.76 examples/s]22879 examples [00:23, 972.21 examples/s]22980 examples [00:23, 982.54 examples/s]23079 examples [00:23, 974.27 examples/s]23180 examples [00:23, 984.45 examples/s]23279 examples [00:23, 978.75 examples/s]23377 examples [00:24, 945.98 examples/s]23475 examples [00:24, 954.30 examples/s]23575 examples [00:24, 965.44 examples/s]23677 examples [00:24, 981.18 examples/s]23778 examples [00:24, 987.34 examples/s]23879 examples [00:24, 993.52 examples/s]23980 examples [00:24, 997.88 examples/s]24080 examples [00:24, 990.99 examples/s]24180 examples [00:24, 991.04 examples/s]24280 examples [00:24, 993.33 examples/s]24380 examples [00:25, 979.07 examples/s]24478 examples [00:25, 965.30 examples/s]24575 examples [00:25, 953.41 examples/s]24677 examples [00:25, 970.37 examples/s]24776 examples [00:25, 975.84 examples/s]24875 examples [00:25, 977.93 examples/s]24973 examples [00:25, 963.46 examples/s]25073 examples [00:25, 971.78 examples/s]25173 examples [00:25, 977.75 examples/s]25271 examples [00:25, 976.55 examples/s]25369 examples [00:26, 953.29 examples/s]25465 examples [00:26, 954.44 examples/s]25563 examples [00:26, 959.81 examples/s]25664 examples [00:26, 971.93 examples/s]25764 examples [00:26, 978.91 examples/s]25864 examples [00:26, 984.19 examples/s]25963 examples [00:26, 957.37 examples/s]26061 examples [00:26, 963.38 examples/s]26161 examples [00:26, 973.29 examples/s]26260 examples [00:26, 977.42 examples/s]26358 examples [00:27, 964.80 examples/s]26456 examples [00:27, 967.81 examples/s]26557 examples [00:27, 978.78 examples/s]26655 examples [00:27, 957.94 examples/s]26754 examples [00:27, 966.96 examples/s]26854 examples [00:27, 974.53 examples/s]26952 examples [00:27, 969.95 examples/s]27052 examples [00:27, 978.32 examples/s]27153 examples [00:27, 984.90 examples/s]27254 examples [00:28, 991.62 examples/s]27354 examples [00:28, 991.85 examples/s]27454 examples [00:28, 985.74 examples/s]27553 examples [00:28, 986.81 examples/s]27652 examples [00:28, 966.40 examples/s]27753 examples [00:28, 977.35 examples/s]27854 examples [00:28, 985.17 examples/s]27954 examples [00:28, 985.66 examples/s]28054 examples [00:28, 987.77 examples/s]28156 examples [00:28, 994.75 examples/s]28257 examples [00:29, 998.85 examples/s]28357 examples [00:29, 998.90 examples/s]28457 examples [00:29, 981.51 examples/s]28556 examples [00:29, 975.11 examples/s]28657 examples [00:29, 983.47 examples/s]28756 examples [00:29, 980.86 examples/s]28855 examples [00:29, 979.61 examples/s]28953 examples [00:29, 965.81 examples/s]29052 examples [00:29, 972.79 examples/s]29151 examples [00:29, 975.69 examples/s]29250 examples [00:30, 978.19 examples/s]29348 examples [00:30, 961.78 examples/s]29446 examples [00:30, 966.83 examples/s]29545 examples [00:30, 971.80 examples/s]29643 examples [00:30, 969.44 examples/s]29743 examples [00:30, 977.77 examples/s]29842 examples [00:30, 979.66 examples/s]29940 examples [00:30, 965.66 examples/s]30037 examples [00:30, 922.69 examples/s]30131 examples [00:30, 926.71 examples/s]30232 examples [00:31, 948.28 examples/s]30333 examples [00:31, 964.48 examples/s]30431 examples [00:31, 966.46 examples/s]30528 examples [00:31, 950.77 examples/s]30624 examples [00:31, 945.04 examples/s]30722 examples [00:31, 954.87 examples/s]30819 examples [00:31, 956.42 examples/s]30916 examples [00:31, 958.89 examples/s]31015 examples [00:31, 967.30 examples/s]31113 examples [00:31, 968.49 examples/s]31210 examples [00:32, 957.15 examples/s]31306 examples [00:32, 951.69 examples/s]31402 examples [00:32, 937.53 examples/s]31496 examples [00:32, 908.51 examples/s]31590 examples [00:32, 916.11 examples/s]31687 examples [00:32, 929.86 examples/s]31782 examples [00:32, 933.80 examples/s]31879 examples [00:32, 941.80 examples/s]31979 examples [00:32, 956.35 examples/s]32078 examples [00:33, 964.83 examples/s]32180 examples [00:33, 976.97 examples/s]32278 examples [00:33, 939.16 examples/s]32373 examples [00:33, 941.42 examples/s]32470 examples [00:33, 949.78 examples/s]32569 examples [00:33, 959.73 examples/s]32667 examples [00:33, 963.40 examples/s]32768 examples [00:33, 974.94 examples/s]32866 examples [00:33, 974.93 examples/s]32966 examples [00:33, 980.76 examples/s]33066 examples [00:34, 984.94 examples/s]33166 examples [00:34, 989.05 examples/s]33265 examples [00:34, 984.82 examples/s]33364 examples [00:34, 985.43 examples/s]33463 examples [00:34, 978.21 examples/s]33564 examples [00:34, 986.42 examples/s]33665 examples [00:34, 991.81 examples/s]33765 examples [00:34, 993.96 examples/s]33865 examples [00:34, 983.09 examples/s]33964 examples [00:34, 972.50 examples/s]34064 examples [00:35, 978.76 examples/s]34163 examples [00:35, 979.49 examples/s]34263 examples [00:35, 983.92 examples/s]34362 examples [00:35, 943.37 examples/s]34458 examples [00:35, 945.72 examples/s]34554 examples [00:35, 949.36 examples/s]34655 examples [00:35, 965.70 examples/s]34756 examples [00:35, 976.10 examples/s]34854 examples [00:35, 971.18 examples/s]34956 examples [00:35, 984.47 examples/s]35058 examples [00:36, 992.17 examples/s]35158 examples [00:36, 994.09 examples/s]35258 examples [00:36, 983.35 examples/s]35357 examples [00:36, 961.04 examples/s]35454 examples [00:36, 957.89 examples/s]35551 examples [00:36, 961.33 examples/s]35652 examples [00:36, 974.03 examples/s]35751 examples [00:36, 976.40 examples/s]35851 examples [00:36, 982.23 examples/s]35952 examples [00:36, 988.79 examples/s]36053 examples [00:37, 992.81 examples/s]36155 examples [00:37, 998.46 examples/s]36255 examples [00:37, 995.71 examples/s]36356 examples [00:37, 998.07 examples/s]36456 examples [00:37, 984.26 examples/s]36557 examples [00:37, 989.00 examples/s]36656 examples [00:37, 980.27 examples/s]36755 examples [00:37, 981.69 examples/s]36854 examples [00:37, 972.22 examples/s]36952 examples [00:38, 969.28 examples/s]37052 examples [00:38, 977.53 examples/s]37150 examples [00:38, 968.71 examples/s]37248 examples [00:38, 969.49 examples/s]37348 examples [00:38, 978.18 examples/s]37449 examples [00:38, 986.61 examples/s]37548 examples [00:38, 986.61 examples/s]37647 examples [00:38, 987.12 examples/s]37747 examples [00:38, 990.58 examples/s]37848 examples [00:38, 994.49 examples/s]37950 examples [00:39, 999.08 examples/s]38050 examples [00:39, 998.70 examples/s]38152 examples [00:39, 1002.29 examples/s]38253 examples [00:39, 1001.91 examples/s]38354 examples [00:39, 964.33 examples/s] 38455 examples [00:39, 975.68 examples/s]38556 examples [00:39, 985.25 examples/s]38656 examples [00:39, 986.80 examples/s]38755 examples [00:39, 975.29 examples/s]38856 examples [00:39, 985.31 examples/s]38959 examples [00:40, 995.70 examples/s]39062 examples [00:40, 1003.01 examples/s]39163 examples [00:40, 985.69 examples/s] 39264 examples [00:40, 991.80 examples/s]39364 examples [00:40, 986.82 examples/s]39465 examples [00:40, 991.54 examples/s]39565 examples [00:40, 993.82 examples/s]39665 examples [00:40, 990.34 examples/s]39765 examples [00:40, 989.99 examples/s]39865 examples [00:40, 988.38 examples/s]39967 examples [00:41, 996.52 examples/s]40067 examples [00:41, 957.48 examples/s]40169 examples [00:41, 974.26 examples/s]40271 examples [00:41, 985.22 examples/s]40370 examples [00:41, 981.57 examples/s]40469 examples [00:41, 981.79 examples/s]40568 examples [00:41, 955.02 examples/s]40666 examples [00:41, 959.97 examples/s]40766 examples [00:41, 969.16 examples/s]40867 examples [00:41, 979.23 examples/s]40968 examples [00:42, 987.27 examples/s]41070 examples [00:42, 994.49 examples/s]41172 examples [00:42, 999.46 examples/s]41273 examples [00:42, 975.84 examples/s]41371 examples [00:42, 960.57 examples/s]41468 examples [00:42, 957.67 examples/s]41564 examples [00:42, 958.18 examples/s]41665 examples [00:42, 972.88 examples/s]41766 examples [00:42, 981.78 examples/s]41865 examples [00:42, 976.12 examples/s]41963 examples [00:43, 974.36 examples/s]42061 examples [00:43, 957.67 examples/s]42157 examples [00:43, 946.75 examples/s]42256 examples [00:43, 957.24 examples/s]42352 examples [00:43, 954.58 examples/s]42450 examples [00:43, 961.61 examples/s]42547 examples [00:43, 962.96 examples/s]42648 examples [00:43, 973.36 examples/s]42748 examples [00:43, 980.65 examples/s]42847 examples [00:44, 976.20 examples/s]42948 examples [00:44, 984.76 examples/s]43049 examples [00:44, 991.01 examples/s]43149 examples [00:44, 984.43 examples/s]43250 examples [00:44, 989.41 examples/s]43349 examples [00:44, 984.05 examples/s]43449 examples [00:44, 987.00 examples/s]43548 examples [00:44, 977.05 examples/s]43647 examples [00:44, 978.67 examples/s]43747 examples [00:44, 982.23 examples/s]43846 examples [00:45, 965.37 examples/s]43946 examples [00:45, 975.00 examples/s]44046 examples [00:45, 981.26 examples/s]44147 examples [00:45, 988.06 examples/s]44246 examples [00:45, 985.81 examples/s]44345 examples [00:45, 948.01 examples/s]44447 examples [00:45, 968.25 examples/s]44549 examples [00:45, 983.21 examples/s]44651 examples [00:45, 992.39 examples/s]44752 examples [00:45, 996.28 examples/s]44852 examples [00:46, 992.09 examples/s]44952 examples [00:46, 991.63 examples/s]45052 examples [00:46, 984.26 examples/s]45151 examples [00:46, 980.79 examples/s]45250 examples [00:46, 966.24 examples/s]45348 examples [00:46, 970.11 examples/s]45446 examples [00:46, 971.06 examples/s]45545 examples [00:46, 975.47 examples/s]45645 examples [00:46, 982.23 examples/s]45746 examples [00:46, 988.64 examples/s]45845 examples [00:47, 988.83 examples/s]45945 examples [00:47, 991.75 examples/s]46045 examples [00:47, 992.19 examples/s]46145 examples [00:47, 984.19 examples/s]46248 examples [00:47, 994.85 examples/s]46348 examples [00:47, 991.53 examples/s]46448 examples [00:47, 992.25 examples/s]46548 examples [00:47, 993.97 examples/s]46648 examples [00:47, 985.05 examples/s]46748 examples [00:47, 988.08 examples/s]46847 examples [00:48, 982.79 examples/s]46948 examples [00:48, 988.58 examples/s]47049 examples [00:48, 994.48 examples/s]47151 examples [00:48, 1000.23 examples/s]47252 examples [00:48, 1000.43 examples/s]47353 examples [00:48, 961.54 examples/s] 47455 examples [00:48, 976.53 examples/s]47557 examples [00:48, 988.46 examples/s]47657 examples [00:48, 991.33 examples/s]47757 examples [00:49, 990.28 examples/s]47857 examples [00:49, 989.18 examples/s]47958 examples [00:49, 994.75 examples/s]48059 examples [00:49, 996.52 examples/s]48161 examples [00:49, 1000.98 examples/s]48262 examples [00:49, 996.16 examples/s] 48362 examples [00:49, 969.46 examples/s]48460 examples [00:49, 962.24 examples/s]48560 examples [00:49, 971.97 examples/s]48661 examples [00:49, 981.81 examples/s]48761 examples [00:50, 986.12 examples/s]48860 examples [00:50, 986.86 examples/s]48963 examples [00:50, 997.12 examples/s]49064 examples [00:50, 1000.71 examples/s]49166 examples [00:50, 1004.91 examples/s]49267 examples [00:50, 1005.73 examples/s]49368 examples [00:50, 1000.36 examples/s]49470 examples [00:50, 1004.33 examples/s]49571 examples [00:50, 999.92 examples/s] 49672 examples [00:50, 1002.55 examples/s]49773 examples [00:51, 1004.64 examples/s]49874 examples [00:51, 993.20 examples/s] 49975 examples [00:51, 995.21 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6812/50000 [00:00<00:00, 68118.28 examples/s] 37%|███▋      | 18710/50000 [00:00<00:00, 78138.18 examples/s] 61%|██████    | 30366/50000 [00:00<00:00, 86713.16 examples/s] 85%|████████▍ | 42288/50000 [00:00<00:00, 94436.18 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 706.81 examples/s]169 examples [00:00, 769.52 examples/s]271 examples [00:00, 830.09 examples/s]371 examples [00:00, 873.36 examples/s]473 examples [00:00, 909.41 examples/s]572 examples [00:00, 930.46 examples/s]671 examples [00:00, 947.13 examples/s]772 examples [00:00, 964.19 examples/s]867 examples [00:00, 959.48 examples/s]961 examples [00:01, 952.46 examples/s]1062 examples [00:01, 967.85 examples/s]1158 examples [00:01, 962.10 examples/s]1254 examples [00:01, 959.36 examples/s]1350 examples [00:01, 951.10 examples/s]1445 examples [00:01, 769.29 examples/s]1546 examples [00:01, 827.93 examples/s]1647 examples [00:01, 875.07 examples/s]1739 examples [00:01, 886.88 examples/s]1838 examples [00:01, 913.40 examples/s]1932 examples [00:02, 915.79 examples/s]2033 examples [00:02, 941.06 examples/s]2133 examples [00:02, 957.60 examples/s]2233 examples [00:02, 968.36 examples/s]2331 examples [00:02, 958.25 examples/s]2428 examples [00:02, 938.83 examples/s]2525 examples [00:02, 945.76 examples/s]2624 examples [00:02, 957.54 examples/s]2721 examples [00:02, 934.65 examples/s]2818 examples [00:03, 943.43 examples/s]2918 examples [00:03, 959.20 examples/s]3019 examples [00:03, 972.98 examples/s]3121 examples [00:03, 985.83 examples/s]3223 examples [00:03, 993.68 examples/s]3324 examples [00:03, 997.57 examples/s]3427 examples [00:03, 1004.60 examples/s]3529 examples [00:03, 1006.37 examples/s]3631 examples [00:03, 1007.45 examples/s]3732 examples [00:03, 1006.27 examples/s]3833 examples [00:04, 1000.89 examples/s]3934 examples [00:04, 985.47 examples/s] 4036 examples [00:04, 992.75 examples/s]4137 examples [00:04, 995.23 examples/s]4237 examples [00:04, 994.51 examples/s]4337 examples [00:04, 982.56 examples/s]4436 examples [00:04, 980.86 examples/s]4537 examples [00:04, 988.90 examples/s]4637 examples [00:04, 991.57 examples/s]4737 examples [00:04, 983.63 examples/s]4836 examples [00:05, 982.72 examples/s]4936 examples [00:05, 985.06 examples/s]5037 examples [00:05, 990.07 examples/s]5137 examples [00:05, 979.16 examples/s]5236 examples [00:05, 979.68 examples/s]5334 examples [00:05, 978.02 examples/s]5433 examples [00:05, 978.92 examples/s]5532 examples [00:05, 980.92 examples/s]5631 examples [00:05, 981.55 examples/s]5730 examples [00:05, 931.95 examples/s]5830 examples [00:06, 949.96 examples/s]5926 examples [00:06, 937.93 examples/s]6027 examples [00:06, 957.05 examples/s]6124 examples [00:06, 957.30 examples/s]6223 examples [00:06, 964.77 examples/s]6321 examples [00:06, 969.18 examples/s]6424 examples [00:06, 984.66 examples/s]6523 examples [00:06, 980.09 examples/s]6623 examples [00:06, 985.92 examples/s]6722 examples [00:06, 985.15 examples/s]6821 examples [00:07, 985.56 examples/s]6920 examples [00:07, 955.90 examples/s]7022 examples [00:07, 972.31 examples/s]7124 examples [00:07, 983.67 examples/s]7223 examples [00:07, 979.58 examples/s]7325 examples [00:07, 991.26 examples/s]7425 examples [00:07, 993.78 examples/s]7527 examples [00:07, 1000.33 examples/s]7628 examples [00:07, 1002.12 examples/s]7729 examples [00:08, 994.92 examples/s] 7829 examples [00:08, 995.34 examples/s]7931 examples [00:08, 1000.16 examples/s]8033 examples [00:08, 1004.49 examples/s]8135 examples [00:08, 1008.48 examples/s]8236 examples [00:08, 1006.70 examples/s]8337 examples [00:08, 1006.14 examples/s]8438 examples [00:08, 1007.03 examples/s]8540 examples [00:08, 1010.42 examples/s]8643 examples [00:08, 1016.07 examples/s]8745 examples [00:09, 973.43 examples/s] 8843 examples [00:09, 965.97 examples/s]8946 examples [00:09, 981.59 examples/s]9047 examples [00:09, 988.92 examples/s]9148 examples [00:09, 994.53 examples/s]9248 examples [00:09, 990.17 examples/s]9348 examples [00:09, 984.44 examples/s]9449 examples [00:09, 991.60 examples/s]9550 examples [00:09, 995.18 examples/s]9650 examples [00:09, 996.46 examples/s]9750 examples [00:10, 994.87 examples/s]9850 examples [00:10, 992.86 examples/s]9951 examples [00:10, 996.11 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU4P2I4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteU4P2I4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fed289798c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fed289798c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fed289798c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed94df3e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed0a149588>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fed289798c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fed289798c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fed289798c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed0aa6e390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fed0aa6e240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fed0035c7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fed0035c7b8> 

  function with postional parmater data_info <function split_train_valid at 0x7fed0035c7b8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fed0035c8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fed0035c8c8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fed0035c8c8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=adedf2b152a9d2e6d162fb0272958053b83d97fc3d0864c7683fbcabbe87280d
  Stored in directory: /tmp/pip-ephem-wheel-cache-bjkyn66y/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:37:36, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:46:32, 14.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:47:47, 20.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:15:55, 28.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:46:15, 41.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.67M/862M [00:01<4:00:44, 59.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:48:05, 84.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:56:58, 120kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<1:21:27, 172kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<56:55, 245kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:02<39:40, 349kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:02<27:39, 497kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<19:26, 704kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<13:35, 1.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<09:36, 1.41MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<07:36, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<07:13, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<07:19, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:41, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<06:17, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:38, 2.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:07<03:24, 3.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<25:47, 515kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<19:35, 678kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<14:01, 945kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<12:34, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<11:53, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<08:57, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<06:33, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<07:49, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<06:51, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<05:08, 2.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<05:45, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:15<03:11, 4.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<36:22, 358kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:17<26:47, 485kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<18:59, 683kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<16:20, 792kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<12:32, 1.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:19<09:06, 1.42MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:19<06:31, 1.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<53:10, 242kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<39:52, 323kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<28:26, 452kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:21<20:03, 640kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<19:25, 659kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<14:54, 858kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<10:44, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<10:29, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<08:37, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<06:17, 2.02MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:25, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:48, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:01, 2.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:22, 2.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<05:26, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<8:02:11, 26.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<5:37:22, 37.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<3:57:40, 52.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<2:49:02, 74.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<1:58:52, 105kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<1:23:02, 150kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:28:04, 142kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:02:53, 198kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<44:13, 281kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<33:46, 367kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<26:14, 472kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<18:58, 653kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:13, 810kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:57, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:37, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:54, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:44, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:39, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:51, 2.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:44, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:41, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:59, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:19, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:43, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:58, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:26, 2.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:16, 3.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<30:50, 391kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<24:18, 496kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<17:40, 681kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<12:29, 960kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<22:58, 522kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:47, 638kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:50, 865kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<09:49, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<18:58, 628kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:53, 750kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<11:45, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<08:22, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<20:47, 570kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<17:11, 689kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<12:36, 939kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<08:57, 1.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<11:46, 1.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<10:48, 1.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:11, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<05:52, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<20:00, 586kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<19:17, 607kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<14:48, 790kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<10:37, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:50, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:33, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<07:14, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:18, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:47, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:13, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:34, 2.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:05, 2.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:43, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:24, 1.55MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:43, 2.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:11, 2.73MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:36, 1.73MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:10, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:38, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:05, 2.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<17:13, 660kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<14:28, 785kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:38, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<07:39, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:36, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:39, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:36, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:48, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:49, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:16, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:38, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:05, 2.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<08:18, 1.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<08:11, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:15, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:30, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:31, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:33, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:33, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:41, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:26, 885kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<11:05, 993kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:20, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<05:56, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<21:39, 505kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<17:47, 615kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<13:00, 840kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<09:16, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<10:04, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:23, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:11, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<05:10, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<22:05, 490kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<18:12, 594kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<13:22, 807kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<09:29, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<17:01, 631kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<14:51, 722kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<10:55, 982kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<07:59, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:44, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:51, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:09, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:25, 2.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<14:17, 742kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<12:17, 862kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<09:05, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:57, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:43, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:57, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<04:16, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<48:51, 214kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<36:48, 284kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<26:11, 399kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<18:34, 561kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<15:19, 678kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<12:42, 817kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<09:17, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:38, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:59, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<08:44, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:42, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:48, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<12:27, 823kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<10:54, 940kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<08:05, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:49, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:36, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:07, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:25, 1.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:31, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:01, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:46, 2.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:26, 2.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<14:58, 671kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<12:07, 828kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<08:50, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<06:18, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<11:36, 860kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<10:15, 973kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:38, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:00, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:28, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:54, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:11, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:40, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:28, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:32, 2.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<7:03:52, 23.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<4:57:33, 32.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<3:27:58, 46.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<2:27:15, 65.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<1:47:03, 90.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<1:15:45, 128kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<53:09, 182kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<39:12, 246kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<29:24, 328kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<21:03, 457kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<14:46, 648kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<1:00:34, 158kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<43:44, 219kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<30:53, 309kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<23:18, 408kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<18:08, 524kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<13:03, 726kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<09:19, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:14, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:19, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:16, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:18, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:22, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:23, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:38, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:44, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:27, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:08, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:44, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:50, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:18, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:38, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:03, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:12, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:25, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:40, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:22, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:59, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:44, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:26, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:09, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:13, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:30, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:01, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:11, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:44, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:32, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:17, 2.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:23, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:22, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:37, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:10, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<06:18, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:34, 2.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:38, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:11, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:01, 2.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:15, 2.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:51, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:49, 3.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:55, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:40, 1.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:54, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<04:15, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<07:32, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:16, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:30, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:58, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:52, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:56, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<04:32, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<03:15, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:52, 946kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:18, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:14, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:22, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:42, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:29, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:15, 2.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:06, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:44, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:08, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:42, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<09:46, 837kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<09:06, 898kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:55, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:57, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<07:12, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<06:42, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:06, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:46, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<05:09, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:16, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:04, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:40, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<04:53, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:03, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:47, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:30, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<04:51, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<05:50, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:36, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:20, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:29, 3.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<29:12, 268kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<22:50, 343kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<16:28, 475kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<11:37, 671kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<08:15, 941kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<49:27, 157kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<37:00, 210kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<26:27, 293kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<18:35, 415kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<15:27, 498kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<13:02, 590kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<09:35, 802kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:49, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:53, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<3:37:58, 35.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<2:34:45, 49.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<1:48:40, 70.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<1:15:55, 100kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<53:04, 142kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<4:09:15, 30.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<2:56:29, 42.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<2:03:59, 60.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<1:26:36, 86.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<1:00:30, 124kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<4:12:24, 29.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<2:58:47, 41.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<2:05:32, 59.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:27:36, 85.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<1:03:40, 117kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<46:42, 159kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<33:10, 223kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<23:14, 317kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<18:45, 392kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<15:15, 482kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<11:06, 661kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<07:52, 929kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:38, 954kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<07:27, 977kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:38, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:03, 1.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:50, 1.49MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<05:30, 1.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:18, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:08, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:34, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:16, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:13, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:04, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:30, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:05, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:02, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:57, 2.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:32, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:05, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<04:02, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:57, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:30, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:03, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:57, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:53, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:21, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:52, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:48, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:30, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:33, 2.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<15:03, 452kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<13:06, 518kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<09:44, 697kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:55, 977kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:00, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:57, 968kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<07:25, 905kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:50, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:12, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:04, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<07:26, 895kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<07:45, 858kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<06:01, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:20, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<03:07, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<1:14:03, 89.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<54:09, 122kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<38:22, 172kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<26:52, 244kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<18:54, 345kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<14:36, 447kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<4:23:50, 24.7kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<3:05:03, 35.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<2:09:06, 50.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<1:31:38, 70.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<1:08:44, 93.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<49:17, 131kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<34:42, 185kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<24:15, 264kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<21:07, 302kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<17:01, 375kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<12:27, 512kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<08:49, 719kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:46, 813kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:30, 841kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:42, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:07, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:59, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<06:35, 949kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<06:40, 936kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:06, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:40, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:42, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:17, 983kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:25, 961kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:56, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:33, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:37, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<06:31, 936kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<06:42, 910kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<05:12, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:10, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:03, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:02, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:57, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:38, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:38, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<03:40, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:40, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:59, 2.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:59, 986kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:17, 939kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:53, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:32, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:58, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:41, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:42, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:41, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:00, 2.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:56, 969kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:03, 952kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:38, 1.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<03:23, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:49, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:33, 1.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:40, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:39, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:03, 2.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:15, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:44, 979kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:42, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:26, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:29, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:59, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:13, 893kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:00, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:39, 1.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:39, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<07:18, 752kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<07:33, 725kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:47, 946kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:18, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<03:06, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<02:18, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<12:08, 447kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<10:55, 496kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<08:07, 666kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:49, 925kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<04:11, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:48, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:45, 930kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:35, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:19, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:25, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<12:31, 422kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<10:55, 483kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:05, 652kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:46, 908kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<04:06, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<11:31, 452kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<10:02, 519kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<07:33, 689kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:23, 961kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<03:50, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<23:43, 217kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<18:43, 275kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<13:31, 380kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<09:34, 535kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<06:47, 750kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:39, 761kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:45, 750kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:11, 975kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:45, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:41, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<08:13, 608kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<07:48, 641kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<05:58, 837kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:18, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<03:04, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<20:56, 236kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<16:40, 296kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<12:05, 408kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<08:33, 573kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<06:05, 802kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<06:21, 765kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<06:26, 755kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:57, 980kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:35, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<02:34, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<10:14, 469kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<09:07, 525kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:48, 704kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:52, 978kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:29, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:28, 863kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:46, 819kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:26, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:15, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:21, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:25, 1.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:19, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:29, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:32, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:52, 2.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:48, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:32, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:34, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:35, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:56, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:59, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:56, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:12, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:20, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:43, 2.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:44, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<04:26, 1.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:28, 1.28MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:32, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:51, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:43, 928kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<04:59, 878kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:55, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:50, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:02, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<1:18:01, 55.3kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<56:13, 76.7kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<39:39, 109kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<27:43, 154kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<19:21, 220kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<17:31, 242kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<14:01, 303kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<10:09, 417kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<07:10, 587kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<05:04, 824kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:16, 666kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:05, 684kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:40, 892kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:20, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:24, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:43, 870kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:51, 845kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<03:48, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:44, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:00, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:21, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:53, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:06, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:15, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:12, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:33, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:52, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:24, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:27, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:05, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:31, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:50, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:20, 2.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<06:57, 551kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<2:26:23, 26.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<1:42:35, 37.3kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<1:11:24, 53.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<50:28, 74.6kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<37:59, 99.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<27:11, 138kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<19:05, 196kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<13:18, 279kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<11:32, 320kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<09:17, 398kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:44, 547kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<04:46, 767kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<03:24, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:24, 823kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:11, 864kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<02:19, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:42, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:05, 870kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:19, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:27, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:45, 1.99MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:52, 1.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:01, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:20, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:41, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:14, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<19:46, 173kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<14:49, 231kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<10:33, 323kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<07:25, 456kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<05:51, 571kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<05:04, 660kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:44, 894kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:40, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:37, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:41, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:29, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:13, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:23, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:52, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:20, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:20, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:22, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:49, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:19, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:42, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:52, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:29, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:04, 2.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:46, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:34, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:56, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:22, 2.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:15, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:12, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:42, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:12, 2.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:13, 889kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:57, 968kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:14, 1.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:35, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:32, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:18, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:44, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:14, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:25, 616kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:36, 754kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:38, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:51, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<06:21, 418kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:55, 538kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:33, 743kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:52, 900kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:26, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:48, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:40, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:46, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:21, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:59, 2.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:16, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:24, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:06, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:47, 3.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:37, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:40, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:17, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:55, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:55, 590kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<03:14, 714kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:22, 967kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:39, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<05:09, 435kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<04:05, 548kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:57, 755kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<02:04, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:13, 976kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:52, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:22, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:19, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:19, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:00, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:43, 2.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:00, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:47, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:20, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:14, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:08, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:51, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:56, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:00, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:46, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:33, 3.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:35, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:26, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:05, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:02, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:02, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:48, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:50, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:52, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:41, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:44, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:46, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:36, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:26, 3.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:55, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:42, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:30, 2.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:50, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:38, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:27, 3.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:07, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:35, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:24, 3.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<12:55, 99.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<09:14, 138kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<06:27, 196kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<04:32, 267kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<03:23, 358kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<02:23, 501kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:38, 707kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:33, 731kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:18, 877kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:56, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:39, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:05, 990kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:57, 1.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:42, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:38, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:27, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:20, 2.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:16, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:38, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:38, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:29, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:20, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:27, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:21, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:15, 2.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:20, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:20, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:23, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:11, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:17, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:17, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:09, 2.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:29, 641kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:24, 766kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:17, 1.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:11, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:11, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:14, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:11, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:07, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:10, 966kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:26, 89.4kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:18, 122kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:10, 171kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:01, 243kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<21:45:11,  5.11it/s]  0%|          | 766/400000 [00:00<15:12:08,  7.29it/s]  0%|          | 1536/400000 [00:00<10:37:31, 10.42it/s]  1%|          | 2287/400000 [00:00<7:25:41, 14.87it/s]   1%|          | 3043/400000 [00:00<5:11:39, 21.23it/s]  1%|          | 3815/400000 [00:00<3:37:59, 30.29it/s]  1%|          | 4588/400000 [00:00<2:32:33, 43.20it/s]  1%|▏         | 5336/400000 [00:00<1:46:50, 61.56it/s]  2%|▏         | 6057/400000 [00:00<1:14:55, 87.62it/s]  2%|▏         | 6815/400000 [00:01<52:36, 124.56it/s]   2%|▏         | 7552/400000 [00:01<37:01, 176.66it/s]  2%|▏         | 8323/400000 [00:01<26:07, 249.92it/s]  2%|▏         | 9102/400000 [00:01<18:29, 352.18it/s]  2%|▏         | 9857/400000 [00:01<13:10, 493.25it/s]  3%|▎         | 10636/400000 [00:01<09:27, 686.02it/s]  3%|▎         | 11396/400000 [00:01<06:52, 943.03it/s]  3%|▎         | 12181/400000 [00:01<05:02, 1281.20it/s]  3%|▎         | 12953/400000 [00:01<03:46, 1708.56it/s]  3%|▎         | 13720/400000 [00:02<02:53, 2223.72it/s]  4%|▎         | 14494/400000 [00:02<02:16, 2828.13it/s]  4%|▍         | 15262/400000 [00:02<01:50, 3489.10it/s]  4%|▍         | 16028/400000 [00:02<01:32, 4152.89it/s]  4%|▍         | 16807/400000 [00:02<01:19, 4828.18it/s]  4%|▍         | 17573/400000 [00:02<01:10, 5415.85it/s]  5%|▍         | 18343/400000 [00:02<01:04, 5943.18it/s]  5%|▍         | 19117/400000 [00:02<00:59, 6387.47it/s]  5%|▍         | 19887/400000 [00:02<00:56, 6729.81it/s]  5%|▌         | 20655/400000 [00:02<00:54, 6965.15it/s]  5%|▌         | 21421/400000 [00:03<00:53, 7115.69it/s]  6%|▌         | 22182/400000 [00:03<00:52, 7213.21it/s]  6%|▌         | 22944/400000 [00:03<00:51, 7327.22it/s]  6%|▌         | 23702/400000 [00:03<00:50, 7395.57it/s]  6%|▌         | 24460/400000 [00:03<00:50, 7448.05it/s]  6%|▋         | 25218/400000 [00:03<00:50, 7487.02it/s]  6%|▋         | 25976/400000 [00:03<00:50, 7460.14it/s]  7%|▋         | 26754/400000 [00:03<00:49, 7552.00it/s]  7%|▋         | 27534/400000 [00:03<00:48, 7624.19it/s]  7%|▋         | 28311/400000 [00:03<00:48, 7665.16it/s]  7%|▋         | 29080/400000 [00:04<00:49, 7563.19it/s]  7%|▋         | 29856/400000 [00:04<00:48, 7619.57it/s]  8%|▊         | 30633/400000 [00:04<00:48, 7663.14it/s]  8%|▊         | 31401/400000 [00:04<00:48, 7634.32it/s]  8%|▊         | 32180/400000 [00:04<00:47, 7678.25it/s]  8%|▊         | 32949/400000 [00:04<00:48, 7611.16it/s]  8%|▊         | 33711/400000 [00:04<00:48, 7575.45it/s]  9%|▊         | 34493/400000 [00:04<00:47, 7646.81it/s]  9%|▉         | 35259/400000 [00:04<00:48, 7565.04it/s]  9%|▉         | 36020/400000 [00:04<00:48, 7578.07it/s]  9%|▉         | 36779/400000 [00:05<00:48, 7565.51it/s]  9%|▉         | 37536/400000 [00:05<00:48, 7521.51it/s] 10%|▉         | 38303/400000 [00:05<00:47, 7562.99it/s] 10%|▉         | 39075/400000 [00:05<00:47, 7606.76it/s] 10%|▉         | 39848/400000 [00:05<00:47, 7642.76it/s] 10%|█         | 40613/400000 [00:05<00:47, 7601.74it/s] 10%|█         | 41374/400000 [00:05<00:47, 7602.31it/s] 11%|█         | 42135/400000 [00:05<00:47, 7569.49it/s] 11%|█         | 42918/400000 [00:05<00:46, 7643.29it/s] 11%|█         | 43701/400000 [00:05<00:46, 7696.95it/s] 11%|█         | 44471/400000 [00:06<00:46, 7638.78it/s] 11%|█▏        | 45245/400000 [00:06<00:46, 7667.85it/s] 12%|█▏        | 46013/400000 [00:06<00:46, 7666.77it/s] 12%|█▏        | 46782/400000 [00:06<00:46, 7672.87it/s] 12%|█▏        | 47558/400000 [00:06<00:45, 7696.18it/s] 12%|█▏        | 48328/400000 [00:06<00:45, 7659.42it/s] 12%|█▏        | 49101/400000 [00:06<00:45, 7679.17it/s] 12%|█▏        | 49872/400000 [00:06<00:45, 7687.81it/s] 13%|█▎        | 50641/400000 [00:06<00:45, 7649.07it/s] 13%|█▎        | 51411/400000 [00:06<00:45, 7664.00it/s] 13%|█▎        | 52178/400000 [00:07<00:45, 7636.34it/s] 13%|█▎        | 52949/400000 [00:07<00:45, 7657.24it/s] 13%|█▎        | 53715/400000 [00:07<00:45, 7569.29it/s] 14%|█▎        | 54480/400000 [00:07<00:45, 7592.18it/s] 14%|█▍        | 55240/400000 [00:07<00:45, 7565.84it/s] 14%|█▍        | 56006/400000 [00:07<00:45, 7593.21it/s] 14%|█▍        | 56778/400000 [00:07<00:44, 7629.03it/s] 14%|█▍        | 57550/400000 [00:07<00:44, 7655.13it/s] 15%|█▍        | 58321/400000 [00:07<00:44, 7670.11it/s] 15%|█▍        | 59089/400000 [00:07<00:44, 7615.05it/s] 15%|█▍        | 59851/400000 [00:08<00:45, 7557.97it/s] 15%|█▌        | 60608/400000 [00:08<00:45, 7534.21it/s] 15%|█▌        | 61382/400000 [00:08<00:44, 7593.22it/s] 16%|█▌        | 62152/400000 [00:08<00:44, 7623.65it/s] 16%|█▌        | 62921/400000 [00:08<00:44, 7642.71it/s] 16%|█▌        | 63686/400000 [00:08<00:44, 7558.26it/s] 16%|█▌        | 64458/400000 [00:08<00:44, 7605.43it/s] 16%|█▋        | 65230/400000 [00:08<00:43, 7638.42it/s] 17%|█▋        | 66001/400000 [00:08<00:43, 7658.19it/s] 17%|█▋        | 66767/400000 [00:08<00:43, 7656.85it/s] 17%|█▋        | 67538/400000 [00:09<00:43, 7671.87it/s] 17%|█▋        | 68306/400000 [00:09<00:44, 7512.79it/s] 17%|█▋        | 69072/400000 [00:09<00:43, 7555.78it/s] 17%|█▋        | 69831/400000 [00:09<00:43, 7563.25it/s] 18%|█▊        | 70600/400000 [00:09<00:43, 7509.36it/s] 18%|█▊        | 71352/400000 [00:09<00:44, 7444.58it/s] 18%|█▊        | 72097/400000 [00:09<00:44, 7328.51it/s] 18%|█▊        | 72841/400000 [00:09<00:44, 7360.34it/s] 18%|█▊        | 73600/400000 [00:09<00:43, 7426.93it/s] 19%|█▊        | 74369/400000 [00:09<00:43, 7502.60it/s] 19%|█▉        | 75120/400000 [00:10<00:43, 7460.96it/s] 19%|█▉        | 75887/400000 [00:10<00:43, 7522.40it/s] 19%|█▉        | 76640/400000 [00:10<00:43, 7463.15it/s] 19%|█▉        | 77408/400000 [00:10<00:42, 7524.82it/s] 20%|█▉        | 78174/400000 [00:10<00:42, 7562.54it/s] 20%|█▉        | 78934/400000 [00:10<00:42, 7571.82it/s] 20%|█▉        | 79706/400000 [00:10<00:42, 7613.13it/s] 20%|██        | 80474/400000 [00:10<00:41, 7630.87it/s] 20%|██        | 81238/400000 [00:10<00:42, 7558.36it/s] 21%|██        | 82013/400000 [00:10<00:41, 7613.83it/s] 21%|██        | 82775/400000 [00:11<00:41, 7591.97it/s] 21%|██        | 83546/400000 [00:11<00:41, 7625.21it/s] 21%|██        | 84320/400000 [00:11<00:41, 7657.35it/s] 21%|██▏       | 85086/400000 [00:11<00:41, 7575.14it/s] 21%|██▏       | 85844/400000 [00:11<00:42, 7389.65it/s] 22%|██▏       | 86585/400000 [00:11<00:42, 7387.93it/s] 22%|██▏       | 87325/400000 [00:11<00:42, 7356.96it/s] 22%|██▏       | 88096/400000 [00:11<00:41, 7459.17it/s] 22%|██▏       | 88874/400000 [00:11<00:41, 7550.02it/s] 22%|██▏       | 89630/400000 [00:12<00:41, 7526.74it/s] 23%|██▎       | 90384/400000 [00:12<00:41, 7491.18it/s] 23%|██▎       | 91143/400000 [00:12<00:41, 7520.42it/s] 23%|██▎       | 91907/400000 [00:12<00:40, 7552.98it/s] 23%|██▎       | 92677/400000 [00:12<00:40, 7593.76it/s] 23%|██▎       | 93455/400000 [00:12<00:40, 7647.70it/s] 24%|██▎       | 94221/400000 [00:12<00:40, 7552.02it/s] 24%|██▎       | 94977/400000 [00:12<00:41, 7369.85it/s] 24%|██▍       | 95740/400000 [00:12<00:40, 7445.06it/s] 24%|██▍       | 96513/400000 [00:12<00:40, 7527.55it/s] 24%|██▍       | 97267/400000 [00:13<00:40, 7429.67it/s] 25%|██▍       | 98011/400000 [00:13<00:40, 7380.15it/s] 25%|██▍       | 98762/400000 [00:13<00:40, 7418.06it/s] 25%|██▍       | 99541/400000 [00:13<00:39, 7524.51it/s] 25%|██▌       | 100316/400000 [00:13<00:39, 7588.56it/s] 25%|██▌       | 101093/400000 [00:13<00:39, 7642.03it/s] 25%|██▌       | 101858/400000 [00:13<00:39, 7564.82it/s] 26%|██▌       | 102616/400000 [00:13<00:39, 7508.00it/s] 26%|██▌       | 103385/400000 [00:13<00:39, 7560.55it/s] 26%|██▌       | 104162/400000 [00:13<00:38, 7620.35it/s] 26%|██▌       | 104935/400000 [00:14<00:38, 7652.69it/s] 26%|██▋       | 105701/400000 [00:14<00:38, 7572.84it/s] 27%|██▋       | 106459/400000 [00:14<00:38, 7561.61it/s] 27%|██▋       | 107216/400000 [00:14<00:39, 7480.67it/s] 27%|██▋       | 107979/400000 [00:14<00:38, 7522.85it/s] 27%|██▋       | 108752/400000 [00:14<00:38, 7583.17it/s] 27%|██▋       | 109511/400000 [00:14<00:38, 7573.25it/s] 28%|██▊       | 110279/400000 [00:14<00:38, 7604.42it/s] 28%|██▊       | 111040/400000 [00:14<00:38, 7513.98it/s] 28%|██▊       | 111806/400000 [00:14<00:38, 7554.22it/s] 28%|██▊       | 112582/400000 [00:15<00:37, 7613.78it/s] 28%|██▊       | 113344/400000 [00:15<00:37, 7602.52it/s] 29%|██▊       | 114114/400000 [00:15<00:37, 7628.88it/s] 29%|██▊       | 114894/400000 [00:15<00:37, 7678.51it/s] 29%|██▉       | 115663/400000 [00:15<00:37, 7637.41it/s] 29%|██▉       | 116439/400000 [00:15<00:36, 7672.60it/s] 29%|██▉       | 117207/400000 [00:15<00:37, 7568.55it/s] 29%|██▉       | 117982/400000 [00:15<00:37, 7620.33it/s] 30%|██▉       | 118755/400000 [00:15<00:36, 7651.19it/s] 30%|██▉       | 119529/400000 [00:15<00:36, 7677.42it/s] 30%|███       | 120297/400000 [00:16<00:36, 7620.82it/s] 30%|███       | 121060/400000 [00:16<00:36, 7595.61it/s] 30%|███       | 121821/400000 [00:16<00:36, 7599.53it/s] 31%|███       | 122582/400000 [00:16<00:36, 7594.58it/s] 31%|███       | 123359/400000 [00:16<00:36, 7644.36it/s] 31%|███       | 124124/400000 [00:16<00:36, 7615.67it/s] 31%|███       | 124894/400000 [00:16<00:36, 7639.76it/s] 31%|███▏      | 125659/400000 [00:16<00:35, 7636.94it/s] 32%|███▏      | 126437/400000 [00:16<00:35, 7677.71it/s] 32%|███▏      | 127215/400000 [00:16<00:35, 7705.99it/s] 32%|███▏      | 127994/400000 [00:17<00:35, 7729.16it/s] 32%|███▏      | 128768/400000 [00:17<00:35, 7590.01it/s] 32%|███▏      | 129539/400000 [00:17<00:35, 7624.24it/s] 33%|███▎      | 130319/400000 [00:17<00:35, 7675.31it/s] 33%|███▎      | 131096/400000 [00:17<00:34, 7702.40it/s] 33%|███▎      | 131870/400000 [00:17<00:34, 7711.87it/s] 33%|███▎      | 132642/400000 [00:17<00:35, 7444.67it/s] 33%|███▎      | 133405/400000 [00:17<00:35, 7498.56it/s] 34%|███▎      | 134178/400000 [00:17<00:35, 7565.85it/s] 34%|███▎      | 134936/400000 [00:17<00:35, 7509.28it/s] 34%|███▍      | 135688/400000 [00:18<00:35, 7504.81it/s] 34%|███▍      | 136451/400000 [00:18<00:34, 7539.25it/s] 34%|███▍      | 137206/400000 [00:18<00:35, 7490.24it/s] 34%|███▍      | 137965/400000 [00:18<00:34, 7518.96it/s] 35%|███▍      | 138736/400000 [00:18<00:34, 7573.60it/s] 35%|███▍      | 139508/400000 [00:18<00:34, 7614.11it/s] 35%|███▌      | 140274/400000 [00:18<00:34, 7624.98it/s] 35%|███▌      | 141041/400000 [00:18<00:33, 7637.16it/s] 35%|███▌      | 141805/400000 [00:18<00:34, 7589.79it/s] 36%|███▌      | 142576/400000 [00:18<00:33, 7624.21it/s] 36%|███▌      | 143351/400000 [00:19<00:33, 7659.83it/s] 36%|███▌      | 144119/400000 [00:19<00:33, 7665.60it/s] 36%|███▌      | 144886/400000 [00:19<00:33, 7646.65it/s] 36%|███▋      | 145651/400000 [00:19<00:33, 7613.29it/s] 37%|███▋      | 146430/400000 [00:19<00:33, 7663.12it/s] 37%|███▋      | 147197/400000 [00:19<00:33, 7607.82it/s] 37%|███▋      | 147974/400000 [00:19<00:32, 7655.59it/s] 37%|███▋      | 148740/400000 [00:19<00:32, 7641.50it/s] 37%|███▋      | 149507/400000 [00:19<00:32, 7649.48it/s] 38%|███▊      | 150273/400000 [00:19<00:32, 7603.10it/s] 38%|███▊      | 151052/400000 [00:20<00:32, 7655.35it/s] 38%|███▊      | 151834/400000 [00:20<00:32, 7703.53it/s] 38%|███▊      | 152605/400000 [00:20<00:32, 7673.75it/s] 38%|███▊      | 153382/400000 [00:20<00:32, 7699.66it/s] 39%|███▊      | 154153/400000 [00:20<00:32, 7599.74it/s] 39%|███▊      | 154930/400000 [00:20<00:32, 7647.56it/s] 39%|███▉      | 155696/400000 [00:20<00:31, 7642.00it/s] 39%|███▉      | 156461/400000 [00:20<00:31, 7615.82it/s] 39%|███▉      | 157238/400000 [00:20<00:31, 7660.73it/s] 40%|███▉      | 158014/400000 [00:21<00:31, 7688.68it/s] 40%|███▉      | 158784/400000 [00:21<00:31, 7610.11it/s] 40%|███▉      | 159554/400000 [00:21<00:31, 7633.40it/s] 40%|████      | 160318/400000 [00:21<00:31, 7555.42it/s] 40%|████      | 161093/400000 [00:21<00:31, 7610.94it/s] 40%|████      | 161866/400000 [00:21<00:31, 7644.49it/s] 41%|████      | 162643/400000 [00:21<00:30, 7678.25it/s] 41%|████      | 163412/400000 [00:21<00:31, 7628.50it/s] 41%|████      | 164176/400000 [00:21<00:30, 7621.78it/s] 41%|████      | 164947/400000 [00:21<00:30, 7647.38it/s] 41%|████▏     | 165712/400000 [00:22<00:30, 7645.00it/s] 42%|████▏     | 166483/400000 [00:22<00:30, 7664.25it/s] 42%|████▏     | 167250/400000 [00:22<00:30, 7626.10it/s] 42%|████▏     | 168013/400000 [00:22<00:30, 7613.99it/s] 42%|████▏     | 168792/400000 [00:22<00:30, 7663.31it/s] 42%|████▏     | 169572/400000 [00:22<00:29, 7702.56it/s] 43%|████▎     | 170349/400000 [00:22<00:29, 7721.45it/s] 43%|████▎     | 171128/400000 [00:22<00:29, 7739.33it/s] 43%|████▎     | 171903/400000 [00:22<00:30, 7556.00it/s] 43%|████▎     | 172660/400000 [00:22<00:30, 7543.45it/s] 43%|████▎     | 173428/400000 [00:23<00:29, 7583.09it/s] 44%|████▎     | 174187/400000 [00:23<00:30, 7412.05it/s] 44%|████▎     | 174947/400000 [00:23<00:30, 7466.86it/s] 44%|████▍     | 175695/400000 [00:23<00:30, 7433.61it/s] 44%|████▍     | 176445/400000 [00:23<00:30, 7451.47it/s] 44%|████▍     | 177213/400000 [00:23<00:29, 7516.04it/s] 44%|████▍     | 177988/400000 [00:23<00:29, 7583.10it/s] 45%|████▍     | 178762/400000 [00:23<00:29, 7627.12it/s] 45%|████▍     | 179526/400000 [00:23<00:28, 7621.62it/s] 45%|████▌     | 180289/400000 [00:23<00:28, 7598.64it/s] 45%|████▌     | 181067/400000 [00:24<00:28, 7649.56it/s] 45%|████▌     | 181844/400000 [00:24<00:28, 7685.00it/s] 46%|████▌     | 182621/400000 [00:24<00:28, 7707.93it/s] 46%|████▌     | 183392/400000 [00:24<00:28, 7639.58it/s] 46%|████▌     | 184157/400000 [00:24<00:28, 7624.89it/s] 46%|████▌     | 184939/400000 [00:24<00:27, 7681.23it/s] 46%|████▋     | 185708/400000 [00:24<00:28, 7615.15it/s] 47%|████▋     | 186472/400000 [00:24<00:28, 7620.39it/s] 47%|████▋     | 187235/400000 [00:24<00:27, 7605.25it/s] 47%|████▋     | 188013/400000 [00:24<00:27, 7655.39it/s] 47%|████▋     | 188779/400000 [00:25<00:27, 7603.81it/s] 47%|████▋     | 189551/400000 [00:25<00:27, 7637.81it/s] 48%|████▊     | 190328/400000 [00:25<00:27, 7675.30it/s] 48%|████▊     | 191096/400000 [00:25<00:27, 7654.16it/s] 48%|████▊     | 191872/400000 [00:25<00:27, 7683.06it/s] 48%|████▊     | 192647/400000 [00:25<00:26, 7701.07it/s] 48%|████▊     | 193418/400000 [00:25<00:26, 7655.62it/s] 49%|████▊     | 194198/400000 [00:25<00:26, 7697.07it/s] 49%|████▊     | 194968/400000 [00:25<00:26, 7694.24it/s] 49%|████▉     | 195738/400000 [00:25<00:26, 7682.47it/s] 49%|████▉     | 196507/400000 [00:26<00:26, 7643.89it/s] 49%|████▉     | 197272/400000 [00:26<00:26, 7553.87it/s] 50%|████▉     | 198028/400000 [00:26<00:26, 7537.37it/s] 50%|████▉     | 198786/400000 [00:26<00:26, 7549.26it/s] 50%|████▉     | 199558/400000 [00:26<00:26, 7598.51it/s] 50%|█████     | 200319/400000 [00:26<00:26, 7592.72it/s] 50%|█████     | 201095/400000 [00:26<00:26, 7640.70it/s] 50%|█████     | 201860/400000 [00:26<00:26, 7590.62it/s] 51%|█████     | 202621/400000 [00:26<00:25, 7595.06it/s] 51%|█████     | 203387/400000 [00:26<00:25, 7613.66it/s] 51%|█████     | 204158/400000 [00:27<00:25, 7642.17it/s] 51%|█████     | 204938/400000 [00:27<00:25, 7686.35it/s] 51%|█████▏    | 205709/400000 [00:27<00:25, 7692.47it/s] 52%|█████▏    | 206479/400000 [00:27<00:25, 7685.17it/s] 52%|█████▏    | 207257/400000 [00:27<00:24, 7710.82it/s] 52%|█████▏    | 208039/400000 [00:27<00:24, 7743.13it/s] 52%|█████▏    | 208815/400000 [00:27<00:24, 7747.25it/s] 52%|█████▏    | 209590/400000 [00:27<00:24, 7662.90it/s] 53%|█████▎    | 210357/400000 [00:27<00:25, 7554.08it/s] 53%|█████▎    | 211122/400000 [00:27<00:24, 7581.14it/s] 53%|█████▎    | 211894/400000 [00:28<00:24, 7621.73it/s] 53%|█████▎    | 212671/400000 [00:28<00:24, 7663.46it/s] 53%|█████▎    | 213451/400000 [00:28<00:24, 7702.38it/s] 54%|█████▎    | 214222/400000 [00:28<00:24, 7646.57it/s] 54%|█████▎    | 214987/400000 [00:28<00:24, 7611.68it/s] 54%|█████▍    | 215770/400000 [00:28<00:24, 7675.51it/s] 54%|█████▍    | 216546/400000 [00:28<00:23, 7698.28it/s] 54%|█████▍    | 217319/400000 [00:28<00:23, 7706.60it/s] 55%|█████▍    | 218093/400000 [00:28<00:23, 7714.55it/s] 55%|█████▍    | 218865/400000 [00:28<00:23, 7600.41it/s] 55%|█████▍    | 219630/400000 [00:29<00:23, 7614.07it/s] 55%|█████▌    | 220402/400000 [00:29<00:23, 7645.42it/s] 55%|█████▌    | 221175/400000 [00:29<00:23, 7669.94it/s] 55%|█████▌    | 221943/400000 [00:29<00:23, 7668.52it/s] 56%|█████▌    | 222710/400000 [00:29<00:23, 7582.81it/s] 56%|█████▌    | 223480/400000 [00:29<00:23, 7616.59it/s] 56%|█████▌    | 224265/400000 [00:29<00:22, 7682.52it/s] 56%|█████▋    | 225034/400000 [00:29<00:22, 7680.45it/s] 56%|█████▋    | 225803/400000 [00:29<00:22, 7672.04it/s] 57%|█████▋    | 226571/400000 [00:29<00:22, 7629.50it/s] 57%|█████▋    | 227335/400000 [00:30<00:22, 7541.80it/s] 57%|█████▋    | 228097/400000 [00:30<00:22, 7564.56it/s] 57%|█████▋    | 228865/400000 [00:30<00:22, 7598.73it/s] 57%|█████▋    | 229638/400000 [00:30<00:22, 7636.67it/s] 58%|█████▊    | 230402/400000 [00:30<00:22, 7630.76it/s] 58%|█████▊    | 231170/400000 [00:30<00:22, 7645.35it/s] 58%|█████▊    | 231937/400000 [00:30<00:21, 7649.74it/s] 58%|█████▊    | 232704/400000 [00:30<00:21, 7653.25it/s] 58%|█████▊    | 233481/400000 [00:30<00:21, 7686.86it/s] 59%|█████▊    | 234250/400000 [00:30<00:21, 7605.05it/s] 59%|█████▉    | 235027/400000 [00:31<00:21, 7653.62it/s] 59%|█████▉    | 235793/400000 [00:31<00:21, 7597.98it/s] 59%|█████▉    | 236568/400000 [00:31<00:21, 7641.74it/s] 59%|█████▉    | 237340/400000 [00:31<00:21, 7662.97it/s] 60%|█████▉    | 238107/400000 [00:31<00:21, 7656.62it/s] 60%|█████▉    | 238879/400000 [00:31<00:20, 7674.61it/s] 60%|█████▉    | 239647/400000 [00:31<00:21, 7590.81it/s] 60%|██████    | 240407/400000 [00:31<00:21, 7522.22it/s] 60%|██████    | 241177/400000 [00:31<00:20, 7574.53it/s] 60%|██████    | 241935/400000 [00:32<00:20, 7558.00it/s] 61%|██████    | 242694/400000 [00:32<00:20, 7565.93it/s] 61%|██████    | 243472/400000 [00:32<00:20, 7627.48it/s] 61%|██████    | 244235/400000 [00:32<00:20, 7603.07it/s] 61%|██████▏   | 245011/400000 [00:32<00:20, 7647.18it/s] 61%|██████▏   | 245776/400000 [00:32<00:20, 7620.06it/s] 62%|██████▏   | 246541/400000 [00:32<00:20, 7628.41it/s] 62%|██████▏   | 247311/400000 [00:32<00:19, 7648.62it/s] 62%|██████▏   | 248076/400000 [00:32<00:19, 7645.18it/s] 62%|██████▏   | 248841/400000 [00:32<00:19, 7606.01it/s] 62%|██████▏   | 249602/400000 [00:33<00:19, 7560.83it/s] 63%|██████▎   | 250359/400000 [00:33<00:19, 7540.61it/s] 63%|██████▎   | 251139/400000 [00:33<00:19, 7615.09it/s] 63%|██████▎   | 251921/400000 [00:33<00:19, 7672.79it/s] 63%|██████▎   | 252689/400000 [00:33<00:19, 7565.27it/s] 63%|██████▎   | 253447/400000 [00:33<00:19, 7520.23it/s] 64%|██████▎   | 254224/400000 [00:33<00:19, 7591.69it/s] 64%|██████▍   | 255004/400000 [00:33<00:18, 7652.65it/s] 64%|██████▍   | 255780/400000 [00:33<00:18, 7681.88it/s] 64%|██████▍   | 256564/400000 [00:33<00:18, 7728.56it/s] 64%|██████▍   | 257338/400000 [00:34<00:18, 7664.48it/s] 65%|██████▍   | 258105/400000 [00:34<00:18, 7657.64it/s] 65%|██████▍   | 258882/400000 [00:34<00:18, 7690.71it/s] 65%|██████▍   | 259654/400000 [00:34<00:18, 7697.17it/s] 65%|██████▌   | 260424/400000 [00:34<00:18, 7683.70it/s] 65%|██████▌   | 261193/400000 [00:34<00:18, 7610.13it/s] 65%|██████▌   | 261961/400000 [00:34<00:18, 7630.89it/s] 66%|██████▌   | 262735/400000 [00:34<00:17, 7661.26it/s] 66%|██████▌   | 263507/400000 [00:34<00:17, 7676.56it/s] 66%|██████▌   | 264279/400000 [00:34<00:17, 7689.38it/s] 66%|██████▋   | 265049/400000 [00:35<00:17, 7627.51it/s] 66%|██████▋   | 265812/400000 [00:35<00:17, 7572.12it/s] 67%|██████▋   | 266581/400000 [00:35<00:17, 7605.89it/s] 67%|██████▋   | 267355/400000 [00:35<00:17, 7643.01it/s] 67%|██████▋   | 268134/400000 [00:35<00:17, 7685.87it/s] 67%|██████▋   | 268903/400000 [00:35<00:17, 7623.71it/s] 67%|██████▋   | 269670/400000 [00:35<00:17, 7636.11it/s] 68%|██████▊   | 270434/400000 [00:35<00:17, 7595.55it/s] 68%|██████▊   | 271194/400000 [00:35<00:16, 7586.78it/s] 68%|██████▊   | 271954/400000 [00:35<00:16, 7590.34it/s] 68%|██████▊   | 272714/400000 [00:36<00:16, 7566.21it/s] 68%|██████▊   | 273475/400000 [00:36<00:16, 7576.66it/s] 69%|██████▊   | 274233/400000 [00:36<00:16, 7512.84it/s] 69%|██████▉   | 275009/400000 [00:36<00:16, 7584.57it/s] 69%|██████▉   | 275776/400000 [00:36<00:16, 7607.81it/s] 69%|██████▉   | 276537/400000 [00:36<00:16, 7561.80it/s] 69%|██████▉   | 277317/400000 [00:36<00:16, 7631.10it/s] 70%|██████▉   | 278096/400000 [00:36<00:15, 7676.24it/s] 70%|██████▉   | 278864/400000 [00:36<00:15, 7614.57it/s] 70%|██████▉   | 279639/400000 [00:36<00:15, 7654.17it/s] 70%|███████   | 280405/400000 [00:37<00:15, 7616.40it/s] 70%|███████   | 281172/400000 [00:37<00:15, 7630.44it/s] 70%|███████   | 281946/400000 [00:37<00:15, 7660.72it/s] 71%|███████   | 282713/400000 [00:37<00:15, 7617.07it/s] 71%|███████   | 283490/400000 [00:37<00:15, 7661.73it/s] 71%|███████   | 284257/400000 [00:37<00:15, 7547.88it/s] 71%|███████▏  | 285016/400000 [00:37<00:15, 7559.13it/s] 71%|███████▏  | 285792/400000 [00:37<00:14, 7615.46it/s] 72%|███████▏  | 286570/400000 [00:37<00:14, 7661.29it/s] 72%|███████▏  | 287337/400000 [00:37<00:14, 7595.04it/s] 72%|███████▏  | 288100/400000 [00:38<00:14, 7605.15it/s] 72%|███████▏  | 288872/400000 [00:38<00:14, 7636.52it/s] 72%|███████▏  | 289646/400000 [00:38<00:14, 7665.26it/s] 73%|███████▎  | 290413/400000 [00:38<00:14, 7663.79it/s] 73%|███████▎  | 291180/400000 [00:38<00:14, 7533.47it/s] 73%|███████▎  | 291941/400000 [00:38<00:14, 7554.13it/s] 73%|███████▎  | 292697/400000 [00:38<00:14, 7445.43it/s] 73%|███████▎  | 293475/400000 [00:38<00:14, 7541.04it/s] 74%|███████▎  | 294241/400000 [00:38<00:13, 7573.79it/s] 74%|███████▍  | 295019/400000 [00:38<00:13, 7634.07it/s] 74%|███████▍  | 295783/400000 [00:39<00:13, 7511.78it/s] 74%|███████▍  | 296561/400000 [00:39<00:13, 7590.23it/s] 74%|███████▍  | 297330/400000 [00:39<00:13, 7617.85it/s] 75%|███████▍  | 298093/400000 [00:39<00:13, 7610.17it/s] 75%|███████▍  | 298865/400000 [00:39<00:13, 7642.76it/s] 75%|███████▍  | 299630/400000 [00:39<00:13, 7562.34it/s] 75%|███████▌  | 300403/400000 [00:39<00:13, 7611.11it/s] 75%|███████▌  | 301180/400000 [00:39<00:12, 7655.42it/s] 75%|███████▌  | 301955/400000 [00:39<00:12, 7681.31it/s] 76%|███████▌  | 302729/400000 [00:39<00:12, 7697.81it/s] 76%|███████▌  | 303499/400000 [00:40<00:12, 7668.24it/s] 76%|███████▌  | 304266/400000 [00:40<00:12, 7513.52it/s] 76%|███████▋  | 305039/400000 [00:40<00:12, 7577.12it/s] 76%|███████▋  | 305817/400000 [00:40<00:12, 7633.87it/s] 77%|███████▋  | 306594/400000 [00:40<00:12, 7673.33it/s] 77%|███████▋  | 307362/400000 [00:40<00:12, 7621.71it/s] 77%|███████▋  | 308135/400000 [00:40<00:12, 7650.73it/s] 77%|███████▋  | 308901/400000 [00:40<00:11, 7633.55it/s] 77%|███████▋  | 309665/400000 [00:40<00:12, 7523.11it/s] 78%|███████▊  | 310430/400000 [00:40<00:11, 7559.16it/s] 78%|███████▊  | 311187/400000 [00:41<00:11, 7537.53it/s] 78%|███████▊  | 311952/400000 [00:41<00:11, 7568.42it/s] 78%|███████▊  | 312710/400000 [00:41<00:11, 7334.06it/s] 78%|███████▊  | 313471/400000 [00:41<00:11, 7413.33it/s] 79%|███████▊  | 314238/400000 [00:41<00:11, 7486.76it/s] 79%|███████▊  | 314988/400000 [00:41<00:11, 7470.73it/s] 79%|███████▉  | 315759/400000 [00:41<00:11, 7538.68it/s] 79%|███████▉  | 316537/400000 [00:41<00:10, 7607.63it/s] 79%|███████▉  | 317299/400000 [00:41<00:10, 7595.61it/s] 80%|███████▉  | 318076/400000 [00:42<00:10, 7645.57it/s] 80%|███████▉  | 318841/400000 [00:42<00:10, 7596.65it/s] 80%|███████▉  | 319613/400000 [00:42<00:10, 7632.55it/s] 80%|████████  | 320377/400000 [00:42<00:10, 7623.24it/s] 80%|████████  | 321140/400000 [00:42<00:10, 7566.12it/s] 80%|████████  | 321904/400000 [00:42<00:10, 7586.31it/s] 81%|████████  | 322663/400000 [00:42<00:10, 7563.13it/s] 81%|████████  | 323420/400000 [00:42<00:10, 7474.48it/s] 81%|████████  | 324181/400000 [00:42<00:10, 7513.82it/s] 81%|████████  | 324951/400000 [00:42<00:09, 7566.25it/s] 81%|████████▏ | 325708/400000 [00:43<00:09, 7526.22it/s] 82%|████████▏ | 326461/400000 [00:43<00:09, 7513.35it/s] 82%|████████▏ | 327232/400000 [00:43<00:09, 7570.89it/s] 82%|████████▏ | 327996/400000 [00:43<00:09, 7589.89it/s] 82%|████████▏ | 328759/400000 [00:43<00:09, 7601.06it/s] 82%|████████▏ | 329520/400000 [00:43<00:09, 7434.73it/s] 83%|████████▎ | 330265/400000 [00:43<00:09, 7381.16it/s] 83%|████████▎ | 331023/400000 [00:43<00:09, 7439.36it/s] 83%|████████▎ | 331780/400000 [00:43<00:09, 7476.34it/s] 83%|████████▎ | 332537/400000 [00:43<00:08, 7503.69it/s] 83%|████████▎ | 333305/400000 [00:44<00:08, 7553.45it/s] 84%|████████▎ | 334061/400000 [00:44<00:08, 7380.50it/s] 84%|████████▎ | 334801/400000 [00:44<00:08, 7287.85it/s] 84%|████████▍ | 335550/400000 [00:44<00:08, 7344.60it/s] 84%|████████▍ | 336316/400000 [00:44<00:08, 7435.68it/s] 84%|████████▍ | 337082/400000 [00:44<00:08, 7499.54it/s] 84%|████████▍ | 337842/400000 [00:44<00:08, 7528.12it/s] 85%|████████▍ | 338599/400000 [00:44<00:08, 7539.77it/s] 85%|████████▍ | 339364/400000 [00:44<00:08, 7571.12it/s] 85%|████████▌ | 340132/400000 [00:44<00:07, 7600.95it/s] 85%|████████▌ | 340906/400000 [00:45<00:07, 7641.83it/s] 85%|████████▌ | 341671/400000 [00:45<00:07, 7574.41it/s] 86%|████████▌ | 342429/400000 [00:45<00:07, 7452.18it/s] 86%|████████▌ | 343200/400000 [00:45<00:07, 7526.52it/s] 86%|████████▌ | 343971/400000 [00:45<00:07, 7580.22it/s] 86%|████████▌ | 344738/400000 [00:45<00:07, 7605.64it/s] 86%|████████▋ | 345499/400000 [00:45<00:07, 7596.98it/s] 87%|████████▋ | 346263/400000 [00:45<00:07, 7608.42it/s] 87%|████████▋ | 347025/400000 [00:45<00:06, 7571.87it/s] 87%|████████▋ | 347791/400000 [00:45<00:06, 7596.90it/s] 87%|████████▋ | 348551/400000 [00:46<00:06, 7594.28it/s] 87%|████████▋ | 349311/400000 [00:46<00:06, 7585.97it/s] 88%|████████▊ | 350078/400000 [00:46<00:06, 7608.68it/s] 88%|████████▊ | 350839/400000 [00:46<00:06, 7583.36it/s] 88%|████████▊ | 351610/400000 [00:46<00:06, 7618.45it/s] 88%|████████▊ | 352372/400000 [00:46<00:06, 7586.00it/s] 88%|████████▊ | 353131/400000 [00:46<00:06, 7454.71it/s] 88%|████████▊ | 353880/400000 [00:46<00:06, 7463.78it/s] 89%|████████▊ | 354649/400000 [00:46<00:06, 7529.70it/s] 89%|████████▉ | 355405/400000 [00:46<00:05, 7536.63it/s] 89%|████████▉ | 356177/400000 [00:47<00:05, 7589.36it/s] 89%|████████▉ | 356937/400000 [00:47<00:05, 7558.15it/s] 89%|████████▉ | 357708/400000 [00:47<00:05, 7600.72it/s] 90%|████████▉ | 358469/400000 [00:47<00:05, 7588.08it/s] 90%|████████▉ | 359228/400000 [00:47<00:05, 7533.81it/s] 90%|████████▉ | 359982/400000 [00:47<00:05, 7509.14it/s] 90%|█████████ | 360743/400000 [00:47<00:05, 7536.23it/s] 90%|█████████ | 361508/400000 [00:47<00:05, 7567.65it/s] 91%|█████████ | 362274/400000 [00:47<00:04, 7593.01it/s] 91%|█████████ | 363047/400000 [00:47<00:04, 7632.51it/s] 91%|█████████ | 363811/400000 [00:48<00:04, 7588.44it/s] 91%|█████████ | 364570/400000 [00:48<00:04, 7584.63it/s] 91%|█████████▏| 365338/400000 [00:48<00:04, 7611.98it/s] 92%|█████████▏| 366102/400000 [00:48<00:04, 7618.44it/s] 92%|█████████▏| 366876/400000 [00:48<00:04, 7651.52it/s] 92%|█████████▏| 367642/400000 [00:48<00:04, 7593.84it/s] 92%|█████████▏| 368404/400000 [00:48<00:04, 7598.72it/s] 92%|█████████▏| 369165/400000 [00:48<00:04, 7602.07it/s] 92%|█████████▏| 369932/400000 [00:48<00:03, 7620.42it/s] 93%|█████████▎| 370696/400000 [00:48<00:03, 7626.20it/s] 93%|█████████▎| 371459/400000 [00:49<00:03, 7559.96it/s] 93%|█████████▎| 372216/400000 [00:49<00:03, 7394.64it/s] 93%|█████████▎| 372974/400000 [00:49<00:03, 7446.70it/s] 93%|█████████▎| 373749/400000 [00:49<00:03, 7532.89it/s] 94%|█████████▎| 374508/400000 [00:49<00:03, 7548.18it/s] 94%|█████████▍| 375264/400000 [00:49<00:03, 7337.59it/s] 94%|█████████▍| 376012/400000 [00:49<00:03, 7379.16it/s] 94%|█████████▍| 376752/400000 [00:49<00:03, 7366.50it/s] 94%|█████████▍| 377515/400000 [00:49<00:03, 7443.06it/s] 95%|█████████▍| 378279/400000 [00:50<00:02, 7500.93it/s] 95%|█████████▍| 379046/400000 [00:50<00:02, 7549.33it/s] 95%|█████████▍| 379812/400000 [00:50<00:02, 7579.90it/s] 95%|█████████▌| 380571/400000 [00:50<00:02, 7496.72it/s] 95%|█████████▌| 381334/400000 [00:50<00:02, 7533.38it/s] 96%|█████████▌| 382102/400000 [00:50<00:02, 7575.83it/s] 96%|█████████▌| 382871/400000 [00:50<00:02, 7607.84it/s] 96%|█████████▌| 383633/400000 [00:50<00:02, 7610.13it/s] 96%|█████████▌| 384395/400000 [00:50<00:02, 7587.39it/s] 96%|█████████▋| 385154/400000 [00:50<00:02, 7352.51it/s] 96%|█████████▋| 385914/400000 [00:51<00:01, 7424.97it/s] 97%|█████████▋| 386683/400000 [00:51<00:01, 7501.08it/s] 97%|█████████▋| 387440/400000 [00:51<00:01, 7519.64it/s] 97%|█████████▋| 388200/400000 [00:51<00:01, 7541.98it/s] 97%|█████████▋| 388955/400000 [00:51<00:01, 7510.71it/s] 97%|█████████▋| 389727/400000 [00:51<00:01, 7571.48it/s] 98%|█████████▊| 390497/400000 [00:51<00:01, 7608.82it/s] 98%|█████████▊| 391259/400000 [00:51<00:01, 7601.75it/s] 98%|█████████▊| 392026/400000 [00:51<00:01, 7619.89it/s] 98%|█████████▊| 392798/400000 [00:51<00:00, 7648.80it/s] 98%|█████████▊| 393564/400000 [00:52<00:00, 7610.31it/s] 99%|█████████▊| 394331/400000 [00:52<00:00, 7625.96it/s] 99%|█████████▉| 395103/400000 [00:52<00:00, 7651.24it/s] 99%|█████████▉| 395869/400000 [00:52<00:00, 7589.79it/s] 99%|█████████▉| 396633/400000 [00:52<00:00, 7604.18it/s] 99%|█████████▉| 397394/400000 [00:52<00:00, 7491.17it/s]100%|█████████▉| 398169/400000 [00:52<00:00, 7564.13it/s]100%|█████████▉| 398936/400000 [00:52<00:00, 7593.78it/s]100%|█████████▉| 399701/400000 [00:52<00:00, 7609.52it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7565.64it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fed003070f0>, <torchtext.data.dataset.TabularDataset object at 0x7fed00307240>, <torchtext.vocab.Vocab object at 0x7fed00307160>), {}) 

