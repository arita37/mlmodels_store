
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc96dcb52f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc96dcb52f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc9dfd0a0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc9dfd0a0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc97749b9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc97749b9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc98d5ee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc98d5ee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc98d5ee8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2646016/9912422 [00:00<00:00, 26379367.25it/s]9920512it [00:00, 35285165.40it/s]                             
0it [00:00, ?it/s]32768it [00:00, 437627.02it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151709.34it/s]1654784it [00:00, 9596657.19it/s]                          
0it [00:00, ?it/s]8192it [00:00, 154712.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc96eb89be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc96eb89cc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc98d5ee510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc98d5ee510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc98d5ee510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:52,  3.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:52,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:52,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:52,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:51,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:51,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:35,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:35,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:35,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:34,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:34,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  5.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  5.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 11.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 15.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:06, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:06, 15.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 20.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 20.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 20.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 20.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 66.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 72.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 77.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 77.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 82.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 85.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.79 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ url]
0 examples [00:00, ? examples/s]2020-06-06 00:08:34.080538: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-06 00:08:34.094018: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-06 00:08:34.094201: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ebd5a352a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-06 00:08:34.094218: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
21 examples [00:00, 208.41 examples/s]124 examples [00:00, 273.91 examples/s]224 examples [00:00, 349.99 examples/s]332 examples [00:00, 438.75 examples/s]442 examples [00:00, 534.80 examples/s]549 examples [00:00, 629.03 examples/s]640 examples [00:00, 692.68 examples/s]733 examples [00:00, 748.52 examples/s]841 examples [00:00, 824.20 examples/s]953 examples [00:01, 893.64 examples/s]1056 examples [00:01, 927.92 examples/s]1169 examples [00:01, 979.35 examples/s]1274 examples [00:01, 991.36 examples/s]1385 examples [00:01, 1021.04 examples/s]1498 examples [00:01, 1049.09 examples/s]1610 examples [00:01, 1068.01 examples/s]1719 examples [00:01, 1042.82 examples/s]1825 examples [00:01, 1023.21 examples/s]1935 examples [00:01, 1043.42 examples/s]2043 examples [00:02, 1051.55 examples/s]2149 examples [00:02, 1050.28 examples/s]2255 examples [00:02, 1045.95 examples/s]2360 examples [00:02, 1028.22 examples/s]2468 examples [00:02, 1041.75 examples/s]2578 examples [00:02, 1058.08 examples/s]2685 examples [00:02, 1048.86 examples/s]2792 examples [00:02, 1054.48 examples/s]2898 examples [00:02, 1044.37 examples/s]3003 examples [00:02, 1026.12 examples/s]3106 examples [00:03, 1025.97 examples/s]3217 examples [00:03, 1048.89 examples/s]3327 examples [00:03, 1062.90 examples/s]3434 examples [00:03, 1049.93 examples/s]3541 examples [00:03, 1055.44 examples/s]3647 examples [00:03, 1048.38 examples/s]3758 examples [00:03, 1064.89 examples/s]3865 examples [00:03, 1064.91 examples/s]3972 examples [00:03, 1050.37 examples/s]4078 examples [00:03, 1004.89 examples/s]4185 examples [00:04, 1021.45 examples/s]4288 examples [00:04, 1018.21 examples/s]4391 examples [00:04, 982.13 examples/s] 4498 examples [00:04, 1004.82 examples/s]4609 examples [00:04, 1033.44 examples/s]4717 examples [00:04, 1045.93 examples/s]4823 examples [00:04, 1047.97 examples/s]4932 examples [00:04, 1059.87 examples/s]5043 examples [00:04, 1071.75 examples/s]5152 examples [00:05, 1075.65 examples/s]5263 examples [00:05, 1084.85 examples/s]5376 examples [00:05, 1096.71 examples/s]5486 examples [00:05, 1081.59 examples/s]5595 examples [00:05, 1079.77 examples/s]5704 examples [00:05, 1074.59 examples/s]5812 examples [00:05, 1072.56 examples/s]5920 examples [00:05, 1048.59 examples/s]6026 examples [00:05, 1026.07 examples/s]6134 examples [00:05, 1038.60 examples/s]6239 examples [00:06, 1033.99 examples/s]6343 examples [00:06, 1031.74 examples/s]6455 examples [00:06, 1055.05 examples/s]6561 examples [00:06, 1054.56 examples/s]6672 examples [00:06, 1069.04 examples/s]6784 examples [00:06, 1081.83 examples/s]6893 examples [00:06, 1058.51 examples/s]7000 examples [00:06, 1057.78 examples/s]7106 examples [00:06, 1041.78 examples/s]7213 examples [00:06, 1048.16 examples/s]7325 examples [00:07, 1067.15 examples/s]7437 examples [00:07, 1082.35 examples/s]7546 examples [00:07, 1064.78 examples/s]7653 examples [00:07, 1018.31 examples/s]7756 examples [00:07, 1006.49 examples/s]7866 examples [00:07, 1031.59 examples/s]7977 examples [00:07, 1052.20 examples/s]8087 examples [00:07, 1064.09 examples/s]8194 examples [00:07, 1050.64 examples/s]8300 examples [00:07, 1047.38 examples/s]8407 examples [00:08, 1051.85 examples/s]8517 examples [00:08, 1065.49 examples/s]8624 examples [00:08, 1052.85 examples/s]8730 examples [00:08, 1041.94 examples/s]8835 examples [00:08, 1033.78 examples/s]8942 examples [00:08, 1043.13 examples/s]9052 examples [00:08, 1059.41 examples/s]9162 examples [00:08, 1068.55 examples/s]9269 examples [00:08, 1013.97 examples/s]9372 examples [00:09, 946.54 examples/s] 9473 examples [00:09, 963.07 examples/s]9578 examples [00:09, 985.69 examples/s]9678 examples [00:09, 969.62 examples/s]9776 examples [00:09, 972.03 examples/s]9886 examples [00:09, 1004.98 examples/s]9991 examples [00:09, 1016.99 examples/s]10094 examples [00:09, 973.12 examples/s]10197 examples [00:09, 988.22 examples/s]10307 examples [00:09, 1017.22 examples/s]10416 examples [00:10, 1036.83 examples/s]10526 examples [00:10, 1052.81 examples/s]10636 examples [00:10, 1066.05 examples/s]10743 examples [00:10, 1058.83 examples/s]10850 examples [00:10, 1023.28 examples/s]10953 examples [00:10, 1012.97 examples/s]11065 examples [00:10, 1042.67 examples/s]11176 examples [00:10, 1060.44 examples/s]11283 examples [00:10, 1056.20 examples/s]11389 examples [00:11, 1032.41 examples/s]11493 examples [00:11, 1006.27 examples/s]11603 examples [00:11, 1030.31 examples/s]11712 examples [00:11, 1045.53 examples/s]11817 examples [00:11, 1037.54 examples/s]11928 examples [00:11, 1055.93 examples/s]12034 examples [00:11, 1049.66 examples/s]12144 examples [00:11, 1063.22 examples/s]12254 examples [00:11, 1073.03 examples/s]12362 examples [00:11, 1060.97 examples/s]12471 examples [00:12, 1067.38 examples/s]12578 examples [00:12, 1048.61 examples/s]12684 examples [00:12, 1043.79 examples/s]12794 examples [00:12, 1059.62 examples/s]12901 examples [00:12, 1037.86 examples/s]13005 examples [00:12, 1015.33 examples/s]13108 examples [00:12, 1019.30 examples/s]13218 examples [00:12, 1038.95 examples/s]13328 examples [00:12, 1054.38 examples/s]13434 examples [00:12, 1029.84 examples/s]13543 examples [00:13, 1045.92 examples/s]13656 examples [00:13, 1067.35 examples/s]13765 examples [00:13, 1072.87 examples/s]13873 examples [00:13, 1059.57 examples/s]13980 examples [00:13, 1039.25 examples/s]14085 examples [00:13, 1042.43 examples/s]14190 examples [00:13, 1037.03 examples/s]14294 examples [00:13, 1034.15 examples/s]14404 examples [00:13, 1052.67 examples/s]14510 examples [00:13, 1037.62 examples/s]14621 examples [00:14, 1057.27 examples/s]14734 examples [00:14, 1075.76 examples/s]14843 examples [00:14, 1079.33 examples/s]14952 examples [00:14, 1081.56 examples/s]15061 examples [00:14, 1074.28 examples/s]15176 examples [00:14, 1094.43 examples/s]15286 examples [00:14, 1089.76 examples/s]15398 examples [00:14, 1098.59 examples/s]15515 examples [00:14, 1117.48 examples/s]15627 examples [00:15, 1101.40 examples/s]15738 examples [00:15, 1086.80 examples/s]15851 examples [00:15, 1098.05 examples/s]15961 examples [00:15, 1060.42 examples/s]16068 examples [00:15, 1051.55 examples/s]16174 examples [00:15, 1025.49 examples/s]16284 examples [00:15, 1046.24 examples/s]16396 examples [00:15, 1067.20 examples/s]16505 examples [00:15, 1072.76 examples/s]16615 examples [00:15, 1077.70 examples/s]16723 examples [00:16, 1068.00 examples/s]16830 examples [00:16, 1058.28 examples/s]16936 examples [00:16, 1048.28 examples/s]17041 examples [00:16, 1043.06 examples/s]17146 examples [00:16, 1040.98 examples/s]17251 examples [00:16, 1043.51 examples/s]17366 examples [00:16, 1073.21 examples/s]17477 examples [00:16, 1082.87 examples/s]17586 examples [00:16, 1072.11 examples/s]17694 examples [00:16, 1068.45 examples/s]17801 examples [00:17, 1046.20 examples/s]17913 examples [00:17, 1067.17 examples/s]18025 examples [00:17, 1081.04 examples/s]18134 examples [00:17, 1081.94 examples/s]18243 examples [00:17, 1083.97 examples/s]18352 examples [00:17, 1066.75 examples/s]18459 examples [00:17, 1032.22 examples/s]18568 examples [00:17, 1048.16 examples/s]18679 examples [00:17, 1065.92 examples/s]18791 examples [00:17, 1080.89 examples/s]18900 examples [00:18, 1076.13 examples/s]19008 examples [00:18, 1058.73 examples/s]19115 examples [00:18, 1044.71 examples/s]19221 examples [00:18, 1047.01 examples/s]19327 examples [00:18, 1050.69 examples/s]19433 examples [00:18, 1033.71 examples/s]19540 examples [00:18, 1044.15 examples/s]19649 examples [00:18, 1057.45 examples/s]19758 examples [00:18, 1065.48 examples/s]19867 examples [00:19, 1071.63 examples/s]19975 examples [00:19, 1034.95 examples/s]20079 examples [00:19, 919.11 examples/s] 20177 examples [00:19, 934.71 examples/s]20277 examples [00:19, 951.87 examples/s]20381 examples [00:19, 975.24 examples/s]20492 examples [00:19, 1010.56 examples/s]20598 examples [00:19, 1024.89 examples/s]20706 examples [00:19, 1039.12 examples/s]20816 examples [00:19, 1056.31 examples/s]20923 examples [00:20, 1036.33 examples/s]21031 examples [00:20, 1047.39 examples/s]21140 examples [00:20, 1059.09 examples/s]21248 examples [00:20, 1064.15 examples/s]21355 examples [00:20, 1024.24 examples/s]21458 examples [00:20, 1010.76 examples/s]21570 examples [00:20, 1040.41 examples/s]21680 examples [00:20, 1054.79 examples/s]21789 examples [00:20, 1064.10 examples/s]21896 examples [00:20, 1051.76 examples/s]22002 examples [00:21, 1033.43 examples/s]22106 examples [00:21, 1008.45 examples/s]22218 examples [00:21, 1037.99 examples/s]22324 examples [00:21, 1043.82 examples/s]22437 examples [00:21, 1066.46 examples/s]22544 examples [00:21, 1065.83 examples/s]22651 examples [00:21, 1059.42 examples/s]22763 examples [00:21, 1074.59 examples/s]22874 examples [00:21, 1083.26 examples/s]22983 examples [00:22, 1078.65 examples/s]23091 examples [00:22, 1059.65 examples/s]23203 examples [00:22, 1074.84 examples/s]23313 examples [00:22, 1080.96 examples/s]23423 examples [00:22, 1085.63 examples/s]23532 examples [00:22, 1076.54 examples/s]23640 examples [00:22, 1037.51 examples/s]23749 examples [00:22, 1052.46 examples/s]23855 examples [00:22, 1051.95 examples/s]23961 examples [00:22, 1051.27 examples/s]24069 examples [00:23, 1058.53 examples/s]24175 examples [00:23, 1045.89 examples/s]24281 examples [00:23, 1048.12 examples/s]24387 examples [00:23, 1051.11 examples/s]24493 examples [00:23, 1052.69 examples/s]24602 examples [00:23, 1063.48 examples/s]24709 examples [00:23, 1053.72 examples/s]24819 examples [00:23, 1066.14 examples/s]24930 examples [00:23, 1078.33 examples/s]25038 examples [00:23, 1063.95 examples/s]25146 examples [00:24, 1063.64 examples/s]25253 examples [00:24, 1012.00 examples/s]25355 examples [00:24, 1012.26 examples/s]25457 examples [00:24, 1011.33 examples/s]25559 examples [00:24, 1012.94 examples/s]25668 examples [00:24, 1032.63 examples/s]25772 examples [00:24, 963.53 examples/s] 25875 examples [00:24, 980.52 examples/s]25983 examples [00:24, 1007.66 examples/s]26085 examples [00:25, 1009.27 examples/s]26195 examples [00:25, 1034.37 examples/s]26299 examples [00:25, 1026.66 examples/s]26407 examples [00:25, 1039.41 examples/s]26512 examples [00:25, 1038.99 examples/s]26617 examples [00:25, 1040.24 examples/s]26725 examples [00:25, 1051.10 examples/s]26831 examples [00:25, 1052.82 examples/s]26939 examples [00:25, 1057.67 examples/s]27045 examples [00:25, 1057.43 examples/s]27151 examples [00:26, 1057.08 examples/s]27257 examples [00:26, 1054.25 examples/s]27363 examples [00:26, 1035.86 examples/s]27471 examples [00:26, 1046.76 examples/s]27576 examples [00:26, 1023.20 examples/s]27679 examples [00:26, 1019.43 examples/s]27786 examples [00:26, 1029.20 examples/s]27890 examples [00:26, 1027.78 examples/s]27998 examples [00:26, 1041.61 examples/s]28103 examples [00:26, 1036.03 examples/s]28210 examples [00:27, 1044.94 examples/s]28319 examples [00:27, 1056.43 examples/s]28425 examples [00:27, 1035.79 examples/s]28531 examples [00:27, 1040.89 examples/s]28637 examples [00:27, 1044.51 examples/s]28746 examples [00:27, 1056.18 examples/s]28854 examples [00:27, 1061.50 examples/s]28961 examples [00:27, 1050.50 examples/s]29070 examples [00:27, 1059.08 examples/s]29176 examples [00:27, 1007.93 examples/s]29278 examples [00:28, 1000.74 examples/s]29379 examples [00:28, 994.74 examples/s] 29486 examples [00:28, 1013.95 examples/s]29588 examples [00:28, 1006.18 examples/s]29689 examples [00:28, 972.25 examples/s] 29787 examples [00:28, 960.30 examples/s]29884 examples [00:28, 963.08 examples/s]29982 examples [00:28, 968.06 examples/s]30079 examples [00:28, 940.59 examples/s]30187 examples [00:29, 978.18 examples/s]30296 examples [00:29, 1006.70 examples/s]30400 examples [00:29, 1015.53 examples/s]30503 examples [00:29, 1007.52 examples/s]30605 examples [00:29, 960.26 examples/s] 30702 examples [00:29, 926.56 examples/s]30796 examples [00:29, 910.78 examples/s]30898 examples [00:29, 939.79 examples/s]31002 examples [00:29, 967.55 examples/s]31111 examples [00:29, 1000.67 examples/s]31221 examples [00:30, 1027.04 examples/s]31333 examples [00:30, 1052.29 examples/s]31444 examples [00:30, 1068.49 examples/s]31555 examples [00:30, 1078.92 examples/s]31665 examples [00:30, 1083.85 examples/s]31774 examples [00:30, 1081.49 examples/s]31883 examples [00:30, 1041.13 examples/s]31988 examples [00:30, 995.09 examples/s] 32089 examples [00:30, 993.22 examples/s]32189 examples [00:30, 984.74 examples/s]32292 examples [00:31, 997.35 examples/s]32393 examples [00:31, 995.71 examples/s]32494 examples [00:31, 998.13 examples/s]32606 examples [00:31, 1029.99 examples/s]32715 examples [00:31, 1045.67 examples/s]32826 examples [00:31, 1062.96 examples/s]32933 examples [00:31, 1063.46 examples/s]33040 examples [00:31, 1055.67 examples/s]33148 examples [00:31, 1060.76 examples/s]33257 examples [00:32, 1068.25 examples/s]33364 examples [00:32, 1035.63 examples/s]33470 examples [00:32, 1042.33 examples/s]33577 examples [00:32, 1048.31 examples/s]33685 examples [00:32, 1056.00 examples/s]33794 examples [00:32, 1065.08 examples/s]33901 examples [00:32, 1063.69 examples/s]34008 examples [00:32, 974.10 examples/s] 34112 examples [00:32, 991.95 examples/s]34223 examples [00:32, 1022.17 examples/s]34336 examples [00:33, 1050.64 examples/s]34442 examples [00:33, 1051.21 examples/s]34548 examples [00:33, 1020.47 examples/s]34653 examples [00:33, 1028.59 examples/s]34761 examples [00:33, 1043.12 examples/s]34869 examples [00:33, 1051.76 examples/s]34977 examples [00:33, 1060.06 examples/s]35084 examples [00:33, 1054.19 examples/s]35190 examples [00:33, 1055.92 examples/s]35296 examples [00:33, 1045.78 examples/s]35404 examples [00:34, 1055.24 examples/s]35517 examples [00:34, 1074.09 examples/s]35625 examples [00:34, 1075.12 examples/s]35733 examples [00:34, 1042.82 examples/s]35845 examples [00:34, 1062.35 examples/s]35957 examples [00:34, 1078.26 examples/s]36069 examples [00:34, 1086.52 examples/s]36182 examples [00:34, 1097.29 examples/s]36292 examples [00:34, 1079.52 examples/s]36404 examples [00:34, 1088.67 examples/s]36514 examples [00:35, 991.48 examples/s] 36615 examples [00:35, 980.65 examples/s]36715 examples [00:35, 952.90 examples/s]36814 examples [00:35, 960.78 examples/s]36911 examples [00:35, 959.66 examples/s]37016 examples [00:35, 983.62 examples/s]37115 examples [00:35, 979.48 examples/s]37214 examples [00:35, 979.83 examples/s]37313 examples [00:35, 935.98 examples/s]37410 examples [00:36, 945.47 examples/s]37518 examples [00:36, 980.65 examples/s]37618 examples [00:36, 984.85 examples/s]37726 examples [00:36, 1011.06 examples/s]37831 examples [00:36, 1021.68 examples/s]37939 examples [00:36, 1038.46 examples/s]38048 examples [00:36, 1052.37 examples/s]38154 examples [00:36, 1047.89 examples/s]38259 examples [00:36, 1047.30 examples/s]38365 examples [00:36, 1050.08 examples/s]38471 examples [00:37, 1048.65 examples/s]38583 examples [00:37, 1068.34 examples/s]38695 examples [00:37, 1082.87 examples/s]38805 examples [00:37, 1086.39 examples/s]38914 examples [00:37, 1066.44 examples/s]39023 examples [00:37, 1072.99 examples/s]39131 examples [00:37, 1053.85 examples/s]39239 examples [00:37, 1059.01 examples/s]39346 examples [00:37, 1048.32 examples/s]39452 examples [00:37, 1051.53 examples/s]39558 examples [00:38, 1041.98 examples/s]39668 examples [00:38, 1055.87 examples/s]39774 examples [00:38, 1042.85 examples/s]39880 examples [00:38, 1046.17 examples/s]39985 examples [00:38, 1041.49 examples/s]40090 examples [00:38, 1011.02 examples/s]40202 examples [00:38, 1041.30 examples/s]40315 examples [00:38, 1063.78 examples/s]40426 examples [00:38, 1074.32 examples/s]40534 examples [00:39, 1061.42 examples/s]40641 examples [00:39, 1052.84 examples/s]40752 examples [00:39, 1068.62 examples/s]40862 examples [00:39, 1071.66 examples/s]40973 examples [00:39, 1081.17 examples/s]41083 examples [00:39, 1084.18 examples/s]41192 examples [00:39, 1041.98 examples/s]41297 examples [00:39, 1010.40 examples/s]41399 examples [00:39, 996.78 examples/s] 41500 examples [00:39, 997.03 examples/s]41605 examples [00:40, 1009.95 examples/s]41714 examples [00:40, 1032.67 examples/s]41826 examples [00:40, 1056.04 examples/s]41934 examples [00:40, 1061.74 examples/s]42041 examples [00:40, 1052.83 examples/s]42148 examples [00:40, 1056.85 examples/s]42254 examples [00:40, 1037.95 examples/s]42363 examples [00:40, 1051.79 examples/s]42474 examples [00:40, 1067.12 examples/s]42581 examples [00:40, 1055.01 examples/s]42692 examples [00:41, 1068.54 examples/s]42804 examples [00:41, 1081.41 examples/s]42913 examples [00:41, 1083.74 examples/s]43024 examples [00:41, 1090.63 examples/s]43134 examples [00:41, 1078.76 examples/s]43243 examples [00:41, 1081.92 examples/s]43354 examples [00:41, 1089.58 examples/s]43466 examples [00:41, 1096.53 examples/s]43576 examples [00:41, 1097.39 examples/s]43686 examples [00:41, 1084.36 examples/s]43796 examples [00:42, 1086.08 examples/s]43907 examples [00:42, 1092.77 examples/s]44017 examples [00:42, 1072.23 examples/s]44129 examples [00:42, 1085.82 examples/s]44238 examples [00:42, 1078.53 examples/s]44349 examples [00:42, 1087.18 examples/s]44460 examples [00:42, 1091.85 examples/s]44570 examples [00:42, 1086.65 examples/s]44679 examples [00:42, 1086.79 examples/s]44788 examples [00:43, 1057.48 examples/s]44894 examples [00:43, 1051.78 examples/s]45000 examples [00:43, 1024.36 examples/s]45111 examples [00:43, 1048.03 examples/s]45220 examples [00:43, 1057.23 examples/s]45326 examples [00:43, 1039.81 examples/s]45431 examples [00:43, 1032.04 examples/s]45541 examples [00:43, 1050.54 examples/s]45648 examples [00:43, 1054.14 examples/s]45761 examples [00:43, 1075.63 examples/s]45869 examples [00:44, 1057.42 examples/s]45978 examples [00:44, 1064.35 examples/s]46086 examples [00:44, 1067.24 examples/s]46197 examples [00:44, 1077.85 examples/s]46311 examples [00:44, 1093.22 examples/s]46421 examples [00:44, 1083.00 examples/s]46531 examples [00:44, 1087.17 examples/s]46640 examples [00:44, 1076.40 examples/s]46750 examples [00:44, 1081.37 examples/s]46862 examples [00:44, 1089.91 examples/s]46972 examples [00:45, 1079.61 examples/s]47083 examples [00:45, 1087.10 examples/s]47192 examples [00:45, 1082.85 examples/s]47301 examples [00:45, 1074.61 examples/s]47409 examples [00:45, 1052.22 examples/s]47515 examples [00:45, 1033.80 examples/s]47629 examples [00:45, 1060.97 examples/s]47739 examples [00:45, 1070.00 examples/s]47847 examples [00:45, 1072.19 examples/s]47955 examples [00:45, 1068.04 examples/s]48062 examples [00:46, 1044.77 examples/s]48167 examples [00:46, 1025.63 examples/s]48273 examples [00:46, 1035.35 examples/s]48377 examples [00:46, 1025.42 examples/s]48481 examples [00:46, 1028.80 examples/s]48584 examples [00:46, 1028.24 examples/s]48693 examples [00:46, 1044.09 examples/s]48798 examples [00:46, 1044.33 examples/s]48903 examples [00:46, 1039.86 examples/s]49008 examples [00:47, 1029.70 examples/s]49112 examples [00:47, 988.11 examples/s] 49217 examples [00:47, 1003.86 examples/s]49326 examples [00:47, 1026.65 examples/s]49436 examples [00:47, 1046.90 examples/s]49550 examples [00:47, 1070.58 examples/s]49658 examples [00:47, 1054.55 examples/s]49769 examples [00:47, 1068.14 examples/s]49880 examples [00:47, 1078.77 examples/s]49991 examples [00:47, 1086.25 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8397/50000 [00:00<00:00, 83966.48 examples/s] 43%|████▎     | 21672/50000 [00:00<00:00, 94370.09 examples/s] 70%|███████   | 35142/50000 [00:00<00:00, 103682.01 examples/s] 97%|█████████▋| 48567/50000 [00:00<00:00, 111282.65 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 896.61 examples/s]184 examples [00:00, 908.10 examples/s]275 examples [00:00, 906.45 examples/s]382 examples [00:00, 949.30 examples/s]490 examples [00:00, 983.80 examples/s]595 examples [00:00, 1001.13 examples/s]697 examples [00:00, 838.42 examples/s] 791 examples [00:00, 865.78 examples/s]906 examples [00:00, 935.06 examples/s]1011 examples [00:01, 964.99 examples/s]1121 examples [00:01, 999.60 examples/s]1225 examples [00:01, 1010.25 examples/s]1331 examples [00:01, 1023.64 examples/s]1434 examples [00:01, 1001.28 examples/s]1535 examples [00:01, 980.78 examples/s] 1634 examples [00:01, 976.10 examples/s]1732 examples [00:01, 974.38 examples/s]1839 examples [00:01, 999.27 examples/s]1940 examples [00:02, 969.77 examples/s]2044 examples [00:02, 988.48 examples/s]2153 examples [00:02, 1015.71 examples/s]2258 examples [00:02, 1025.01 examples/s]2368 examples [00:02, 1046.35 examples/s]2477 examples [00:02, 1058.51 examples/s]2584 examples [00:02, 1030.29 examples/s]2695 examples [00:02, 1052.63 examples/s]2801 examples [00:02, 1054.66 examples/s]2912 examples [00:02, 1069.94 examples/s]3024 examples [00:03, 1081.60 examples/s]3134 examples [00:03, 1086.05 examples/s]3243 examples [00:03, 1061.84 examples/s]3350 examples [00:03, 1038.07 examples/s]3455 examples [00:03, 1034.93 examples/s]3565 examples [00:03, 1052.57 examples/s]3675 examples [00:03, 1065.00 examples/s]3782 examples [00:03, 1056.84 examples/s]3888 examples [00:03, 1028.13 examples/s]3992 examples [00:03, 1006.07 examples/s]4099 examples [00:04, 1022.53 examples/s]4202 examples [00:04, 1023.23 examples/s]4305 examples [00:04, 1011.48 examples/s]4407 examples [00:04, 976.07 examples/s] 4507 examples [00:04, 981.23 examples/s]4609 examples [00:04, 991.26 examples/s]4714 examples [00:04, 1006.23 examples/s]4817 examples [00:04, 1010.71 examples/s]4919 examples [00:04, 1010.57 examples/s]5021 examples [00:04, 1005.85 examples/s]5126 examples [00:05, 1016.02 examples/s]5235 examples [00:05, 1036.38 examples/s]5340 examples [00:05, 1038.38 examples/s]5444 examples [00:05, 1031.70 examples/s]5548 examples [00:05, 1033.78 examples/s]5654 examples [00:05, 1039.47 examples/s]5758 examples [00:05, 1015.70 examples/s]5861 examples [00:05, 1018.89 examples/s]5966 examples [00:05, 1026.93 examples/s]6070 examples [00:05, 1029.46 examples/s]6177 examples [00:06, 1038.88 examples/s]6283 examples [00:06, 1042.42 examples/s]6388 examples [00:06, 1020.72 examples/s]6491 examples [00:06, 1021.61 examples/s]6594 examples [00:06, 1019.42 examples/s]6697 examples [00:06, 1019.58 examples/s]6806 examples [00:06, 1037.34 examples/s]6913 examples [00:06, 1043.67 examples/s]7018 examples [00:06, 1038.06 examples/s]7123 examples [00:07, 1038.55 examples/s]7232 examples [00:07, 1052.77 examples/s]7339 examples [00:07, 1057.35 examples/s]7445 examples [00:07, 1049.90 examples/s]7552 examples [00:07, 1055.25 examples/s]7660 examples [00:07, 1061.70 examples/s]7771 examples [00:07, 1075.40 examples/s]7881 examples [00:07, 1081.54 examples/s]7990 examples [00:07, 1068.15 examples/s]8097 examples [00:07, 1068.59 examples/s]8205 examples [00:08, 1071.21 examples/s]8313 examples [00:08, 1051.10 examples/s]8420 examples [00:08, 1054.01 examples/s]8526 examples [00:08, 1030.32 examples/s]8635 examples [00:08, 1045.31 examples/s]8745 examples [00:08, 1059.76 examples/s]8852 examples [00:08, 1056.93 examples/s]8961 examples [00:08, 1065.08 examples/s]9068 examples [00:08, 1051.53 examples/s]9174 examples [00:08, 1039.92 examples/s]9282 examples [00:09, 1049.66 examples/s]9388 examples [00:09, 1030.60 examples/s]9498 examples [00:09, 1048.79 examples/s]9604 examples [00:09, 1043.63 examples/s]9712 examples [00:09, 1052.66 examples/s]9819 examples [00:09, 1057.45 examples/s]9927 examples [00:09, 1061.13 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete92OEG3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete92OEG3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc98d5ee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc98d5ee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc98d5ee8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc9f94675c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc96c1fccf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc98d5ee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc98d5ee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc98d5ee8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc96c17e4a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc9712fd320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc96410d510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc96410d510> 

  function with postional parmater data_info <function split_train_valid at 0x7fc96410d510> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc96410d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc96410d620> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc96410d620> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=84f53796b1d108671066b5c9b8278f149cffe650e5d245bfffc976dd8ca59b69
  Stored in directory: /tmp/pip-ephem-wheel-cache-8zkg4cfr/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<141:47:44, 1.69kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<99:29:08, 2.41kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<69:40:54, 3.44kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:05<48:45:17, 4.91kB/s].vector_cache/glove.6B.zip:   0%|          | 3.56M/862M [00:05<34:01:33, 7.01kB/s].vector_cache/glove.6B.zip:   1%|          | 9.83M/862M [00:05<23:38:43, 10.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:05<16:26:21, 14.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:05<11:27:54, 20.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.8M/862M [00:05<7:58:19, 29.2kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:06<5:32:40, 41.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:06<3:51:22, 59.5kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:06<2:40:53, 85.0kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.3M/862M [00:06<1:51:58, 121kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:06<1:18:11, 173kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:06<55:51, 242kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:08<40:49, 329kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:09<32:13, 417kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:09<23:22, 574kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:09<16:30, 811kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:10<16:28, 811kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:10<12:41, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:11<09:12, 1.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:12<09:31, 1.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:12<08:01, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:13<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:14<07:17, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:14<06:26, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:15<04:50, 2.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:16<06:29, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<05:53, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:17<04:26, 2.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:18<06:12, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<05:40, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:19<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:20<06:05, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:20<06:54, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:20<05:22, 2.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:21<03:57, 3.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:22<08:19, 1.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<07:10, 1.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:22<05:17, 2.44MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:24<06:44, 1.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:24<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:24<04:32, 2.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:26<06:12, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<06:57, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:26<05:30, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:28<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:28<05:25, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:28<04:04, 3.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:49, 2.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<06:39, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<05:14, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<03:47, 3.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<25:51, 487kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<19:24, 649kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:32<13:52, 906kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<12:38, 992kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<10:07, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<07:23, 1.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<08:07, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<08:13, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<06:17, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<04:32, 2.73MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<10:31, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<08:39, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<06:21, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<07:20, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<07:39, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<05:58, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<06:09, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<05:33, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<04:09, 2.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<05:46, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<05:18, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<04:00, 3.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:46<05:40, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:12, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<03:56, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<05:36, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<06:22, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<05:04, 2.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<03:40, 3.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<11:36:51, 17.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<8:08:45, 24.5kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<5:41:39, 35.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<4:01:14, 49.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<2:51:15, 69.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<2:00:21, 98.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<1:25:48, 138kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<1:01:15, 193kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<43:06, 274kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<32:48, 359kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<25:22, 464kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<18:20, 641kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<14:41, 798kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<11:16, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<08:08, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<05:50, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<33:27, 348kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<25:47, 452kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<18:31, 628kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<13:04, 887kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<17:29, 662kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<13:25, 862kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<09:37, 1.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<09:26, 1.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<07:47, 1.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<05:43, 2.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<06:42, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<05:51, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<04:22, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:45, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:11, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<03:54, 2.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<05:24, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:07, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<04:50, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:11, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<04:48, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<03:36, 3.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:07, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:51, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<04:34, 2.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<03:21, 3.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<07:10, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<06:10, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<04:35, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<05:48, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<06:18, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<04:52, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<03:33, 3.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<08:07, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<06:48, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<05:02, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<06:04, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<06:23, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<05:01, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<05:15, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<04:48, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<03:37, 2.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<05:03, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<05:43, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<04:33, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<04:54, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<03:25, 3.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<04:54, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<04:33, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<03:25, 3.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<04:52, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<04:18, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<03:16, 3.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<04:46, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<05:31, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<04:18, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<03:10, 3.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:35<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<05:21, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<03:58, 2.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:37<05:12, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:46, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<04:28, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<03:23, 3.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:39<04:54, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<04:43, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<03:28, 2.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<05:39, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<06:03, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<05:30, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<04:05, 2.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<03:02, 3.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<08:22, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<08:21, 1.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<06:23, 1.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<04:37, 2.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<09:11, 1.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<08:56, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<06:50, 1.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<04:55, 2.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<09:33, 1.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<09:07, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<06:58, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:47<05:01, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:49<09:35, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<09:01, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<06:55, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<04:57, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<09:18, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<08:48, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<06:39, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<04:46, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<07:47, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<07:45, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<05:56, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<04:17, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<06:40, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<07:02, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:28, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<03:56, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<07:30, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<07:38, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<05:55, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<04:16, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<08:50, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<08:35, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<06:30, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<04:49, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<03:29, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<38:02, 250kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<28:49, 330kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<20:44, 458kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<14:34, 649kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<15:05, 625kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<12:45, 740kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<09:29, 993kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<06:43, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<09:59, 938kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<09:24, 996kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<07:06, 1.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<05:11, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<05:45, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<06:19, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<03:39, 2.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<04:39, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<05:25, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<03:09, 2.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<07:29, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<07:23, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<05:43, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<05:47, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<06:09, 1.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<04:50, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<03:29, 2.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<08:16, 1.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<07:48, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<05:53, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<04:14, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<06:49, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<06:58, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<05:20, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<03:51, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<05:44, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<06:18, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<04:53, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<03:31, 2.51MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<06:33, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<06:27, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<04:55, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<03:32, 2.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<07:32, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<07:09, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<05:23, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<03:52, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<06:32, 1.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<06:42, 1.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<05:08, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<03:43, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<05:51, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<06:06, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<04:44, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<03:25, 2.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<07:56, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<07:33, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<05:41, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<04:05, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<06:16, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<06:21, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:52, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<03:31, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<05:44, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<05:58, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<04:40, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<03:23, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<07:27, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<07:04, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<05:23, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<03:52, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<08:16, 998kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<07:13, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<05:25, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<03:53, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<2:00:04, 68.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<1:25:29, 95.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<1:00:05, 136kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<43:11, 188kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<32:06, 253kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<22:55, 354kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<16:04, 502kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<17:55, 449kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<13:51, 581kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<10:00, 803kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<08:22, 954kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<07:33, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<05:38, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<04:03, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:59, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:48, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:46<04:28, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<03:12, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<32:28, 242kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<24:18, 323kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<17:23, 451kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<12:11, 638kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<32:04, 243kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<24:01, 324kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<17:11, 452kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<12:03, 640kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<42:18, 182kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<31:14, 247kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<22:14, 346kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<15:34, 491kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<35:08, 218kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<25:38, 298kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<18:08, 420kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<14:03, 539kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<12:41, 597kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<09:28, 798kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<06:46, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<04:50, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<17:55, 419kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<13:51, 542kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<10:01, 748kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<08:13, 905kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<07:08, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<05:17, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:47, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<06:36, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<06:02, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<04:33, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<03:21, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:10, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<03:20, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<03:31, 2.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:49, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:00, 2.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<03:17, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<03:38, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<02:52, 2.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:11, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:33, 1.99MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<02:49, 2.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<03:07, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<03:31, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:12<02:47, 2.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<03:05, 2.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<03:29, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<02:45, 2.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<03:07, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<04:47, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<03:54, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<02:52, 2.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<03:29, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<03:43, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<02:55, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<03:09, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<03:28, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<02:44, 2.45MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:22<03:01, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:22<03:22, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<02:40, 2.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:57, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<03:18, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:37, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<02:54, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:26<03:16, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<02:33, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<01:51, 3.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<05:25, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:28<05:00, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<03:45, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:41, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<06:15, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<05:34, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<04:11, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<03:04, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<03:10, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:34<03:24, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<02:40, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<02:53, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:08, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<02:29, 2.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<02:45, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<02:24, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<01:44, 3.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<51:52, 117kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<37:26, 162kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<26:23, 229kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<18:24, 326kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<18:20, 327kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<13:57, 429kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<10:01, 596kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<07:56, 746kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<06:21, 932kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<04:37, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<04:21, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:48, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<02:50, 2.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:07, 1.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<02:28, 2.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<02:42, 2.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<02:53, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<02:15, 2.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<02:32, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<02:10, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<02:28, 2.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<02:40, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:53<02:04, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<01:30, 3.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<05:51, 940kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<05:03, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:55<03:44, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<03:32, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<03:25, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<02:37, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<02:44, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:59<02:11, 2.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<01:35, 3.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<05:33, 956kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<04:49, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<03:33, 1.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<02:32, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<06:17, 833kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<05:17, 990kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<03:55, 1.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<03:36, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<03:25, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<02:34, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<01:51, 2.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:07<04:09, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:07<03:47, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<02:52, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:09<02:51, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<02:52, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<02:13, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:11<02:23, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:11<02:30, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<01:57, 2.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:12, 2.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:24, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<01:53, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:15<02:20, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<01:48, 2.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<01:18, 3.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<04:29, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<03:58, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:58, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<02:52, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<02:48, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:19<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:21<02:17, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:21<02:24, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:21<01:52, 2.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:23<02:04, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:23<02:14, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:23<01:44, 2.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:25<01:58, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<02:10, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:25<01:42, 2.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<01:14, 3.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:27<04:53, 901kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:27<03:59, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:27<02:54, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<02:52, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:29<02:46, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<02:05, 2.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<01:31, 2.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<03:06, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:31<02:53, 1.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<02:12, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<02:14, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:33<02:16, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:33<01:46, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:35<01:53, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:35<01:26, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<01:48, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:37<02:41, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:37<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:38, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:14, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:15, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:39<01:44, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<01:53, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<01:59, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:41<01:32, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<01:43, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<01:51, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<01:26, 2.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:45<01:24, 2.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<01:37, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:24, 2.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:36, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:18, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:55, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<01:25, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:58, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:59, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:30, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<01:05, 3.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<03:35, 981kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<03:06, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<02:18, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:11, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:07, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:36, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<01:10, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<02:16, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<02:10, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<01:39, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<01:45, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<01:20, 2.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<00:58, 3.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<03:28, 934kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<02:58, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<02:12, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:04, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:58, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<01:30, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:34, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:38, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<01:16, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:24, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:30, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:10, 2.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<01:19, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:06, 2.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<01:15, 2.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<01:22, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:04, 2.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:21, 2.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<01:03, 2.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<01:12, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<01:18, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:00, 2.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<00:43, 3.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<06:38, 405kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<05:06, 527kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<03:39, 731kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<02:32, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<17:17, 152kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<12:31, 209kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<08:49, 295kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<06:30, 393kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<04:58, 512kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<03:33, 712kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<02:28, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<08:44, 284kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<06:33, 379kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<04:39, 529kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<03:36, 668kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<02:56, 822kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:08, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:26<01:52, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<01:42, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:26<01:17, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:28<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<00:59, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<01:03, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<01:07, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<00:52, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<00:58, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<01:02, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<00:48, 2.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:54, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:34<00:59, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<00:45, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:52, 2.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:36<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<00:44, 2.70MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:36<00:31, 3.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<02:36, 742kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:38<02:04, 933kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<01:29, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:23, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:37, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<01:17, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:55, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<01:07, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:00, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<00:51, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:53, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<00:40, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<00:29, 3.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<01:33, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<01:21, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<01:00, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:57, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:56, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<00:42, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<00:44, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:36, 2.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:39, 2.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:42, 2.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:32, 2.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:23, 3.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<01:16, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<01:07, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:53<00:50, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:47, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:46, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:35, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:36, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<00:38, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:57<00:29, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:32, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:34, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:26, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:29, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:31, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:24, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:17, 3.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<01:11, 870kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<01:00, 1.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:44, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:39, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:48, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:37, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:26, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<00:32, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<00:32, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:24, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:24, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:25, 1.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:19, 2.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:11<00:20, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:22, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:17, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:13<00:18, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:15, 2.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:16, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:17, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:13, 2.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:14, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:12, 2.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<00:12, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:10, 2.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<00:10, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<00:16, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:13, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:09, 2.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:23<00:11, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:23<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:25<00:08, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:25<00:08, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:06, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:27<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:06, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:04, 2.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:29<00:03, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:04, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:02, 2.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:31<00:02, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:31<00:02, 2.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:01, 2.66MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:33<00:00, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:33<00:00, 2.09MB/s].vector_cache/glove.6B.zip: 862MB [06:33, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<108:59:56,  1.02it/s]  0%|          | 858/400000 [00:01<76:08:23,  1.46it/s]  0%|          | 1785/400000 [00:01<53:10:39,  2.08it/s]  1%|          | 2632/400000 [00:01<37:08:56,  2.97it/s]  1%|          | 3456/400000 [00:01<25:57:16,  4.24it/s]  1%|          | 4228/400000 [00:01<18:08:13,  6.06it/s]  1%|▏         | 5047/400000 [00:01<12:40:25,  8.66it/s]  2%|▏         | 6006/400000 [00:01<8:51:12, 12.36it/s]   2%|▏         | 6836/400000 [00:01<6:11:17, 17.65it/s]  2%|▏         | 7769/400000 [00:01<4:19:30, 25.19it/s]  2%|▏         | 8705/400000 [00:01<3:01:25, 35.95it/s]  2%|▏         | 9617/400000 [00:02<2:06:55, 51.26it/s]  3%|▎         | 10540/400000 [00:02<1:28:50, 73.06it/s]  3%|▎         | 11435/400000 [00:02<1:02:17, 103.97it/s]  3%|▎         | 12345/400000 [00:02<43:42, 147.80it/s]    3%|▎         | 13250/400000 [00:02<30:44, 209.68it/s]  4%|▎         | 14180/400000 [00:02<21:40, 296.67it/s]  4%|▍         | 15120/400000 [00:02<15:20, 418.16it/s]  4%|▍         | 16032/400000 [00:02<10:56, 585.15it/s]  4%|▍         | 16975/400000 [00:02<07:50, 814.26it/s]  4%|▍         | 17888/400000 [00:02<05:41, 1120.41it/s]  5%|▍         | 18799/400000 [00:03<04:11, 1517.00it/s]  5%|▍         | 19743/400000 [00:03<03:07, 2027.39it/s]  5%|▌         | 20656/400000 [00:03<02:24, 2620.06it/s]  5%|▌         | 21572/400000 [00:03<01:53, 3334.21it/s]  6%|▌         | 22501/400000 [00:03<01:31, 4127.90it/s]  6%|▌         | 23440/400000 [00:03<01:15, 4961.77it/s]  6%|▌         | 24356/400000 [00:03<01:05, 5715.62it/s]  6%|▋         | 25263/400000 [00:03<00:58, 6394.17it/s]  7%|▋         | 26201/400000 [00:03<00:52, 7067.77it/s]  7%|▋         | 27112/400000 [00:04<00:49, 7567.15it/s]  7%|▋         | 28048/400000 [00:04<00:46, 8026.57it/s]  7%|▋         | 28968/400000 [00:04<00:44, 8344.27it/s]  7%|▋         | 29886/400000 [00:04<00:43, 8521.22it/s]  8%|▊         | 30798/400000 [00:04<00:42, 8687.99it/s]  8%|▊         | 31742/400000 [00:04<00:41, 8899.95it/s]  8%|▊         | 32681/400000 [00:04<00:40, 9040.76it/s]  8%|▊         | 33608/400000 [00:04<00:40, 9053.42it/s]  9%|▊         | 34529/400000 [00:04<00:40, 8958.46it/s]  9%|▉         | 35440/400000 [00:04<00:40, 9001.56it/s]  9%|▉         | 36348/400000 [00:05<00:40, 9017.93it/s]  9%|▉         | 37256/400000 [00:05<00:40, 9008.29it/s] 10%|▉         | 38161/400000 [00:05<00:40, 8878.87it/s] 10%|▉         | 39052/400000 [00:05<00:41, 8751.56it/s] 10%|▉         | 39966/400000 [00:05<00:40, 8864.08it/s] 10%|█         | 40860/400000 [00:05<00:40, 8884.16it/s] 10%|█         | 41750/400000 [00:05<00:40, 8855.80it/s] 11%|█         | 42676/400000 [00:05<00:39, 8972.82it/s] 11%|█         | 43575/400000 [00:05<00:39, 8961.59it/s] 11%|█         | 44517/400000 [00:05<00:39, 9092.13it/s] 11%|█▏        | 45442/400000 [00:06<00:38, 9137.95it/s] 12%|█▏        | 46357/400000 [00:06<00:39, 9002.83it/s] 12%|█▏        | 47259/400000 [00:06<00:39, 8943.34it/s] 12%|█▏        | 48155/400000 [00:06<00:39, 8906.04it/s] 12%|█▏        | 49053/400000 [00:06<00:39, 8926.91it/s] 12%|█▏        | 49947/400000 [00:06<00:39, 8875.40it/s] 13%|█▎        | 50835/400000 [00:06<00:39, 8765.46it/s] 13%|█▎        | 51776/400000 [00:06<00:38, 8948.14it/s] 13%|█▎        | 52673/400000 [00:06<00:38, 8952.45it/s] 13%|█▎        | 53606/400000 [00:06<00:38, 9061.49it/s] 14%|█▎        | 54564/400000 [00:07<00:37, 9209.08it/s] 14%|█▍        | 55487/400000 [00:07<00:38, 9011.22it/s] 14%|█▍        | 56441/400000 [00:07<00:37, 9160.68it/s] 14%|█▍        | 57359/400000 [00:07<00:37, 9059.59it/s] 15%|█▍        | 58291/400000 [00:07<00:37, 9133.65it/s] 15%|█▍        | 59206/400000 [00:07<00:37, 9126.10it/s] 15%|█▌        | 60135/400000 [00:07<00:37, 9173.93it/s] 15%|█▌        | 61054/400000 [00:07<00:36, 9177.61it/s] 15%|█▌        | 61973/400000 [00:07<00:37, 9116.14it/s] 16%|█▌        | 62894/400000 [00:07<00:36, 9143.68it/s] 16%|█▌        | 63809/400000 [00:08<00:36, 9089.25it/s] 16%|█▌        | 64719/400000 [00:08<00:36, 9083.24it/s] 16%|█▋        | 65660/400000 [00:08<00:36, 9176.74it/s] 17%|█▋        | 66579/400000 [00:08<00:36, 9080.16it/s] 17%|█▋        | 67518/400000 [00:08<00:36, 9166.77it/s] 17%|█▋        | 68442/400000 [00:08<00:36, 9187.91it/s] 17%|█▋        | 69369/400000 [00:08<00:35, 9212.18it/s] 18%|█▊        | 70291/400000 [00:08<00:36, 9151.14it/s] 18%|█▊        | 71207/400000 [00:08<00:36, 8922.97it/s] 18%|█▊        | 72118/400000 [00:08<00:36, 8976.11it/s] 18%|█▊        | 73046/400000 [00:09<00:36, 9064.96it/s] 18%|█▊        | 73987/400000 [00:09<00:35, 9164.63it/s] 19%|█▊        | 74926/400000 [00:09<00:35, 9228.40it/s] 19%|█▉        | 75850/400000 [00:09<00:36, 8993.98it/s] 19%|█▉        | 76752/400000 [00:09<00:35, 9000.87it/s] 19%|█▉        | 77698/400000 [00:09<00:35, 9131.29it/s] 20%|█▉        | 78620/400000 [00:09<00:35, 9157.34it/s] 20%|█▉        | 79549/400000 [00:09<00:34, 9192.24it/s] 20%|██        | 80469/400000 [00:09<00:35, 9071.27it/s] 20%|██        | 81377/400000 [00:10<00:35, 8930.25it/s] 21%|██        | 82300/400000 [00:10<00:35, 9017.04it/s] 21%|██        | 83203/400000 [00:10<00:35, 9020.11it/s] 21%|██        | 84106/400000 [00:10<00:35, 8849.73it/s] 21%|██        | 84993/400000 [00:10<00:36, 8609.34it/s] 21%|██▏       | 85857/400000 [00:10<00:36, 8589.01it/s] 22%|██▏       | 86773/400000 [00:10<00:35, 8751.06it/s] 22%|██▏       | 87722/400000 [00:10<00:34, 8959.75it/s] 22%|██▏       | 88621/400000 [00:10<00:34, 8934.70it/s] 22%|██▏       | 89517/400000 [00:10<00:34, 8927.03it/s] 23%|██▎       | 90451/400000 [00:11<00:34, 9043.57it/s] 23%|██▎       | 91402/400000 [00:11<00:33, 9177.22it/s] 23%|██▎       | 92363/400000 [00:11<00:33, 9301.96it/s] 23%|██▎       | 93327/400000 [00:11<00:32, 9399.35it/s] 24%|██▎       | 94269/400000 [00:11<00:33, 9187.00it/s] 24%|██▍       | 95224/400000 [00:11<00:32, 9292.58it/s] 24%|██▍       | 96155/400000 [00:11<00:34, 8893.69it/s] 24%|██▍       | 97050/400000 [00:11<00:35, 8583.65it/s] 24%|██▍       | 97944/400000 [00:11<00:34, 8684.83it/s] 25%|██▍       | 98817/400000 [00:11<00:35, 8475.71it/s] 25%|██▍       | 99780/400000 [00:12<00:34, 8789.74it/s] 25%|██▌       | 100740/400000 [00:12<00:33, 9017.17it/s] 25%|██▌       | 101699/400000 [00:12<00:32, 9179.28it/s] 26%|██▌       | 102622/400000 [00:12<00:32, 9078.18it/s] 26%|██▌       | 103534/400000 [00:12<00:32, 9009.15it/s] 26%|██▌       | 104486/400000 [00:12<00:32, 9155.23it/s] 26%|██▋       | 105421/400000 [00:12<00:31, 9212.52it/s] 27%|██▋       | 106360/400000 [00:12<00:31, 9262.96it/s] 27%|██▋       | 107304/400000 [00:12<00:31, 9313.59it/s] 27%|██▋       | 108237/400000 [00:12<00:32, 9061.42it/s] 27%|██▋       | 109181/400000 [00:13<00:31, 9170.66it/s] 28%|██▊       | 110148/400000 [00:13<00:31, 9312.66it/s] 28%|██▊       | 111090/400000 [00:13<00:30, 9342.78it/s] 28%|██▊       | 112062/400000 [00:13<00:30, 9451.85it/s] 28%|██▊       | 113009/400000 [00:13<00:30, 9314.27it/s] 28%|██▊       | 113971/400000 [00:13<00:30, 9403.48it/s] 29%|██▊       | 114937/400000 [00:13<00:30, 9475.81it/s] 29%|██▉       | 115893/400000 [00:13<00:29, 9498.20it/s] 29%|██▉       | 116844/400000 [00:13<00:29, 9466.67it/s] 29%|██▉       | 117792/400000 [00:14<00:30, 9180.26it/s] 30%|██▉       | 118713/400000 [00:14<00:31, 8981.76it/s] 30%|██▉       | 119658/400000 [00:14<00:30, 9114.66it/s] 30%|███       | 120626/400000 [00:14<00:30, 9275.00it/s] 30%|███       | 121556/400000 [00:14<00:30, 9250.58it/s] 31%|███       | 122483/400000 [00:14<00:30, 9093.48it/s] 31%|███       | 123436/400000 [00:14<00:29, 9219.67it/s] 31%|███       | 124368/400000 [00:14<00:29, 9249.30it/s] 31%|███▏      | 125331/400000 [00:14<00:29, 9358.06it/s] 32%|███▏      | 126298/400000 [00:14<00:28, 9446.78it/s] 32%|███▏      | 127244/400000 [00:15<00:29, 9284.56it/s] 32%|███▏      | 128174/400000 [00:15<00:29, 9253.25it/s] 32%|███▏      | 129119/400000 [00:15<00:29, 9308.68it/s] 33%|███▎      | 130062/400000 [00:15<00:28, 9344.15it/s] 33%|███▎      | 130997/400000 [00:15<00:28, 9327.10it/s] 33%|███▎      | 131931/400000 [00:15<00:29, 9134.62it/s] 33%|███▎      | 132846/400000 [00:15<00:29, 9037.90it/s] 33%|███▎      | 133751/400000 [00:15<00:29, 9041.48it/s] 34%|███▎      | 134704/400000 [00:15<00:28, 9180.66it/s] 34%|███▍      | 135657/400000 [00:15<00:28, 9280.01it/s] 34%|███▍      | 136586/400000 [00:16<00:28, 9207.74it/s] 34%|███▍      | 137520/400000 [00:16<00:28, 9246.95it/s] 35%|███▍      | 138450/400000 [00:16<00:28, 9262.08it/s] 35%|███▍      | 139421/400000 [00:16<00:27, 9391.92it/s] 35%|███▌      | 140361/400000 [00:16<00:27, 9392.69it/s] 35%|███▌      | 141301/400000 [00:16<00:28, 9226.79it/s] 36%|███▌      | 142251/400000 [00:16<00:27, 9305.47it/s] 36%|███▌      | 143198/400000 [00:16<00:27, 9351.61it/s] 36%|███▌      | 144173/400000 [00:16<00:27, 9466.57it/s] 36%|███▋      | 145121/400000 [00:16<00:26, 9447.99it/s] 37%|███▋      | 146067/400000 [00:17<00:27, 9205.81it/s] 37%|███▋      | 146990/400000 [00:17<00:27, 9206.55it/s] 37%|███▋      | 147922/400000 [00:17<00:27, 9238.58it/s] 37%|███▋      | 148847/400000 [00:17<00:27, 9114.61it/s] 37%|███▋      | 149771/400000 [00:17<00:27, 9150.12it/s] 38%|███▊      | 150687/400000 [00:17<00:27, 9049.25it/s] 38%|███▊      | 151639/400000 [00:17<00:27, 9183.14it/s] 38%|███▊      | 152559/400000 [00:17<00:27, 9071.24it/s] 38%|███▊      | 153468/400000 [00:17<00:27, 9034.56it/s] 39%|███▊      | 154406/400000 [00:17<00:26, 9133.27it/s] 39%|███▉      | 155321/400000 [00:18<00:27, 9046.97it/s] 39%|███▉      | 156283/400000 [00:18<00:26, 9209.47it/s] 39%|███▉      | 157206/400000 [00:18<00:26, 9214.86it/s] 40%|███▉      | 158147/400000 [00:18<00:26, 9271.69it/s] 40%|███▉      | 159077/400000 [00:18<00:25, 9279.65it/s] 40%|████      | 160006/400000 [00:18<00:26, 9205.98it/s] 40%|████      | 160964/400000 [00:18<00:25, 9314.75it/s] 40%|████      | 161909/400000 [00:18<00:25, 9353.46it/s] 41%|████      | 162859/400000 [00:18<00:25, 9395.23it/s] 41%|████      | 163799/400000 [00:18<00:25, 9312.14it/s] 41%|████      | 164731/400000 [00:19<00:25, 9220.14it/s] 41%|████▏     | 165680/400000 [00:19<00:25, 9296.73it/s] 42%|████▏     | 166649/400000 [00:19<00:24, 9408.63it/s] 42%|████▏     | 167591/400000 [00:19<00:24, 9376.59it/s] 42%|████▏     | 168553/400000 [00:19<00:24, 9447.49it/s] 42%|████▏     | 169499/400000 [00:19<00:24, 9278.51it/s] 43%|████▎     | 170428/400000 [00:19<00:25, 9153.90it/s] 43%|████▎     | 171345/400000 [00:19<00:25, 9127.76it/s] 43%|████▎     | 172259/400000 [00:19<00:25, 8852.16it/s] 43%|████▎     | 173167/400000 [00:20<00:25, 8912.80it/s] 44%|████▎     | 174061/400000 [00:20<00:25, 8828.03it/s] 44%|████▍     | 175016/400000 [00:20<00:24, 9031.23it/s] 44%|████▍     | 175957/400000 [00:20<00:24, 9139.96it/s] 44%|████▍     | 176910/400000 [00:20<00:24, 9251.25it/s] 44%|████▍     | 177849/400000 [00:20<00:23, 9289.84it/s] 45%|████▍     | 178780/400000 [00:20<00:24, 9145.70it/s] 45%|████▍     | 179696/400000 [00:20<00:24, 9139.89it/s] 45%|████▌     | 180611/400000 [00:20<00:24, 8915.63it/s] 45%|████▌     | 181535/400000 [00:20<00:24, 9009.96it/s] 46%|████▌     | 182477/400000 [00:21<00:23, 9128.87it/s] 46%|████▌     | 183392/400000 [00:21<00:23, 9109.64it/s] 46%|████▌     | 184304/400000 [00:21<00:23, 9075.75it/s] 46%|████▋     | 185257/400000 [00:21<00:23, 9206.21it/s] 47%|████▋     | 186179/400000 [00:21<00:23, 9144.45it/s] 47%|████▋     | 187125/400000 [00:21<00:23, 9236.17it/s] 47%|████▋     | 188050/400000 [00:21<00:23, 9119.98it/s] 47%|████▋     | 188997/400000 [00:21<00:22, 9221.48it/s] 47%|████▋     | 189921/400000 [00:21<00:23, 9123.53it/s] 48%|████▊     | 190835/400000 [00:21<00:24, 8670.67it/s] 48%|████▊     | 191708/400000 [00:22<00:25, 8241.05it/s] 48%|████▊     | 192649/400000 [00:22<00:24, 8559.86it/s] 48%|████▊     | 193597/400000 [00:22<00:23, 8814.53it/s] 49%|████▊     | 194559/400000 [00:22<00:22, 9039.19it/s] 49%|████▉     | 195530/400000 [00:22<00:22, 9228.56it/s] 49%|████▉     | 196459/400000 [00:22<00:22, 9177.43it/s] 49%|████▉     | 197381/400000 [00:22<00:22, 9109.22it/s] 50%|████▉     | 198322/400000 [00:22<00:21, 9197.16it/s] 50%|████▉     | 199290/400000 [00:22<00:21, 9334.20it/s] 50%|█████     | 200240/400000 [00:22<00:21, 9380.38it/s] 50%|█████     | 201180/400000 [00:23<00:21, 9196.56it/s] 51%|█████     | 202102/400000 [00:23<00:21, 9188.52it/s] 51%|█████     | 203074/400000 [00:23<00:21, 9340.05it/s] 51%|█████     | 204038/400000 [00:23<00:20, 9427.82it/s] 51%|█████     | 204988/400000 [00:23<00:20, 9448.91it/s] 51%|█████▏    | 205934/400000 [00:23<00:21, 9014.00it/s] 52%|█████▏    | 206848/400000 [00:23<00:21, 9049.37it/s] 52%|█████▏    | 207757/400000 [00:23<00:21, 8964.65it/s] 52%|█████▏    | 208657/400000 [00:23<00:21, 8838.44it/s] 52%|█████▏    | 209544/400000 [00:24<00:21, 8738.13it/s] 53%|█████▎    | 210420/400000 [00:24<00:22, 8495.67it/s] 53%|█████▎    | 211273/400000 [00:24<00:22, 8496.31it/s] 53%|█████▎    | 212150/400000 [00:24<00:21, 8573.69it/s] 53%|█████▎    | 213023/400000 [00:24<00:21, 8618.05it/s] 53%|█████▎    | 213925/400000 [00:24<00:21, 8733.52it/s] 54%|█████▎    | 214800/400000 [00:24<00:21, 8514.61it/s] 54%|█████▍    | 215697/400000 [00:24<00:21, 8646.15it/s] 54%|█████▍    | 216641/400000 [00:24<00:20, 8869.16it/s] 54%|█████▍    | 217615/400000 [00:24<00:20, 9112.19it/s] 55%|█████▍    | 218530/400000 [00:25<00:19, 9106.79it/s] 55%|█████▍    | 219444/400000 [00:25<00:20, 9012.72it/s] 55%|█████▌    | 220416/400000 [00:25<00:19, 9212.99it/s] 55%|█████▌    | 221380/400000 [00:25<00:19, 9335.55it/s] 56%|█████▌    | 222316/400000 [00:25<00:19, 9289.00it/s] 56%|█████▌    | 223247/400000 [00:25<00:19, 9217.58it/s] 56%|█████▌    | 224170/400000 [00:25<00:19, 8973.17it/s] 56%|█████▋    | 225093/400000 [00:25<00:19, 9046.31it/s] 57%|█████▋    | 226027/400000 [00:25<00:19, 9131.15it/s] 57%|█████▋    | 226942/400000 [00:25<00:19, 9054.38it/s] 57%|█████▋    | 227857/400000 [00:26<00:18, 9079.93it/s] 57%|█████▋    | 228766/400000 [00:26<00:19, 8881.29it/s] 57%|█████▋    | 229719/400000 [00:26<00:18, 9065.79it/s] 58%|█████▊    | 230692/400000 [00:26<00:18, 9253.73it/s] 58%|█████▊    | 231620/400000 [00:26<00:18, 9182.12it/s] 58%|█████▊    | 232540/400000 [00:26<00:18, 9170.14it/s] 58%|█████▊    | 233459/400000 [00:26<00:18, 9166.75it/s] 59%|█████▊    | 234432/400000 [00:26<00:17, 9325.48it/s] 59%|█████▉    | 235377/400000 [00:26<00:17, 9359.17it/s] 59%|█████▉    | 236326/400000 [00:26<00:17, 9397.04it/s] 59%|█████▉    | 237267/400000 [00:27<00:17, 9375.18it/s] 60%|█████▉    | 238206/400000 [00:27<00:17, 9276.11it/s] 60%|█████▉    | 239135/400000 [00:27<00:17, 9256.36it/s] 60%|██████    | 240062/400000 [00:27<00:17, 9222.59it/s] 60%|██████    | 240987/400000 [00:27<00:17, 9230.35it/s] 60%|██████    | 241922/400000 [00:27<00:17, 9264.37it/s] 61%|██████    | 242849/400000 [00:27<00:17, 9105.38it/s] 61%|██████    | 243766/400000 [00:27<00:17, 9122.59it/s] 61%|██████    | 244679/400000 [00:27<00:17, 9104.19it/s] 61%|██████▏   | 245639/400000 [00:27<00:16, 9245.71it/s] 62%|██████▏   | 246573/400000 [00:28<00:16, 9271.79it/s] 62%|██████▏   | 247501/400000 [00:28<00:16, 9213.16it/s] 62%|██████▏   | 248423/400000 [00:28<00:16, 9200.75it/s] 62%|██████▏   | 249369/400000 [00:28<00:16, 9276.91it/s] 63%|██████▎   | 250305/400000 [00:28<00:16, 9299.03it/s] 63%|██████▎   | 251245/400000 [00:28<00:15, 9327.85it/s] 63%|██████▎   | 252179/400000 [00:28<00:16, 9225.36it/s] 63%|██████▎   | 253102/400000 [00:28<00:15, 9189.14it/s] 64%|██████▎   | 254052/400000 [00:28<00:15, 9277.37it/s] 64%|██████▎   | 254981/400000 [00:29<00:15, 9209.20it/s] 64%|██████▍   | 255940/400000 [00:29<00:15, 9318.53it/s] 64%|██████▍   | 256873/400000 [00:29<00:15, 9119.45it/s] 64%|██████▍   | 257788/400000 [00:29<00:15, 9128.30it/s] 65%|██████▍   | 258702/400000 [00:29<00:15, 9113.31it/s] 65%|██████▍   | 259651/400000 [00:29<00:15, 9220.73it/s] 65%|██████▌   | 260599/400000 [00:29<00:14, 9295.43it/s] 65%|██████▌   | 261530/400000 [00:29<00:15, 8963.60it/s] 66%|██████▌   | 262477/400000 [00:29<00:15, 9108.37it/s] 66%|██████▌   | 263394/400000 [00:29<00:14, 9125.64it/s] 66%|██████▌   | 264321/400000 [00:30<00:14, 9166.27it/s] 66%|██████▋   | 265239/400000 [00:30<00:15, 8836.05it/s] 67%|██████▋   | 266127/400000 [00:30<00:15, 8732.88it/s] 67%|██████▋   | 267037/400000 [00:30<00:15, 8839.52it/s] 67%|██████▋   | 267980/400000 [00:30<00:14, 9008.20it/s] 67%|██████▋   | 268909/400000 [00:30<00:14, 9088.42it/s] 67%|██████▋   | 269820/400000 [00:30<00:14, 9090.55it/s] 68%|██████▊   | 270731/400000 [00:30<00:14, 8993.31it/s] 68%|██████▊   | 271653/400000 [00:30<00:14, 9058.37it/s] 68%|██████▊   | 272560/400000 [00:30<00:14, 8792.73it/s] 68%|██████▊   | 273442/400000 [00:31<00:14, 8652.44it/s] 69%|██████▊   | 274310/400000 [00:31<00:14, 8564.11it/s] 69%|██████▉   | 275169/400000 [00:31<00:14, 8505.08it/s] 69%|██████▉   | 276030/400000 [00:31<00:14, 8534.76it/s] 69%|██████▉   | 276958/400000 [00:31<00:14, 8744.46it/s] 69%|██████▉   | 277930/400000 [00:31<00:13, 9014.42it/s] 70%|██████▉   | 278879/400000 [00:31<00:13, 9149.87it/s] 70%|██████▉   | 279797/400000 [00:31<00:13, 9141.80it/s] 70%|███████   | 280773/400000 [00:31<00:12, 9318.46it/s] 70%|███████   | 281748/400000 [00:31<00:12, 9443.80it/s] 71%|███████   | 282695/400000 [00:32<00:12, 9382.62it/s] 71%|███████   | 283635/400000 [00:32<00:13, 8847.01it/s] 71%|███████   | 284528/400000 [00:32<00:13, 8418.97it/s] 71%|███████▏  | 285484/400000 [00:32<00:13, 8730.97it/s] 72%|███████▏  | 286438/400000 [00:32<00:12, 8956.67it/s] 72%|███████▏  | 287342/400000 [00:32<00:12, 8898.58it/s] 72%|███████▏  | 288302/400000 [00:32<00:12, 9097.91it/s] 72%|███████▏  | 289218/400000 [00:32<00:12, 9114.78it/s] 73%|███████▎  | 290179/400000 [00:32<00:11, 9256.84it/s] 73%|███████▎  | 291115/400000 [00:33<00:11, 9287.22it/s] 73%|███████▎  | 292046/400000 [00:33<00:11, 9120.37it/s] 73%|███████▎  | 292980/400000 [00:33<00:11, 9184.59it/s] 73%|███████▎  | 293901/400000 [00:33<00:11, 9133.78it/s] 74%|███████▎  | 294836/400000 [00:33<00:11, 9195.05it/s] 74%|███████▍  | 295757/400000 [00:33<00:11, 9153.75it/s] 74%|███████▍  | 296715/400000 [00:33<00:11, 9274.82it/s] 74%|███████▍  | 297649/400000 [00:33<00:11, 9294.20it/s] 75%|███████▍  | 298580/400000 [00:33<00:11, 9205.98it/s] 75%|███████▍  | 299502/400000 [00:33<00:10, 9192.51it/s] 75%|███████▌  | 300422/400000 [00:34<00:10, 9178.45it/s] 75%|███████▌  | 301341/400000 [00:34<00:10, 9065.90it/s] 76%|███████▌  | 302285/400000 [00:34<00:10, 9172.59it/s] 76%|███████▌  | 303203/400000 [00:34<00:10, 9081.10it/s] 76%|███████▌  | 304150/400000 [00:34<00:10, 9191.98it/s] 76%|███████▋  | 305122/400000 [00:34<00:10, 9343.86it/s] 77%|███████▋  | 306106/400000 [00:34<00:09, 9485.72it/s] 77%|███████▋  | 307075/400000 [00:34<00:09, 9544.78it/s] 77%|███████▋  | 308031/400000 [00:34<00:10, 9128.74it/s] 77%|███████▋  | 308949/400000 [00:34<00:10, 9063.15it/s] 77%|███████▋  | 309890/400000 [00:35<00:09, 9163.66it/s] 78%|███████▊  | 310849/400000 [00:35<00:09, 9285.89it/s] 78%|███████▊  | 311780/400000 [00:35<00:09, 9244.42it/s] 78%|███████▊  | 312707/400000 [00:35<00:09, 9067.81it/s] 78%|███████▊  | 313654/400000 [00:35<00:09, 9182.77it/s] 79%|███████▊  | 314593/400000 [00:35<00:09, 9242.68it/s] 79%|███████▉  | 315551/400000 [00:35<00:09, 9338.64it/s] 79%|███████▉  | 316490/400000 [00:35<00:08, 9352.14it/s] 79%|███████▉  | 317426/400000 [00:35<00:08, 9237.37it/s] 80%|███████▉  | 318389/400000 [00:35<00:08, 9351.68it/s] 80%|███████▉  | 319350/400000 [00:36<00:08, 9426.22it/s] 80%|████████  | 320304/400000 [00:36<00:08, 9457.80it/s] 80%|████████  | 321251/400000 [00:36<00:08, 9054.40it/s] 81%|████████  | 322161/400000 [00:36<00:08, 8959.08it/s] 81%|████████  | 323108/400000 [00:36<00:08, 9104.76it/s] 81%|████████  | 324055/400000 [00:36<00:08, 9208.34it/s] 81%|████████▏ | 325008/400000 [00:36<00:08, 9301.33it/s] 81%|████████▏ | 325954/400000 [00:36<00:07, 9347.81it/s] 82%|████████▏ | 326891/400000 [00:36<00:07, 9260.92it/s] 82%|████████▏ | 327855/400000 [00:36<00:07, 9371.16it/s] 82%|████████▏ | 328816/400000 [00:37<00:07, 9441.05it/s] 82%|████████▏ | 329771/400000 [00:37<00:07, 9471.18it/s] 83%|████████▎ | 330719/400000 [00:37<00:07, 9392.89it/s] 83%|████████▎ | 331659/400000 [00:37<00:07, 9148.37it/s] 83%|████████▎ | 332576/400000 [00:37<00:07, 9151.00it/s] 83%|████████▎ | 333524/400000 [00:37<00:07, 9245.45it/s] 84%|████████▎ | 334450/400000 [00:37<00:07, 9110.06it/s] 84%|████████▍ | 335400/400000 [00:37<00:07, 9222.40it/s] 84%|████████▍ | 336324/400000 [00:37<00:06, 9183.63it/s] 84%|████████▍ | 337288/400000 [00:38<00:06, 9315.85it/s] 85%|████████▍ | 338221/400000 [00:38<00:06, 9311.48it/s] 85%|████████▍ | 339153/400000 [00:38<00:06, 9282.44it/s] 85%|████████▌ | 340119/400000 [00:38<00:06, 9390.72it/s] 85%|████████▌ | 341059/400000 [00:38<00:06, 9285.43it/s] 86%|████████▌ | 342022/400000 [00:38<00:06, 9384.58it/s] 86%|████████▌ | 342962/400000 [00:38<00:06, 9300.96it/s] 86%|████████▌ | 343922/400000 [00:38<00:05, 9387.39it/s] 86%|████████▌ | 344887/400000 [00:38<00:05, 9464.35it/s] 86%|████████▋ | 345835/400000 [00:38<00:05, 9260.36it/s] 87%|████████▋ | 346763/400000 [00:39<00:05, 9242.06it/s] 87%|████████▋ | 347718/400000 [00:39<00:05, 9332.29it/s] 87%|████████▋ | 348653/400000 [00:39<00:05, 9281.31it/s] 87%|████████▋ | 349590/400000 [00:39<00:05, 9305.66it/s] 88%|████████▊ | 350522/400000 [00:39<00:05, 9153.01it/s] 88%|████████▊ | 351440/400000 [00:39<00:05, 9159.32it/s] 88%|████████▊ | 352367/400000 [00:39<00:05, 9190.97it/s] 88%|████████▊ | 353316/400000 [00:39<00:05, 9278.31it/s] 89%|████████▊ | 354277/400000 [00:39<00:04, 9371.66it/s] 89%|████████▉ | 355215/400000 [00:39<00:04, 9059.23it/s] 89%|████████▉ | 356163/400000 [00:40<00:04, 9179.16it/s] 89%|████████▉ | 357093/400000 [00:40<00:04, 9213.17it/s] 90%|████████▉ | 358048/400000 [00:40<00:04, 9308.65it/s] 90%|████████▉ | 358981/400000 [00:40<00:04, 9250.80it/s] 90%|████████▉ | 359908/400000 [00:40<00:04, 8904.69it/s] 90%|█████████ | 360802/400000 [00:40<00:04, 8880.66it/s] 90%|█████████ | 361730/400000 [00:40<00:04, 8994.75it/s] 91%|█████████ | 362667/400000 [00:40<00:04, 9101.58it/s] 91%|█████████ | 363584/400000 [00:40<00:03, 9120.36it/s] 91%|█████████ | 364498/400000 [00:40<00:03, 9027.80it/s] 91%|█████████▏| 365442/400000 [00:41<00:03, 9146.83it/s] 92%|█████████▏| 366395/400000 [00:41<00:03, 9258.36it/s] 92%|█████████▏| 367356/400000 [00:41<00:03, 9358.50it/s] 92%|█████████▏| 368293/400000 [00:41<00:03, 9315.21it/s] 92%|█████████▏| 369226/400000 [00:41<00:03, 8998.73it/s] 93%|█████████▎| 370142/400000 [00:41<00:03, 9044.73it/s] 93%|█████████▎| 371057/400000 [00:41<00:03, 9072.97it/s] 93%|█████████▎| 371993/400000 [00:41<00:03, 9155.50it/s] 93%|█████████▎| 372963/400000 [00:41<00:02, 9311.29it/s] 93%|█████████▎| 373896/400000 [00:41<00:02, 9245.34it/s] 94%|█████████▎| 374822/400000 [00:42<00:02, 9246.04it/s] 94%|█████████▍| 375769/400000 [00:42<00:02, 9311.74it/s] 94%|█████████▍| 376701/400000 [00:42<00:02, 9287.00it/s] 94%|█████████▍| 377631/400000 [00:42<00:02, 8776.37it/s] 95%|█████████▍| 378515/400000 [00:42<00:02, 8480.39it/s] 95%|█████████▍| 379382/400000 [00:42<00:02, 8535.23it/s] 95%|█████████▌| 380332/400000 [00:42<00:02, 8798.71it/s] 95%|█████████▌| 381281/400000 [00:42<00:02, 8995.12it/s] 96%|█████████▌| 382186/400000 [00:42<00:01, 9009.66it/s] 96%|█████████▌| 383112/400000 [00:43<00:01, 9082.16it/s] 96%|█████████▌| 384057/400000 [00:43<00:01, 9188.83it/s] 96%|█████████▌| 384978/400000 [00:43<00:01, 9191.42it/s] 96%|█████████▋| 385935/400000 [00:43<00:01, 9300.32it/s] 97%|█████████▋| 386867/400000 [00:43<00:01, 9178.17it/s] 97%|█████████▋| 387822/400000 [00:43<00:01, 9283.97it/s] 97%|█████████▋| 388757/400000 [00:43<00:01, 9303.17it/s] 97%|█████████▋| 389689/400000 [00:43<00:01, 9190.54it/s] 98%|█████████▊| 390640/400000 [00:43<00:01, 9284.03it/s] 98%|█████████▊| 391570/400000 [00:43<00:00, 9277.14it/s] 98%|█████████▊| 392499/400000 [00:44<00:00, 9246.98it/s] 98%|█████████▊| 393425/400000 [00:44<00:00, 9213.23it/s] 99%|█████████▊| 394347/400000 [00:44<00:00, 9208.32it/s] 99%|█████████▉| 395306/400000 [00:44<00:00, 9317.41it/s] 99%|█████████▉| 396239/400000 [00:44<00:00, 9230.51it/s] 99%|█████████▉| 397175/400000 [00:44<00:00, 9265.83it/s]100%|█████████▉| 398126/400000 [00:44<00:00, 9336.52it/s]100%|█████████▉| 399091/400000 [00:44<00:00, 9426.92it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8923.13it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc9641270f0>, <torchtext.data.dataset.TabularDataset object at 0x7fc964127240>, <torchtext.vocab.Vocab object at 0x7fc964127160>), {}) 

