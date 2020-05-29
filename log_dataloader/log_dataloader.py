
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '097d79840b302a3e2f9efdf341ce6b4d32d7a46c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdd44c4e2f0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdd44c4e2f0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fddb6c9c0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fddb6c9c0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdd4e42d9d8> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdd4e42d9d8> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdd645808c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdd645808c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdd645808c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 48%|████▊     | 4718592/9912422 [00:00<00:00, 46681674.26it/s]9920512it [00:00, 33588452.10it/s]                             
0it [00:00, ?it/s]32768it [00:00, 482711.10it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7689288.60it/s]          
0it [00:00, ?it/s]8192it [00:00, 159471.54it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd46f57588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd46f57cc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fdd64580510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fdd64580510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fdd64580510> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:49,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:48,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:34,  4.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:34,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:34,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:33,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  6.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:23,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:22,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  6.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 14.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 42.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 56.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 65.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 66.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 66.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 66.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 66.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.69 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 66.69 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.68 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.46s/ url]
0 examples [00:00, ? examples/s]2020-05-29 00:09:33.347504: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-29 00:09:33.362952: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-29 00:09:33.363152: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ca5d2e0ca0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 00:09:33.363171: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
30 examples [00:00, 299.20 examples/s]107 examples [00:00, 365.84 examples/s]193 examples [00:00, 441.21 examples/s]281 examples [00:00, 518.43 examples/s]371 examples [00:00, 593.26 examples/s]459 examples [00:00, 657.00 examples/s]547 examples [00:00, 710.08 examples/s]633 examples [00:00, 748.44 examples/s]720 examples [00:00, 781.03 examples/s]804 examples [00:01, 797.50 examples/s]889 examples [00:01, 810.58 examples/s]973 examples [00:01, 814.48 examples/s]1063 examples [00:01, 836.06 examples/s]1149 examples [00:01, 840.86 examples/s]1236 examples [00:01, 847.85 examples/s]1322 examples [00:01, 850.44 examples/s]1408 examples [00:01, 849.70 examples/s]1494 examples [00:01, 850.76 examples/s]1580 examples [00:01, 844.02 examples/s]1668 examples [00:02, 852.90 examples/s]1754 examples [00:02, 848.12 examples/s]1839 examples [00:02, 838.16 examples/s]1926 examples [00:02, 844.95 examples/s]2014 examples [00:02, 852.60 examples/s]2101 examples [00:02, 855.83 examples/s]2188 examples [00:02, 856.80 examples/s]2274 examples [00:02, 854.22 examples/s]2360 examples [00:02, 844.81 examples/s]2445 examples [00:02, 829.71 examples/s]2533 examples [00:03, 842.18 examples/s]2623 examples [00:03, 857.50 examples/s]2713 examples [00:03, 868.56 examples/s]2802 examples [00:03, 873.00 examples/s]2890 examples [00:03, 872.94 examples/s]2978 examples [00:03, 869.95 examples/s]3066 examples [00:03, 865.85 examples/s]3153 examples [00:03, 859.54 examples/s]3240 examples [00:03, 859.96 examples/s]3327 examples [00:03, 862.93 examples/s]3414 examples [00:04, 863.46 examples/s]3503 examples [00:04, 868.78 examples/s]3592 examples [00:04, 874.37 examples/s]3680 examples [00:04, 867.26 examples/s]3767 examples [00:04, 863.99 examples/s]3854 examples [00:04, 862.80 examples/s]3941 examples [00:04, 855.24 examples/s]4027 examples [00:04, 839.21 examples/s]4112 examples [00:04, 829.42 examples/s]4196 examples [00:04, 783.67 examples/s]4281 examples [00:05, 801.80 examples/s]4362 examples [00:05, 804.19 examples/s]4448 examples [00:05, 819.49 examples/s]4536 examples [00:05, 836.36 examples/s]4623 examples [00:05, 843.59 examples/s]4712 examples [00:05, 855.60 examples/s]4798 examples [00:05, 846.88 examples/s]4883 examples [00:05, 842.81 examples/s]4970 examples [00:05, 849.92 examples/s]5057 examples [00:06, 855.13 examples/s]5143 examples [00:06, 852.16 examples/s]5230 examples [00:06, 857.21 examples/s]5316 examples [00:06, 846.79 examples/s]5406 examples [00:06, 860.15 examples/s]5493 examples [00:06, 859.33 examples/s]5580 examples [00:06, 862.48 examples/s]5667 examples [00:06, 845.15 examples/s]5752 examples [00:06, 846.14 examples/s]5842 examples [00:06, 861.17 examples/s]5934 examples [00:07, 877.92 examples/s]6025 examples [00:07, 885.79 examples/s]6114 examples [00:07, 875.54 examples/s]6202 examples [00:07, 793.81 examples/s]6283 examples [00:07, 781.30 examples/s]6367 examples [00:07, 797.61 examples/s]6450 examples [00:07, 804.88 examples/s]6535 examples [00:07, 816.67 examples/s]6622 examples [00:07, 831.40 examples/s]6708 examples [00:07, 837.39 examples/s]6796 examples [00:08, 847.87 examples/s]6882 examples [00:08, 847.69 examples/s]6967 examples [00:08, 830.39 examples/s]7056 examples [00:08, 844.87 examples/s]7143 examples [00:08, 849.38 examples/s]7229 examples [00:08, 836.29 examples/s]7313 examples [00:08, 834.80 examples/s]7397 examples [00:08, 835.30 examples/s]7484 examples [00:08, 844.15 examples/s]7572 examples [00:08, 853.20 examples/s]7661 examples [00:09, 861.30 examples/s]7748 examples [00:09, 855.03 examples/s]7835 examples [00:09, 859.06 examples/s]7922 examples [00:09, 861.11 examples/s]8009 examples [00:09, 862.35 examples/s]8096 examples [00:09, 856.42 examples/s]8182 examples [00:09, 825.61 examples/s]8265 examples [00:09, 815.44 examples/s]8349 examples [00:09, 820.83 examples/s]8432 examples [00:10, 822.04 examples/s]8518 examples [00:10, 832.80 examples/s]8607 examples [00:10, 847.09 examples/s]8693 examples [00:10, 848.52 examples/s]8780 examples [00:10, 854.73 examples/s]8866 examples [00:10, 854.83 examples/s]8952 examples [00:10, 853.51 examples/s]9038 examples [00:10, 853.00 examples/s]9124 examples [00:10, 848.56 examples/s]9212 examples [00:10, 855.07 examples/s]9298 examples [00:11, 852.88 examples/s]9384 examples [00:11, 848.03 examples/s]9469 examples [00:11, 848.43 examples/s]9554 examples [00:11, 828.41 examples/s]9641 examples [00:11, 839.99 examples/s]9728 examples [00:11, 846.22 examples/s]9815 examples [00:11, 851.81 examples/s]9901 examples [00:11, 849.79 examples/s]9987 examples [00:11, 842.09 examples/s]10072 examples [00:11, 795.25 examples/s]10157 examples [00:12, 809.68 examples/s]10242 examples [00:12, 821.24 examples/s]10328 examples [00:12, 830.32 examples/s]10412 examples [00:12, 833.15 examples/s]10496 examples [00:12, 824.85 examples/s]10579 examples [00:12, 820.43 examples/s]10662 examples [00:12, 822.41 examples/s]10748 examples [00:12, 832.03 examples/s]10832 examples [00:12, 801.66 examples/s]10919 examples [00:12, 819.55 examples/s]11002 examples [00:13, 813.72 examples/s]11087 examples [00:13, 823.57 examples/s]11176 examples [00:13, 839.92 examples/s]11262 examples [00:13, 843.04 examples/s]11349 examples [00:13, 850.87 examples/s]11437 examples [00:13, 856.96 examples/s]11525 examples [00:13, 861.92 examples/s]11612 examples [00:13, 860.97 examples/s]11699 examples [00:13, 861.32 examples/s]11787 examples [00:14, 865.68 examples/s]11875 examples [00:14, 869.57 examples/s]11964 examples [00:14, 875.17 examples/s]12053 examples [00:14, 878.77 examples/s]12141 examples [00:14, 868.56 examples/s]12230 examples [00:14, 871.87 examples/s]12318 examples [00:14, 872.74 examples/s]12407 examples [00:14, 875.47 examples/s]12495 examples [00:14, 876.06 examples/s]12583 examples [00:14, 872.73 examples/s]12671 examples [00:15, 871.60 examples/s]12759 examples [00:15, 867.60 examples/s]12847 examples [00:15, 870.24 examples/s]12935 examples [00:15, 870.22 examples/s]13023 examples [00:15, 866.47 examples/s]13110 examples [00:15, 863.98 examples/s]13198 examples [00:15, 867.28 examples/s]13288 examples [00:15, 874.88 examples/s]13376 examples [00:15, 867.52 examples/s]13463 examples [00:15, 858.66 examples/s]13549 examples [00:16, 857.10 examples/s]13636 examples [00:16, 859.70 examples/s]13722 examples [00:16, 858.24 examples/s]13808 examples [00:16, 856.77 examples/s]13894 examples [00:16, 849.05 examples/s]13981 examples [00:16, 853.48 examples/s]14067 examples [00:16, 855.21 examples/s]14153 examples [00:16, 848.27 examples/s]14239 examples [00:16, 849.62 examples/s]14324 examples [00:16, 829.35 examples/s]14408 examples [00:17, 819.57 examples/s]14491 examples [00:17, 821.42 examples/s]14574 examples [00:17, 811.55 examples/s]14656 examples [00:17, 788.90 examples/s]14736 examples [00:17, 780.17 examples/s]14824 examples [00:17, 807.10 examples/s]14914 examples [00:17, 830.70 examples/s]15000 examples [00:17, 838.56 examples/s]15088 examples [00:17, 848.40 examples/s]15174 examples [00:17, 849.07 examples/s]15263 examples [00:18, 858.36 examples/s]15350 examples [00:18, 859.59 examples/s]15437 examples [00:18, 861.64 examples/s]15524 examples [00:18, 857.85 examples/s]15610 examples [00:18, 844.54 examples/s]15696 examples [00:18, 848.81 examples/s]15784 examples [00:18, 857.15 examples/s]15871 examples [00:18, 860.00 examples/s]15958 examples [00:18, 862.11 examples/s]16045 examples [00:18, 857.37 examples/s]16131 examples [00:19, 848.90 examples/s]16219 examples [00:19, 856.50 examples/s]16309 examples [00:19, 867.96 examples/s]16398 examples [00:19, 873.56 examples/s]16486 examples [00:19, 873.61 examples/s]16575 examples [00:19, 876.35 examples/s]16664 examples [00:19, 878.31 examples/s]16752 examples [00:19, 874.16 examples/s]16840 examples [00:19, 866.65 examples/s]16928 examples [00:20, 868.73 examples/s]17015 examples [00:20, 860.18 examples/s]17106 examples [00:20, 872.51 examples/s]17195 examples [00:20, 876.79 examples/s]17283 examples [00:20, 874.37 examples/s]17371 examples [00:20, 817.15 examples/s]17460 examples [00:20, 836.57 examples/s]17551 examples [00:20, 855.82 examples/s]17640 examples [00:20, 863.12 examples/s]17729 examples [00:20, 868.92 examples/s]17818 examples [00:21, 874.56 examples/s]17906 examples [00:21, 851.10 examples/s]17993 examples [00:21, 855.20 examples/s]18080 examples [00:21, 858.32 examples/s]18166 examples [00:21, 856.32 examples/s]18252 examples [00:21, 849.19 examples/s]18338 examples [00:21, 849.10 examples/s]18425 examples [00:21, 854.32 examples/s]18514 examples [00:21, 863.38 examples/s]18603 examples [00:21, 869.41 examples/s]18693 examples [00:22, 876.18 examples/s]18781 examples [00:22, 864.69 examples/s]18868 examples [00:22, 863.96 examples/s]18955 examples [00:22, 862.97 examples/s]19042 examples [00:22, 858.51 examples/s]19128 examples [00:22, 847.76 examples/s]19217 examples [00:22, 857.96 examples/s]19303 examples [00:22, 853.47 examples/s]19391 examples [00:22, 858.35 examples/s]19479 examples [00:22, 864.05 examples/s]19566 examples [00:23, 864.23 examples/s]19653 examples [00:23, 855.99 examples/s]19741 examples [00:23, 860.31 examples/s]19828 examples [00:23, 859.83 examples/s]19915 examples [00:23, 862.43 examples/s]20002 examples [00:23, 796.03 examples/s]20094 examples [00:23, 828.95 examples/s]20179 examples [00:23, 834.64 examples/s]20266 examples [00:23, 843.65 examples/s]20355 examples [00:24, 855.04 examples/s]20443 examples [00:24, 860.53 examples/s]20530 examples [00:24, 859.53 examples/s]20617 examples [00:24, 857.35 examples/s]20703 examples [00:24, 847.03 examples/s]20788 examples [00:24, 843.89 examples/s]20873 examples [00:24, 825.46 examples/s]20956 examples [00:24, 816.90 examples/s]21038 examples [00:24, 808.30 examples/s]21122 examples [00:24, 816.70 examples/s]21204 examples [00:25, 814.53 examples/s]21287 examples [00:25, 818.09 examples/s]21373 examples [00:25, 829.46 examples/s]21457 examples [00:25, 821.47 examples/s]21543 examples [00:25, 830.07 examples/s]21627 examples [00:25, 829.23 examples/s]21712 examples [00:25, 834.29 examples/s]21800 examples [00:25, 844.83 examples/s]21887 examples [00:25, 850.66 examples/s]21975 examples [00:25, 858.73 examples/s]22061 examples [00:26, 844.67 examples/s]22148 examples [00:26, 851.15 examples/s]22235 examples [00:26, 854.29 examples/s]22321 examples [00:26, 849.34 examples/s]22406 examples [00:26, 835.20 examples/s]22490 examples [00:26, 821.34 examples/s]22573 examples [00:26, 817.91 examples/s]22655 examples [00:26, 815.26 examples/s]22737 examples [00:26, 809.73 examples/s]22822 examples [00:26, 820.05 examples/s]22906 examples [00:27, 823.60 examples/s]22989 examples [00:27, 800.00 examples/s]23077 examples [00:27, 819.61 examples/s]23160 examples [00:27, 822.30 examples/s]23246 examples [00:27, 830.92 examples/s]23332 examples [00:27, 833.61 examples/s]23418 examples [00:27, 839.49 examples/s]23504 examples [00:27, 844.25 examples/s]23591 examples [00:27, 851.57 examples/s]23677 examples [00:28, 851.67 examples/s]23763 examples [00:28, 850.44 examples/s]23849 examples [00:28, 847.50 examples/s]23938 examples [00:28, 859.62 examples/s]24025 examples [00:28, 857.30 examples/s]24111 examples [00:28, 856.87 examples/s]24197 examples [00:28, 856.56 examples/s]24284 examples [00:28, 860.03 examples/s]24371 examples [00:28, 851.18 examples/s]24457 examples [00:28, 847.85 examples/s]24542 examples [00:29, 846.13 examples/s]24630 examples [00:29, 854.35 examples/s]24716 examples [00:29, 845.26 examples/s]24803 examples [00:29, 849.95 examples/s]24890 examples [00:29, 854.13 examples/s]24977 examples [00:29, 855.41 examples/s]25063 examples [00:29, 801.28 examples/s]25147 examples [00:29, 812.08 examples/s]25234 examples [00:29, 828.21 examples/s]25318 examples [00:29, 808.37 examples/s]25401 examples [00:30, 812.44 examples/s]25483 examples [00:30, 810.29 examples/s]25565 examples [00:30, 810.62 examples/s]25647 examples [00:30, 804.13 examples/s]25728 examples [00:30, 793.98 examples/s]25811 examples [00:30, 803.70 examples/s]25893 examples [00:30, 807.39 examples/s]25974 examples [00:30, 790.02 examples/s]26058 examples [00:30, 802.25 examples/s]26141 examples [00:30, 809.01 examples/s]26223 examples [00:31, 807.19 examples/s]26307 examples [00:31, 814.42 examples/s]26389 examples [00:31, 798.65 examples/s]26475 examples [00:31, 815.10 examples/s]26559 examples [00:31, 820.90 examples/s]26642 examples [00:31, 816.06 examples/s]26726 examples [00:31, 821.11 examples/s]26810 examples [00:31, 824.40 examples/s]26893 examples [00:31, 820.68 examples/s]26980 examples [00:32, 833.74 examples/s]27064 examples [00:32, 808.32 examples/s]27146 examples [00:32, 811.62 examples/s]27232 examples [00:32, 823.96 examples/s]27317 examples [00:32, 829.77 examples/s]27402 examples [00:32, 833.95 examples/s]27486 examples [00:32, 832.00 examples/s]27573 examples [00:32, 841.78 examples/s]27658 examples [00:32, 813.72 examples/s]27740 examples [00:32, 804.62 examples/s]27822 examples [00:33, 808.20 examples/s]27903 examples [00:33, 791.69 examples/s]27989 examples [00:33, 810.59 examples/s]28075 examples [00:33, 822.23 examples/s]28160 examples [00:33, 829.23 examples/s]28245 examples [00:33, 834.74 examples/s]28330 examples [00:33, 838.84 examples/s]28420 examples [00:33, 855.16 examples/s]28508 examples [00:33, 862.09 examples/s]28598 examples [00:33, 871.66 examples/s]28688 examples [00:34, 879.45 examples/s]28777 examples [00:34, 873.04 examples/s]28867 examples [00:34, 880.11 examples/s]28956 examples [00:34, 865.65 examples/s]29043 examples [00:34, 856.56 examples/s]29129 examples [00:34, 844.20 examples/s]29214 examples [00:34, 831.58 examples/s]29298 examples [00:34, 820.12 examples/s]29385 examples [00:34, 833.86 examples/s]29472 examples [00:34, 841.76 examples/s]29557 examples [00:35, 838.16 examples/s]29641 examples [00:35, 835.94 examples/s]29725 examples [00:35, 829.96 examples/s]29809 examples [00:35, 831.33 examples/s]29893 examples [00:35, 826.20 examples/s]29976 examples [00:35, 812.20 examples/s]30058 examples [00:35, 769.59 examples/s]30142 examples [00:35, 788.61 examples/s]30222 examples [00:35, 773.39 examples/s]30305 examples [00:36, 788.99 examples/s]30395 examples [00:36, 817.71 examples/s]30482 examples [00:36, 830.95 examples/s]30570 examples [00:36, 843.27 examples/s]30655 examples [00:36, 841.66 examples/s]30742 examples [00:36, 848.92 examples/s]30828 examples [00:36, 851.68 examples/s]30914 examples [00:36, 849.38 examples/s]31000 examples [00:36, 848.34 examples/s]31085 examples [00:36, 845.69 examples/s]31171 examples [00:37, 848.76 examples/s]31256 examples [00:37, 844.72 examples/s]31341 examples [00:37, 846.04 examples/s]31426 examples [00:37, 841.86 examples/s]31511 examples [00:37, 824.45 examples/s]31599 examples [00:37, 840.34 examples/s]31689 examples [00:37, 854.99 examples/s]31776 examples [00:37, 859.22 examples/s]31863 examples [00:37, 840.24 examples/s]31952 examples [00:37, 852.73 examples/s]32038 examples [00:38, 849.75 examples/s]32124 examples [00:38, 844.98 examples/s]32209 examples [00:38, 845.08 examples/s]32294 examples [00:38, 843.80 examples/s]32379 examples [00:38, 841.94 examples/s]32466 examples [00:38, 847.12 examples/s]32551 examples [00:38, 845.90 examples/s]32636 examples [00:38, 835.29 examples/s]32721 examples [00:38, 837.18 examples/s]32805 examples [00:38, 808.56 examples/s]32888 examples [00:39, 812.30 examples/s]32970 examples [00:39, 799.12 examples/s]33054 examples [00:39, 809.81 examples/s]33139 examples [00:39, 819.20 examples/s]33225 examples [00:39, 828.94 examples/s]33309 examples [00:39, 822.80 examples/s]33392 examples [00:39, 789.18 examples/s]33472 examples [00:39, 786.59 examples/s]33558 examples [00:39, 804.87 examples/s]33641 examples [00:40, 810.20 examples/s]33725 examples [00:40, 817.89 examples/s]33814 examples [00:40, 835.92 examples/s]33899 examples [00:40, 839.90 examples/s]33984 examples [00:40, 829.75 examples/s]34072 examples [00:40, 842.28 examples/s]34157 examples [00:40, 843.97 examples/s]34243 examples [00:40, 846.74 examples/s]34328 examples [00:40, 840.69 examples/s]34414 examples [00:40, 846.22 examples/s]34500 examples [00:41, 848.26 examples/s]34585 examples [00:41, 841.55 examples/s]34670 examples [00:41, 843.23 examples/s]34755 examples [00:41, 818.84 examples/s]34838 examples [00:41, 813.41 examples/s]34925 examples [00:41, 827.70 examples/s]35011 examples [00:41, 836.17 examples/s]35097 examples [00:41, 841.76 examples/s]35182 examples [00:41, 840.15 examples/s]35267 examples [00:41, 824.48 examples/s]35351 examples [00:42, 827.03 examples/s]35436 examples [00:42, 832.36 examples/s]35525 examples [00:42, 847.41 examples/s]35614 examples [00:42, 859.34 examples/s]35701 examples [00:42, 858.30 examples/s]35790 examples [00:42, 865.50 examples/s]35877 examples [00:42, 866.74 examples/s]35964 examples [00:42, 862.99 examples/s]36051 examples [00:42, 861.31 examples/s]36138 examples [00:42, 862.97 examples/s]36225 examples [00:43, 864.59 examples/s]36313 examples [00:43, 868.49 examples/s]36401 examples [00:43, 868.87 examples/s]36488 examples [00:43, 867.28 examples/s]36575 examples [00:43, 858.32 examples/s]36663 examples [00:43, 863.47 examples/s]36753 examples [00:43, 871.41 examples/s]36841 examples [00:43, 861.89 examples/s]36928 examples [00:43, 862.34 examples/s]37015 examples [00:43, 862.47 examples/s]37102 examples [00:44, 859.92 examples/s]37189 examples [00:44, 856.14 examples/s]37277 examples [00:44, 862.47 examples/s]37364 examples [00:44, 859.31 examples/s]37451 examples [00:44, 861.03 examples/s]37539 examples [00:44, 864.96 examples/s]37627 examples [00:44, 868.70 examples/s]37715 examples [00:44, 870.35 examples/s]37803 examples [00:44, 860.53 examples/s]37890 examples [00:45, 826.51 examples/s]37973 examples [00:45, 820.29 examples/s]38059 examples [00:45, 831.60 examples/s]38145 examples [00:45, 838.79 examples/s]38230 examples [00:45, 840.44 examples/s]38316 examples [00:45, 844.57 examples/s]38403 examples [00:45, 850.31 examples/s]38491 examples [00:45, 857.13 examples/s]38578 examples [00:45, 860.20 examples/s]38666 examples [00:45, 864.66 examples/s]38754 examples [00:46, 868.32 examples/s]38843 examples [00:46, 873.53 examples/s]38931 examples [00:46, 872.57 examples/s]39019 examples [00:46, 864.40 examples/s]39106 examples [00:46, 864.18 examples/s]39194 examples [00:46, 865.95 examples/s]39281 examples [00:46, 864.16 examples/s]39368 examples [00:46, 863.84 examples/s]39455 examples [00:46, 865.13 examples/s]39542 examples [00:46, 863.73 examples/s]39629 examples [00:47, 856.52 examples/s]39715 examples [00:47, 844.99 examples/s]39802 examples [00:47, 850.49 examples/s]39888 examples [00:47, 852.44 examples/s]39974 examples [00:47, 853.22 examples/s]40060 examples [00:47, 801.37 examples/s]40152 examples [00:47, 831.69 examples/s]40243 examples [00:47, 852.14 examples/s]40332 examples [00:47, 861.37 examples/s]40419 examples [00:47, 859.56 examples/s]40506 examples [00:48, 845.81 examples/s]40591 examples [00:48, 845.66 examples/s]40676 examples [00:48, 843.49 examples/s]40761 examples [00:48, 843.10 examples/s]40846 examples [00:48, 845.00 examples/s]40931 examples [00:48, 823.76 examples/s]41015 examples [00:48, 827.11 examples/s]41098 examples [00:48, 816.90 examples/s]41184 examples [00:48, 827.10 examples/s]41271 examples [00:48, 837.09 examples/s]41355 examples [00:49, 828.75 examples/s]41443 examples [00:49, 841.08 examples/s]41531 examples [00:49, 851.29 examples/s]41620 examples [00:49, 861.41 examples/s]41708 examples [00:49, 865.75 examples/s]41795 examples [00:49, 857.22 examples/s]41881 examples [00:49, 849.91 examples/s]41967 examples [00:49, 844.49 examples/s]42052 examples [00:49, 821.55 examples/s]42135 examples [00:50, 818.69 examples/s]42217 examples [00:50, 807.78 examples/s]42298 examples [00:50, 796.78 examples/s]42384 examples [00:50, 813.31 examples/s]42466 examples [00:50, 797.31 examples/s]42548 examples [00:50, 801.53 examples/s]42629 examples [00:50, 787.63 examples/s]42712 examples [00:50, 797.53 examples/s]42797 examples [00:50, 811.95 examples/s]42879 examples [00:50, 812.34 examples/s]42963 examples [00:51, 819.24 examples/s]43046 examples [00:51, 794.38 examples/s]43131 examples [00:51, 809.03 examples/s]43219 examples [00:51, 828.69 examples/s]43304 examples [00:51, 834.23 examples/s]43390 examples [00:51, 840.40 examples/s]43475 examples [00:51, 824.88 examples/s]43558 examples [00:51, 789.88 examples/s]43638 examples [00:51, 790.55 examples/s]43718 examples [00:51, 790.83 examples/s]43798 examples [00:52, 792.40 examples/s]43880 examples [00:52, 797.86 examples/s]43962 examples [00:52, 803.95 examples/s]44046 examples [00:52, 812.45 examples/s]44129 examples [00:52, 816.54 examples/s]44211 examples [00:52, 814.68 examples/s]44294 examples [00:52, 817.20 examples/s]44376 examples [00:52, 814.40 examples/s]44461 examples [00:52, 823.32 examples/s]44544 examples [00:52, 824.39 examples/s]44627 examples [00:53, 811.87 examples/s]44711 examples [00:53, 817.59 examples/s]44793 examples [00:53, 809.42 examples/s]44876 examples [00:53, 813.77 examples/s]44958 examples [00:53, 809.14 examples/s]45040 examples [00:53, 811.86 examples/s]45122 examples [00:53, 811.40 examples/s]45204 examples [00:53, 804.93 examples/s]45290 examples [00:53, 819.36 examples/s]45373 examples [00:54, 820.18 examples/s]45458 examples [00:54, 826.44 examples/s]45541 examples [00:54, 817.45 examples/s]45629 examples [00:54, 834.63 examples/s]45714 examples [00:54, 837.04 examples/s]45798 examples [00:54, 836.33 examples/s]45882 examples [00:54, 833.42 examples/s]45966 examples [00:54, 833.01 examples/s]46050 examples [00:54, 819.52 examples/s]46134 examples [00:54, 824.45 examples/s]46221 examples [00:55, 836.85 examples/s]46305 examples [00:55, 827.69 examples/s]46393 examples [00:55, 841.77 examples/s]46478 examples [00:55, 842.44 examples/s]46563 examples [00:55, 837.64 examples/s]46647 examples [00:55, 832.90 examples/s]46731 examples [00:55, 824.78 examples/s]46814 examples [00:55, 812.72 examples/s]46899 examples [00:55, 823.01 examples/s]46985 examples [00:55, 832.33 examples/s]47072 examples [00:56, 841.58 examples/s]47157 examples [00:56, 843.58 examples/s]47245 examples [00:56, 852.88 examples/s]47332 examples [00:56, 856.12 examples/s]47420 examples [00:56, 861.02 examples/s]47507 examples [00:56, 850.21 examples/s]47593 examples [00:56, 852.85 examples/s]47679 examples [00:56, 830.24 examples/s]47763 examples [00:56, 819.77 examples/s]47847 examples [00:56, 823.19 examples/s]47934 examples [00:57, 834.98 examples/s]48020 examples [00:57, 841.70 examples/s]48105 examples [00:57, 835.03 examples/s]48189 examples [00:57, 830.91 examples/s]48273 examples [00:57, 829.04 examples/s]48356 examples [00:57, 825.43 examples/s]48441 examples [00:57, 830.13 examples/s]48525 examples [00:57, 832.07 examples/s]48609 examples [00:57, 833.64 examples/s]48693 examples [00:57, 832.49 examples/s]48777 examples [00:58, 832.05 examples/s]48862 examples [00:58, 834.24 examples/s]48946 examples [00:58, 831.92 examples/s]49030 examples [00:58, 832.98 examples/s]49115 examples [00:58, 837.72 examples/s]49199 examples [00:58, 834.22 examples/s]49283 examples [00:58, 835.56 examples/s]49367 examples [00:58, 831.09 examples/s]49451 examples [00:58, 832.45 examples/s]49536 examples [00:59, 835.87 examples/s]49623 examples [00:59, 845.46 examples/s]49708 examples [00:59, 835.68 examples/s]49795 examples [00:59, 844.27 examples/s]49883 examples [00:59, 852.75 examples/s]49969 examples [00:59, 846.72 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5159/50000 [00:00<00:00, 51585.50 examples/s] 31%|███       | 15505/50000 [00:00<00:00, 60717.74 examples/s] 52%|█████▏    | 25774/50000 [00:00<00:00, 69202.09 examples/s] 73%|███████▎  | 36380/50000 [00:00<00:00, 77254.86 examples/s] 93%|█████████▎| 46433/50000 [00:00<00:00, 83019.15 examples/s]                                                               0 examples [00:00, ? examples/s]55 examples [00:00, 549.12 examples/s]143 examples [00:00, 617.64 examples/s]225 examples [00:00, 665.68 examples/s]311 examples [00:00, 713.94 examples/s]400 examples [00:00, 757.56 examples/s]480 examples [00:00, 767.50 examples/s]566 examples [00:00, 791.14 examples/s]653 examples [00:00, 811.68 examples/s]733 examples [00:00, 632.20 examples/s]818 examples [00:01, 684.10 examples/s]904 examples [00:01, 726.61 examples/s]988 examples [00:01, 755.38 examples/s]1072 examples [00:01, 778.23 examples/s]1157 examples [00:01, 798.39 examples/s]1239 examples [00:01, 803.01 examples/s]1321 examples [00:01, 800.36 examples/s]1403 examples [00:01, 803.74 examples/s]1487 examples [00:01, 812.11 examples/s]1569 examples [00:02, 803.63 examples/s]1650 examples [00:02, 766.08 examples/s]1735 examples [00:02, 787.96 examples/s]1817 examples [00:02, 795.79 examples/s]1901 examples [00:02, 807.17 examples/s]1988 examples [00:02, 822.56 examples/s]2076 examples [00:02, 837.48 examples/s]2161 examples [00:02, 840.18 examples/s]2246 examples [00:02, 841.02 examples/s]2332 examples [00:02, 843.74 examples/s]2417 examples [00:03, 843.39 examples/s]2502 examples [00:03, 816.14 examples/s]2585 examples [00:03, 818.81 examples/s]2668 examples [00:03, 801.17 examples/s]2752 examples [00:03, 810.93 examples/s]2835 examples [00:03, 814.70 examples/s]2920 examples [00:03, 824.16 examples/s]3007 examples [00:03, 836.70 examples/s]3094 examples [00:03, 845.56 examples/s]3180 examples [00:03, 848.79 examples/s]3266 examples [00:04, 851.15 examples/s]3353 examples [00:04, 855.64 examples/s]3440 examples [00:04, 857.73 examples/s]3527 examples [00:04, 858.84 examples/s]3613 examples [00:04, 858.32 examples/s]3701 examples [00:04, 862.78 examples/s]3788 examples [00:04, 861.00 examples/s]3875 examples [00:04, 859.23 examples/s]3961 examples [00:04, 852.53 examples/s]4047 examples [00:04, 852.93 examples/s]4133 examples [00:05, 836.87 examples/s]4217 examples [00:05, 832.45 examples/s]4304 examples [00:05, 841.09 examples/s]4391 examples [00:05, 846.52 examples/s]4477 examples [00:05, 850.34 examples/s]4563 examples [00:05, 845.86 examples/s]4648 examples [00:05, 823.93 examples/s]4732 examples [00:05, 826.31 examples/s]4815 examples [00:05, 824.81 examples/s]4901 examples [00:05, 832.40 examples/s]4985 examples [00:06, 834.23 examples/s]5071 examples [00:06, 841.56 examples/s]5157 examples [00:06, 846.63 examples/s]5243 examples [00:06, 849.80 examples/s]5330 examples [00:06, 853.80 examples/s]5416 examples [00:06, 852.03 examples/s]5502 examples [00:06, 852.10 examples/s]5588 examples [00:06, 844.74 examples/s]5674 examples [00:06, 848.51 examples/s]5764 examples [00:07, 861.58 examples/s]5852 examples [00:07, 864.72 examples/s]5939 examples [00:07, 862.90 examples/s]6026 examples [00:07, 827.42 examples/s]6111 examples [00:07, 832.32 examples/s]6196 examples [00:07, 837.42 examples/s]6280 examples [00:07, 837.95 examples/s]6368 examples [00:07, 847.66 examples/s]6456 examples [00:07, 856.06 examples/s]6542 examples [00:07, 856.77 examples/s]6628 examples [00:08, 851.85 examples/s]6714 examples [00:08, 842.76 examples/s]6800 examples [00:08, 845.39 examples/s]6889 examples [00:08, 856.18 examples/s]6979 examples [00:08, 867.41 examples/s]7066 examples [00:08, 860.63 examples/s]7153 examples [00:08, 859.91 examples/s]7240 examples [00:08, 860.66 examples/s]7327 examples [00:08, 861.09 examples/s]7414 examples [00:08, 861.36 examples/s]7501 examples [00:09, 857.72 examples/s]7587 examples [00:09, 857.64 examples/s]7674 examples [00:09, 859.17 examples/s]7761 examples [00:09, 859.56 examples/s]7847 examples [00:09, 837.73 examples/s]7933 examples [00:09, 843.51 examples/s]8018 examples [00:09, 834.33 examples/s]8104 examples [00:09, 841.22 examples/s]8192 examples [00:09, 852.49 examples/s]8280 examples [00:09, 858.00 examples/s]8366 examples [00:10, 857.38 examples/s]8453 examples [00:10, 860.99 examples/s]8540 examples [00:10, 855.35 examples/s]8626 examples [00:10, 850.06 examples/s]8712 examples [00:10, 838.70 examples/s]8796 examples [00:10, 825.36 examples/s]8879 examples [00:10, 825.24 examples/s]8964 examples [00:10, 831.27 examples/s]9050 examples [00:10, 837.59 examples/s]9137 examples [00:10, 844.47 examples/s]9222 examples [00:11, 836.05 examples/s]9309 examples [00:11, 845.82 examples/s]9396 examples [00:11, 850.51 examples/s]9483 examples [00:11, 852.85 examples/s]9569 examples [00:11, 851.13 examples/s]9655 examples [00:11, 841.08 examples/s]9740 examples [00:11, 836.86 examples/s]9825 examples [00:11, 838.31 examples/s]9911 examples [00:11, 843.52 examples/s]9998 examples [00:12, 847.34 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRUPROX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRUPROX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdd645808c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdd645808c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdd645808c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fddd0a23da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd4dd63400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdd645808c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdd645808c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdd645808c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd47bc3470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdd47bc3320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fdd3e03d6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fdd3e03d6a8> 

  function with postional parmater data_info <function split_train_valid at 0x7fdd3e03d6a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fdd3e03d7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fdd3e03d7b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fdd3e03d7b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=6971fb2c3cab2c9d465cc88d1c98fa6c9fdadf4410135ed007feaff1f19c3456
  Stored in directory: /tmp/pip-ephem-wheel-cache-bqkva6lr/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:21:22, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:04:52, 18.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:12:40, 26.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:27:24, 37.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:30:32, 52.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.75M/862M [00:01<3:08:07, 75.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:01<2:10:54, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:31:07, 154kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<1:03:27, 219kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<44:12, 313kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:01<30:49, 445kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<21:31, 633kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<15:03, 899kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:02<11:28, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<09:54, 1.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<09:12, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<07:00, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<07:11, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:06<06:26, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<04:48, 2.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<06:23, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<07:03, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<05:36, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<06:04, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:10<05:39, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<04:17, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:01, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<05:29, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<03:59, 3.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<11:17, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<09:31, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:14<07:00, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<07:31, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<08:26, 1.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:16<06:36, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<04:47, 2.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<08:09, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<07:20, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<05:31, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<06:28, 1.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<07:50, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:21<04:35, 2.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<06:19, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:46, 2.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<04:28, 2.86MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:23<03:21, 3.80MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<09:19, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<09:48, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<07:32, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<05:30, 2.30MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:17, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:25, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:50, 2.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<03:35, 3.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:30, 1.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:53, 1.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:42, 1.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<05:32, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<10:35, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:59, 1.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:08, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:03, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:18, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:33, 2.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:23, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:07, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<06:04, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:43, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:53, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:11, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:32, 2.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:39, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:11, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:42, 2.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:44, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:11, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:14, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:51, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:22, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:13, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:46, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:50, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:14, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:30, 2.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:09, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<06:00, 1.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:38, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:33, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:53, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:22, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:59, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:38, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:18, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:23, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:31, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<04:00, 2.92MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:03, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:45, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:40, 2.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<8:28:07, 22.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<5:55:59, 32.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<4:08:35, 46.6kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<2:58:06, 64.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<2:07:27, 90.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<1:29:44, 129kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:02:48, 183kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<47:57, 240kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<34:59, 328kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<24:48, 462kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<19:36, 583kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<16:30, 692kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<12:08, 939kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<08:38, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<10:14, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:35, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<06:21, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:42, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<07:35, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:00, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<04:22, 2.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:26, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:58, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:55, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:22, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:19, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:49, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:14, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:19, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:57, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:51, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:18, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:06, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:32, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:02, 2.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:44, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:04, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:35, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:24, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:25, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:42, 2.92MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:23, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:49, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:23, 2.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:13, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:24, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:03, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:42, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:58, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:28, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:06, 2.59MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:49, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:39, 2.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:35, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:32, 2.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:34, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:44, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:34, 2.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:19, 3.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:51, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:05, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:34, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:17, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:20, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:36, 2.84MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<06:05, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:31, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:55, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:45, 2.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:39, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:51, 1.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:38, 2.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:21, 2.99MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:05, 1.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:12, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:15, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:55, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:42, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:35, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:44, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:38, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:23, 2.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:00, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:37, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:57, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:38, 2.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<08:03, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:51, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:05, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:30, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:55, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:39, 2.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:39, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:43, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:36, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:21, 2.85MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:47, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<06:30, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:48, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:24, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<06:12, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:49, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:31, 2.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:04, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:50, 2.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:33, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:11, 2.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:21, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:20, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:08, 2.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:07, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:09, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:34, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:01, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<04:38, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:31, 2.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:16, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:00, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:04, 2.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:58, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:05, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<04:07, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:59, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:07, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<06:04, 1.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:31, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:43, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:33, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:17, 2.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:16, 1.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:10, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:35, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:31, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:23, 2.56MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:07, 2.10MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:06, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:04, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:57, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:34, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:58, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:44, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:19, 1.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:05, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:01, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<02:55, 2.90MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:34, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:11, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:04, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:59, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:53, 2.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:58, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:13, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:51, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:21, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:09, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:07, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<02:58, 2.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:41, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:32, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:07, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:30, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:13, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:05, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:57, 2.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:05, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:33, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:23, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:58, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:44, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:48, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<02:46, 2.85MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:15, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:22, 1.47MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:57, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:20, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:01, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:02, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:41, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:32, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:35, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:30, 3.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<5:30:54, 23.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<3:51:43, 33.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<2:41:31, 47.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<1:55:54, 65.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<1:22:58, 91.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<58:28, 130kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<40:49, 185kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<32:49, 230kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<23:54, 315kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<16:55, 444kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<13:17, 562kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<10:13, 730kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<07:22, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:37, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<06:16, 1.18MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:47, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<09:43, 754kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<07:35, 965kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<05:28, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:56, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<06:43, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:35, 1.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:34, 2.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:19, 1.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:27, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:20, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:45, 1.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:28, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<03:31, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:32, 2.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:27, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:41, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:29, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:14, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:20, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:25, 2.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<10:04, 686kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:50, 882kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:37, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:22, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:31, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:17, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<03:04, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:52, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:57, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:38, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:53, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:21, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:28, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:30, 2.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:24, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:36, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:23, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:41, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:23, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:32, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:05, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:44, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:57, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:09, 2.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:38, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:14, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:27, 2.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:59, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:44, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:56, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:08, 2.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:44, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:24, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:34, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:02, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:30, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:48, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:01, 3.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<08:00, 768kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:17, 978kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:33, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:29, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:42, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:40, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:38, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:16, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:25, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:16, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:10, 1.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:23, 2.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:51, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:05, 2.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:14, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:34, 2.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:52, 3.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:36, 1.59MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:14, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:26, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:50, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:26, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:42, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:57, 2.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:55, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:25, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:33, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:26, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:00, 2.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:30, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:50, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:03, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:27, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:45, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:59, 2.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:25, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:45, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:59, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:45, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:03, 2.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:28, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:00, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:26, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:45, 2.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:06, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:30, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:36, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:49, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:12, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:30, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:50, 2.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:44, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:28, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:52, 2.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:21, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:50, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:17, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:40, 2.90MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:55, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:20, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<02:28, 1.94MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:41, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:06, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:26, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:46, 2.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:33, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:04, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<02:17, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:31, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:20, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:46, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:09, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:40, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:07, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:34, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:03, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:14, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:27, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:51, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:13, 1.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:36, 2.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:38, 1.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:46, 2.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:05, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:33, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:03, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:29, 2.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:23, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:53, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:07, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:18, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:37, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:05, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<01:30, 2.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:22, 1.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:52, 1.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:07, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:16, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:37, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:03, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<01:29, 2.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:14, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:45, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:02, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:11, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:01, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:31, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:44, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:20, 2.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:40, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:38, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:15, 2.94MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:03, 3.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<2:48:27, 21.7kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<1:57:48, 31.0kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<1:21:32, 44.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<58:50, 61.0kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<42:02, 85.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<29:33, 121kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<20:30, 172kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<16:16, 216kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<11:49, 297kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<08:18, 420kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<06:27, 535kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:56, 698kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<03:32, 967kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:07, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<02:31, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:50, 1.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:20, 2.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:46, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:47, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:09, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:31, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:47, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:21, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:43, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:50, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:03, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:38, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:10, 2.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:35, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:11, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:37, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:44, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:59, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:33, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:06, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:00, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:46, 1.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:19, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:29, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:23, 2.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:01, 2.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:30, 1.88MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:24, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:03, 2.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:17, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:36, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:17, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:55, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:09, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:49, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:29, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:42, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:21, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:58, 2.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:02, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:44, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:16, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:23, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:34, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:13, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:55, 2.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:09, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:04, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:49, 2.91MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:04, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:17, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:01, 2.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:43, 3.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:08, 724kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:28, 916kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:47, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:40, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:43, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:19, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:57, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:10, 1.81MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:05, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:49, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:13, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:43, 2.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:56, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:53, 2.23MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:40, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:05, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:37, 2.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:25, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:13, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:58, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:07, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:52, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:38, 2.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:55, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:51, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:38, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:54, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:42, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:30, 3.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:50, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:45, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:34, 2.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:42, 2.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:52, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:29, 2.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:46, 1.86MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:42, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:31, 2.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:39, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:47, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:27, 2.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:03, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:53, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:39, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:47, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:36, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:25, 2.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:41, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:37, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:32, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:23, 2.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:20, 2.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:25, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:31, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:17, 3.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:42, 1.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:35, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:26, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:27, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:23, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:59, 759kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:46, 955kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:33, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:31, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:16, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:27, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:23, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:16, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:17, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:16, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:10, 2.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:22, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:13, 1.98MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:07, 2.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:12, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:04, 2.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:01, 2.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 713kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 904kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 1.24MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:09:30,  9.14it/s]  0%|          | 692/400000 [00:00<8:30:03, 13.05it/s]  0%|          | 1385/400000 [00:00<5:56:42, 18.62it/s]  1%|          | 2105/400000 [00:00<4:09:31, 26.58it/s]  1%|          | 2813/400000 [00:00<2:54:38, 37.91it/s]  1%|          | 3532/400000 [00:00<2:02:17, 54.03it/s]  1%|          | 4238/400000 [00:00<1:25:44, 76.93it/s]  1%|          | 4921/400000 [00:00<1:00:12, 109.38it/s]  1%|▏         | 5607/400000 [00:00<42:21, 155.19it/s]    2%|▏         | 6286/400000 [00:01<29:53, 219.55it/s]  2%|▏         | 6965/400000 [00:01<21:10, 309.35it/s]  2%|▏         | 7667/400000 [00:01<15:04, 433.73it/s]  2%|▏         | 8345/400000 [00:01<10:49, 602.97it/s]  2%|▏         | 9053/400000 [00:01<07:50, 831.01it/s]  2%|▏         | 9764/400000 [00:01<05:45, 1130.48it/s]  3%|▎         | 10489/400000 [00:01<04:17, 1513.70it/s]  3%|▎         | 11196/400000 [00:01<03:16, 1980.62it/s]  3%|▎         | 11900/400000 [00:01<02:34, 2517.49it/s]  3%|▎         | 12599/400000 [00:01<02:04, 3110.30it/s]  3%|▎         | 13306/400000 [00:02<01:43, 3738.17it/s]  4%|▎         | 14015/400000 [00:02<01:28, 4355.77it/s]  4%|▎         | 14718/400000 [00:02<01:18, 4916.11it/s]  4%|▍         | 15421/400000 [00:02<01:11, 5373.60it/s]  4%|▍         | 16119/400000 [00:02<01:07, 5718.95it/s]  4%|▍         | 16833/400000 [00:02<01:03, 6080.85it/s]  4%|▍         | 17542/400000 [00:02<01:00, 6350.46it/s]  5%|▍         | 18267/400000 [00:02<00:57, 6595.60it/s]  5%|▍         | 18975/400000 [00:02<00:56, 6689.15it/s]  5%|▍         | 19678/400000 [00:02<00:56, 6787.08it/s]  5%|▌         | 20381/400000 [00:03<00:55, 6800.28it/s]  5%|▌         | 21078/400000 [00:03<00:55, 6810.49it/s]  5%|▌         | 21771/400000 [00:03<00:55, 6801.52it/s]  6%|▌         | 22478/400000 [00:03<00:54, 6878.02it/s]  6%|▌         | 23172/400000 [00:03<00:54, 6890.14it/s]  6%|▌         | 23880/400000 [00:03<00:54, 6945.02it/s]  6%|▌         | 24578/400000 [00:03<00:54, 6877.68it/s]  6%|▋         | 25313/400000 [00:03<00:53, 7009.27it/s]  7%|▋         | 26024/400000 [00:03<00:53, 7037.64it/s]  7%|▋         | 26730/400000 [00:03<00:53, 7021.54it/s]  7%|▋         | 27458/400000 [00:04<00:52, 7097.04it/s]  7%|▋         | 28169/400000 [00:04<00:52, 7068.93it/s]  7%|▋         | 28877/400000 [00:04<00:53, 6935.42it/s]  7%|▋         | 29572/400000 [00:04<00:54, 6857.08it/s]  8%|▊         | 30259/400000 [00:04<00:54, 6759.46it/s]  8%|▊         | 30940/400000 [00:04<00:54, 6771.57it/s]  8%|▊         | 31618/400000 [00:04<00:54, 6765.25it/s]  8%|▊         | 32296/400000 [00:04<00:54, 6742.77it/s]  8%|▊         | 32971/400000 [00:04<00:54, 6720.83it/s]  8%|▊         | 33666/400000 [00:04<00:53, 6787.55it/s]  9%|▊         | 34346/400000 [00:05<00:53, 6788.58it/s]  9%|▉         | 35026/400000 [00:05<00:54, 6705.58it/s]  9%|▉         | 35697/400000 [00:05<00:54, 6649.36it/s]  9%|▉         | 36363/400000 [00:05<00:54, 6633.29it/s]  9%|▉         | 37051/400000 [00:05<00:54, 6703.94it/s]  9%|▉         | 37722/400000 [00:05<00:54, 6690.61it/s] 10%|▉         | 38419/400000 [00:05<00:53, 6771.43it/s] 10%|▉         | 39097/400000 [00:05<00:53, 6762.09it/s] 10%|▉         | 39783/400000 [00:05<00:53, 6789.00it/s] 10%|█         | 40463/400000 [00:05<00:53, 6739.13it/s] 10%|█         | 41144/400000 [00:06<00:53, 6758.21it/s] 10%|█         | 41821/400000 [00:06<00:53, 6737.01it/s] 11%|█         | 42512/400000 [00:06<00:52, 6787.35it/s] 11%|█         | 43198/400000 [00:06<00:52, 6806.71it/s] 11%|█         | 43893/400000 [00:06<00:52, 6848.08it/s] 11%|█         | 44578/400000 [00:06<00:52, 6831.92it/s] 11%|█▏        | 45262/400000 [00:06<00:52, 6813.93it/s] 11%|█▏        | 45955/400000 [00:06<00:51, 6844.31it/s] 12%|█▏        | 46640/400000 [00:06<00:52, 6721.11it/s] 12%|█▏        | 47313/400000 [00:06<00:52, 6715.35it/s] 12%|█▏        | 48010/400000 [00:07<00:51, 6788.35it/s] 12%|█▏        | 48700/400000 [00:07<00:51, 6820.24it/s] 12%|█▏        | 49397/400000 [00:07<00:51, 6862.80it/s] 13%|█▎        | 50084/400000 [00:07<00:51, 6836.97it/s] 13%|█▎        | 50768/400000 [00:07<00:51, 6731.16it/s] 13%|█▎        | 51463/400000 [00:07<00:51, 6793.41it/s] 13%|█▎        | 52153/400000 [00:07<00:50, 6823.17it/s] 13%|█▎        | 52842/400000 [00:07<00:50, 6840.35it/s] 13%|█▎        | 53527/400000 [00:07<00:50, 6809.16it/s] 14%|█▎        | 54209/400000 [00:07<00:51, 6761.95it/s] 14%|█▎        | 54886/400000 [00:08<00:51, 6740.89it/s] 14%|█▍        | 55592/400000 [00:08<00:50, 6831.34it/s] 14%|█▍        | 56285/400000 [00:08<00:50, 6859.85it/s] 14%|█▍        | 56972/400000 [00:08<00:50, 6801.59it/s] 14%|█▍        | 57664/400000 [00:08<00:50, 6835.34it/s] 15%|█▍        | 58358/400000 [00:08<00:49, 6864.01it/s] 15%|█▍        | 59045/400000 [00:08<00:51, 6649.30it/s] 15%|█▍        | 59712/400000 [00:08<00:51, 6600.09it/s] 15%|█▌        | 60374/400000 [00:08<00:52, 6457.56it/s] 15%|█▌        | 61048/400000 [00:09<00:51, 6538.90it/s] 15%|█▌        | 61731/400000 [00:09<00:51, 6621.04it/s] 16%|█▌        | 62395/400000 [00:09<00:52, 6468.84it/s] 16%|█▌        | 63053/400000 [00:09<00:51, 6498.23it/s] 16%|█▌        | 63712/400000 [00:09<00:51, 6523.29it/s] 16%|█▌        | 64380/400000 [00:09<00:51, 6568.60it/s] 16%|█▋        | 65071/400000 [00:09<00:50, 6667.31it/s] 16%|█▋        | 65762/400000 [00:09<00:49, 6738.22it/s] 17%|█▋        | 66437/400000 [00:09<00:49, 6735.39it/s] 17%|█▋        | 67112/400000 [00:09<00:50, 6613.05it/s] 17%|█▋        | 67775/400000 [00:10<00:50, 6567.77it/s] 17%|█▋        | 68441/400000 [00:10<00:50, 6593.03it/s] 17%|█▋        | 69101/400000 [00:10<00:50, 6511.31it/s] 17%|█▋        | 69753/400000 [00:10<00:50, 6508.36it/s] 18%|█▊        | 70430/400000 [00:10<00:50, 6583.15it/s] 18%|█▊        | 71101/400000 [00:10<00:49, 6618.85it/s] 18%|█▊        | 71789/400000 [00:10<00:49, 6692.86it/s] 18%|█▊        | 72459/400000 [00:10<00:49, 6595.72it/s] 18%|█▊        | 73135/400000 [00:10<00:49, 6641.89it/s] 18%|█▊        | 73821/400000 [00:10<00:48, 6703.61it/s] 19%|█▊        | 74492/400000 [00:11<00:48, 6705.40it/s] 19%|█▉        | 75163/400000 [00:11<00:49, 6588.13it/s] 19%|█▉        | 75854/400000 [00:11<00:48, 6680.09it/s] 19%|█▉        | 76523/400000 [00:11<00:48, 6644.64it/s] 19%|█▉        | 77189/400000 [00:11<00:48, 6604.26it/s] 19%|█▉        | 77853/400000 [00:11<00:48, 6614.41it/s] 20%|█▉        | 78522/400000 [00:11<00:48, 6636.41it/s] 20%|█▉        | 79186/400000 [00:11<00:48, 6571.68it/s] 20%|█▉        | 79868/400000 [00:11<00:48, 6641.52it/s] 20%|██        | 80533/400000 [00:11<00:48, 6639.75it/s] 20%|██        | 81198/400000 [00:12<00:48, 6616.66it/s] 20%|██        | 81880/400000 [00:12<00:47, 6674.36it/s] 21%|██        | 82565/400000 [00:12<00:47, 6726.06it/s] 21%|██        | 83250/400000 [00:12<00:46, 6761.69it/s] 21%|██        | 83927/400000 [00:12<00:47, 6718.83it/s] 21%|██        | 84600/400000 [00:12<00:47, 6710.36it/s] 21%|██▏       | 85272/400000 [00:12<00:47, 6639.89it/s] 21%|██▏       | 85937/400000 [00:12<00:47, 6599.76it/s] 22%|██▏       | 86613/400000 [00:12<00:47, 6645.69it/s] 22%|██▏       | 87278/400000 [00:12<00:47, 6576.64it/s] 22%|██▏       | 87949/400000 [00:13<00:47, 6615.98it/s] 22%|██▏       | 88622/400000 [00:13<00:46, 6647.46it/s] 22%|██▏       | 89287/400000 [00:13<00:46, 6639.51it/s] 22%|██▏       | 89952/400000 [00:13<00:46, 6603.34it/s] 23%|██▎       | 90623/400000 [00:13<00:46, 6634.07it/s] 23%|██▎       | 91287/400000 [00:13<00:46, 6600.13it/s] 23%|██▎       | 91952/400000 [00:13<00:46, 6614.48it/s] 23%|██▎       | 92640/400000 [00:13<00:45, 6691.95it/s] 23%|██▎       | 93325/400000 [00:13<00:45, 6737.17it/s] 23%|██▎       | 93999/400000 [00:13<00:45, 6702.72it/s] 24%|██▎       | 94670/400000 [00:14<00:45, 6675.22it/s] 24%|██▍       | 95341/400000 [00:14<00:45, 6684.25it/s] 24%|██▍       | 96032/400000 [00:14<00:45, 6748.02it/s] 24%|██▍       | 96710/400000 [00:14<00:44, 6757.11it/s] 24%|██▍       | 97395/400000 [00:14<00:44, 6782.93it/s] 25%|██▍       | 98074/400000 [00:14<00:45, 6637.66it/s] 25%|██▍       | 98751/400000 [00:14<00:45, 6676.82it/s] 25%|██▍       | 99427/400000 [00:14<00:44, 6701.49it/s] 25%|██▌       | 100119/400000 [00:14<00:44, 6764.42it/s] 25%|██▌       | 100796/400000 [00:14<00:45, 6640.17it/s] 25%|██▌       | 101461/400000 [00:15<00:44, 6637.88it/s] 26%|██▌       | 102126/400000 [00:15<00:45, 6615.85it/s] 26%|██▌       | 102799/400000 [00:15<00:44, 6649.35it/s] 26%|██▌       | 103513/400000 [00:15<00:43, 6789.28it/s] 26%|██▌       | 104249/400000 [00:15<00:42, 6950.64it/s] 26%|██▌       | 104946/400000 [00:15<00:42, 6874.47it/s] 26%|██▋       | 105635/400000 [00:15<00:43, 6838.27it/s] 27%|██▋       | 106320/400000 [00:15<00:45, 6519.27it/s] 27%|██▋       | 106976/400000 [00:15<00:44, 6526.02it/s] 27%|██▋       | 107670/400000 [00:16<00:44, 6641.36it/s] 27%|██▋       | 108364/400000 [00:16<00:43, 6726.44it/s] 27%|██▋       | 109083/400000 [00:16<00:42, 6857.94it/s] 27%|██▋       | 109786/400000 [00:16<00:42, 6907.34it/s] 28%|██▊       | 110484/400000 [00:16<00:41, 6925.25it/s] 28%|██▊       | 111178/400000 [00:16<00:42, 6795.86it/s] 28%|██▊       | 111875/400000 [00:16<00:42, 6846.21it/s] 28%|██▊       | 112579/400000 [00:16<00:41, 6902.01it/s] 28%|██▊       | 113277/400000 [00:16<00:41, 6924.02it/s] 28%|██▊       | 113991/400000 [00:16<00:40, 6985.05it/s] 29%|██▊       | 114727/400000 [00:17<00:40, 7092.54it/s] 29%|██▉       | 115438/400000 [00:17<00:40, 6986.62it/s] 29%|██▉       | 116138/400000 [00:17<00:40, 6952.22it/s] 29%|██▉       | 116839/400000 [00:17<00:40, 6969.10it/s] 29%|██▉       | 117537/400000 [00:17<00:41, 6841.21it/s] 30%|██▉       | 118228/400000 [00:17<00:41, 6861.49it/s] 30%|██▉       | 118915/400000 [00:17<00:42, 6583.91it/s] 30%|██▉       | 119582/400000 [00:17<00:42, 6607.67it/s] 30%|███       | 120264/400000 [00:17<00:41, 6667.17it/s] 30%|███       | 120945/400000 [00:17<00:41, 6706.84it/s] 30%|███       | 121617/400000 [00:18<00:42, 6624.00it/s] 31%|███       | 122281/400000 [00:18<00:42, 6610.97it/s] 31%|███       | 122956/400000 [00:18<00:41, 6651.21it/s] 31%|███       | 123648/400000 [00:18<00:41, 6728.05it/s] 31%|███       | 124322/400000 [00:18<00:41, 6713.60it/s] 31%|███▏      | 125022/400000 [00:18<00:40, 6793.15it/s] 31%|███▏      | 125702/400000 [00:18<00:40, 6783.38it/s] 32%|███▏      | 126396/400000 [00:18<00:40, 6827.47it/s] 32%|███▏      | 127099/400000 [00:18<00:39, 6884.60it/s] 32%|███▏      | 127788/400000 [00:18<00:39, 6875.11it/s] 32%|███▏      | 128476/400000 [00:19<00:39, 6847.59it/s] 32%|███▏      | 129193/400000 [00:19<00:39, 6940.94it/s] 32%|███▏      | 129925/400000 [00:19<00:38, 7049.41it/s] 33%|███▎      | 130631/400000 [00:19<00:38, 7051.89it/s] 33%|███▎      | 131337/400000 [00:19<00:38, 7043.46it/s] 33%|███▎      | 132042/400000 [00:19<00:38, 6994.16it/s] 33%|███▎      | 132742/400000 [00:19<00:38, 6908.89it/s] 33%|███▎      | 133445/400000 [00:19<00:38, 6942.43it/s] 34%|███▎      | 134151/400000 [00:19<00:38, 6976.97it/s] 34%|███▎      | 134867/400000 [00:19<00:37, 7028.58it/s] 34%|███▍      | 135571/400000 [00:20<00:37, 7012.25it/s] 34%|███▍      | 136273/400000 [00:20<00:38, 6872.74it/s] 34%|███▍      | 136962/400000 [00:20<00:38, 6763.58it/s] 34%|███▍      | 137647/400000 [00:20<00:38, 6787.67it/s] 35%|███▍      | 138360/400000 [00:20<00:37, 6886.61it/s] 35%|███▍      | 139070/400000 [00:20<00:37, 6947.10it/s] 35%|███▍      | 139766/400000 [00:20<00:37, 6877.75it/s] 35%|███▌      | 140480/400000 [00:20<00:37, 6953.15it/s] 35%|███▌      | 141189/400000 [00:20<00:37, 6990.75it/s] 35%|███▌      | 141889/400000 [00:20<00:37, 6918.63it/s] 36%|███▌      | 142608/400000 [00:21<00:36, 6996.55it/s] 36%|███▌      | 143312/400000 [00:21<00:36, 7006.89it/s] 36%|███▌      | 144037/400000 [00:21<00:36, 7077.41it/s] 36%|███▌      | 144746/400000 [00:21<00:36, 7074.99it/s] 36%|███▋      | 145474/400000 [00:21<00:35, 7132.79it/s] 37%|███▋      | 146188/400000 [00:21<00:35, 7126.98it/s] 37%|███▋      | 146901/400000 [00:21<00:36, 6893.25it/s] 37%|███▋      | 147593/400000 [00:21<00:37, 6752.88it/s] 37%|███▋      | 148278/400000 [00:21<00:37, 6780.46it/s] 37%|███▋      | 148958/400000 [00:22<00:37, 6784.63it/s] 37%|███▋      | 149638/400000 [00:22<00:36, 6776.36it/s] 38%|███▊      | 150317/400000 [00:22<00:38, 6529.32it/s] 38%|███▊      | 150984/400000 [00:22<00:37, 6570.45it/s] 38%|███▊      | 151679/400000 [00:22<00:37, 6677.80it/s] 38%|███▊      | 152356/400000 [00:22<00:36, 6703.99it/s] 38%|███▊      | 153050/400000 [00:22<00:36, 6772.05it/s] 38%|███▊      | 153745/400000 [00:22<00:36, 6824.15it/s] 39%|███▊      | 154435/400000 [00:22<00:35, 6846.11it/s] 39%|███▉      | 155152/400000 [00:22<00:35, 6939.40it/s] 39%|███▉      | 155868/400000 [00:23<00:34, 7001.41it/s] 39%|███▉      | 156571/400000 [00:23<00:34, 7006.59it/s] 39%|███▉      | 157273/400000 [00:23<00:34, 6985.66it/s] 39%|███▉      | 157988/400000 [00:23<00:34, 7032.87it/s] 40%|███▉      | 158692/400000 [00:23<00:35, 6889.07it/s] 40%|███▉      | 159399/400000 [00:23<00:34, 6941.51it/s] 40%|████      | 160119/400000 [00:23<00:34, 7015.33it/s] 40%|████      | 160828/400000 [00:23<00:33, 7035.82it/s] 40%|████      | 161533/400000 [00:23<00:34, 6833.23it/s] 41%|████      | 162218/400000 [00:23<00:34, 6796.63it/s] 41%|████      | 162909/400000 [00:24<00:34, 6828.22it/s] 41%|████      | 163614/400000 [00:24<00:34, 6891.62it/s] 41%|████      | 164319/400000 [00:24<00:33, 6937.27it/s] 41%|████▏     | 165035/400000 [00:24<00:33, 7001.17it/s] 41%|████▏     | 165745/400000 [00:24<00:33, 7028.91it/s] 42%|████▏     | 166455/400000 [00:24<00:33, 7047.77it/s] 42%|████▏     | 167161/400000 [00:24<00:33, 6977.83it/s] 42%|████▏     | 167878/400000 [00:24<00:33, 7033.17it/s] 42%|████▏     | 168587/400000 [00:24<00:32, 7048.85it/s] 42%|████▏     | 169297/400000 [00:24<00:32, 7063.86it/s] 43%|████▎     | 170004/400000 [00:25<00:32, 7048.49it/s] 43%|████▎     | 170709/400000 [00:25<00:33, 6827.94it/s] 43%|████▎     | 171394/400000 [00:25<00:34, 6660.21it/s] 43%|████▎     | 172071/400000 [00:25<00:34, 6690.28it/s] 43%|████▎     | 172787/400000 [00:25<00:33, 6823.61it/s] 43%|████▎     | 173498/400000 [00:25<00:32, 6906.95it/s] 44%|████▎     | 174191/400000 [00:25<00:32, 6871.96it/s] 44%|████▎     | 174899/400000 [00:25<00:32, 6929.88it/s] 44%|████▍     | 175601/400000 [00:25<00:32, 6955.21it/s] 44%|████▍     | 176301/400000 [00:25<00:32, 6967.04it/s] 44%|████▍     | 177018/400000 [00:26<00:31, 7025.93it/s] 44%|████▍     | 177722/400000 [00:26<00:31, 6977.16it/s] 45%|████▍     | 178428/400000 [00:26<00:31, 7001.16it/s] 45%|████▍     | 179142/400000 [00:26<00:31, 7040.16it/s] 45%|████▍     | 179847/400000 [00:26<00:31, 7029.84it/s] 45%|████▌     | 180554/400000 [00:26<00:31, 7039.45it/s] 45%|████▌     | 181266/400000 [00:26<00:30, 7061.54it/s] 45%|████▌     | 181973/400000 [00:26<00:30, 7056.68it/s] 46%|████▌     | 182679/400000 [00:26<00:31, 6987.52it/s] 46%|████▌     | 183378/400000 [00:26<00:31, 6979.50it/s] 46%|████▌     | 184077/400000 [00:27<00:31, 6961.29it/s] 46%|████▌     | 184774/400000 [00:27<00:30, 6961.22it/s] 46%|████▋     | 185471/400000 [00:27<00:30, 6956.05it/s] 47%|████▋     | 186189/400000 [00:27<00:30, 7020.21it/s] 47%|████▋     | 186899/400000 [00:27<00:30, 7040.95it/s] 47%|████▋     | 187626/400000 [00:27<00:29, 7107.49it/s] 47%|████▋     | 188337/400000 [00:27<00:29, 7099.34it/s] 47%|████▋     | 189048/400000 [00:27<00:29, 7084.16it/s] 47%|████▋     | 189773/400000 [00:27<00:29, 7131.66it/s] 48%|████▊     | 190501/400000 [00:27<00:29, 7172.72it/s] 48%|████▊     | 191219/400000 [00:28<00:29, 7108.60it/s] 48%|████▊     | 191931/400000 [00:28<00:29, 7110.37it/s] 48%|████▊     | 192643/400000 [00:28<00:29, 7095.79it/s] 48%|████▊     | 193368/400000 [00:28<00:28, 7136.84it/s] 49%|████▊     | 194102/400000 [00:28<00:28, 7195.73it/s] 49%|████▊     | 194831/400000 [00:28<00:28, 7222.07it/s] 49%|████▉     | 195564/400000 [00:28<00:28, 7253.43it/s] 49%|████▉     | 196290/400000 [00:28<00:28, 7223.70it/s] 49%|████▉     | 197013/400000 [00:28<00:28, 7221.05it/s] 49%|████▉     | 197747/400000 [00:28<00:27, 7255.50it/s] 50%|████▉     | 198489/400000 [00:29<00:27, 7300.83it/s] 50%|████▉     | 199220/400000 [00:29<00:27, 7283.19it/s] 50%|████▉     | 199949/400000 [00:29<00:27, 7267.80it/s] 50%|█████     | 200690/400000 [00:29<00:27, 7307.55it/s] 50%|█████     | 201421/400000 [00:29<00:27, 7241.42it/s] 51%|█████     | 202153/400000 [00:29<00:27, 7264.33it/s] 51%|█████     | 202883/400000 [00:29<00:27, 7272.70it/s] 51%|█████     | 203611/400000 [00:29<00:27, 7199.11it/s] 51%|█████     | 204332/400000 [00:29<00:27, 7187.14it/s] 51%|█████▏    | 205058/400000 [00:30<00:27, 7207.58it/s] 51%|█████▏    | 205779/400000 [00:30<00:27, 7084.13it/s] 52%|█████▏    | 206507/400000 [00:30<00:27, 7140.90it/s] 52%|█████▏    | 207222/400000 [00:30<00:27, 7131.66it/s] 52%|█████▏    | 207936/400000 [00:30<00:27, 6975.45it/s] 52%|█████▏    | 208670/400000 [00:30<00:27, 7079.53it/s] 52%|█████▏    | 209400/400000 [00:30<00:26, 7142.60it/s] 53%|█████▎    | 210131/400000 [00:30<00:26, 7190.47it/s] 53%|█████▎    | 210865/400000 [00:30<00:26, 7234.66it/s] 53%|█████▎    | 211590/400000 [00:30<00:26, 7227.81it/s] 53%|█████▎    | 212337/400000 [00:31<00:25, 7297.41it/s] 53%|█████▎    | 213073/400000 [00:31<00:25, 7313.77it/s] 53%|█████▎    | 213805/400000 [00:31<00:25, 7301.89it/s] 54%|█████▎    | 214536/400000 [00:31<00:25, 7180.31it/s] 54%|█████▍    | 215287/400000 [00:31<00:25, 7273.92it/s] 54%|█████▍    | 216033/400000 [00:31<00:25, 7326.86it/s] 54%|█████▍    | 216776/400000 [00:31<00:24, 7356.02it/s] 54%|█████▍    | 217526/400000 [00:31<00:24, 7395.65it/s] 55%|█████▍    | 218266/400000 [00:31<00:24, 7361.84it/s] 55%|█████▍    | 219003/400000 [00:31<00:24, 7323.51it/s] 55%|█████▍    | 219736/400000 [00:32<00:24, 7309.65it/s] 55%|█████▌    | 220468/400000 [00:32<00:24, 7306.18it/s] 55%|█████▌    | 221202/400000 [00:32<00:24, 7315.33it/s] 55%|█████▌    | 221934/400000 [00:32<00:24, 7260.52it/s] 56%|█████▌    | 222661/400000 [00:32<00:24, 7243.56it/s] 56%|█████▌    | 223390/400000 [00:32<00:24, 7256.44it/s] 56%|█████▌    | 224129/400000 [00:32<00:24, 7293.48it/s] 56%|█████▌    | 224859/400000 [00:32<00:24, 7127.61it/s] 56%|█████▋    | 225573/400000 [00:32<00:24, 7081.43it/s] 57%|█████▋    | 226297/400000 [00:32<00:24, 7127.35it/s] 57%|█████▋    | 227015/400000 [00:33<00:24, 7142.96it/s] 57%|█████▋    | 227759/400000 [00:33<00:23, 7228.22it/s] 57%|█████▋    | 228494/400000 [00:33<00:23, 7263.02it/s] 57%|█████▋    | 229223/400000 [00:33<00:23, 7268.71it/s] 57%|█████▋    | 229975/400000 [00:33<00:23, 7340.54it/s] 58%|█████▊    | 230720/400000 [00:33<00:22, 7372.99it/s] 58%|█████▊    | 231458/400000 [00:33<00:23, 7263.05it/s] 58%|█████▊    | 232185/400000 [00:33<00:23, 7235.04it/s] 58%|█████▊    | 232909/400000 [00:33<00:23, 7210.68it/s] 58%|█████▊    | 233638/400000 [00:33<00:22, 7233.36it/s] 59%|█████▊    | 234382/400000 [00:34<00:22, 7293.80it/s] 59%|█████▉    | 235114/400000 [00:34<00:22, 7299.18it/s] 59%|█████▉    | 235845/400000 [00:34<00:22, 7281.87it/s] 59%|█████▉    | 236574/400000 [00:34<00:22, 7270.22it/s] 59%|█████▉    | 237333/400000 [00:34<00:22, 7360.78it/s] 60%|█████▉    | 238075/400000 [00:34<00:21, 7375.89it/s] 60%|█████▉    | 238827/400000 [00:34<00:21, 7414.97it/s] 60%|█████▉    | 239569/400000 [00:34<00:21, 7392.41it/s] 60%|██████    | 240309/400000 [00:34<00:22, 7253.60it/s] 60%|██████    | 241037/400000 [00:34<00:21, 7258.03it/s] 60%|██████    | 241773/400000 [00:35<00:21, 7286.74it/s] 61%|██████    | 242503/400000 [00:35<00:21, 7159.74it/s] 61%|██████    | 243225/400000 [00:35<00:21, 7175.29it/s] 61%|██████    | 243944/400000 [00:35<00:21, 7114.47it/s] 61%|██████    | 244656/400000 [00:35<00:21, 7084.59it/s] 61%|██████▏   | 245387/400000 [00:35<00:21, 7149.53it/s] 62%|██████▏   | 246113/400000 [00:35<00:21, 7179.45it/s] 62%|██████▏   | 246846/400000 [00:35<00:21, 7222.41it/s] 62%|██████▏   | 247572/400000 [00:35<00:21, 7232.20it/s] 62%|██████▏   | 248296/400000 [00:35<00:21, 7123.50it/s] 62%|██████▏   | 249055/400000 [00:36<00:20, 7255.42it/s] 62%|██████▏   | 249812/400000 [00:36<00:20, 7344.69it/s] 63%|██████▎   | 250548/400000 [00:36<00:20, 7346.76it/s] 63%|██████▎   | 251288/400000 [00:36<00:20, 7362.39it/s] 63%|██████▎   | 252030/400000 [00:36<00:20, 7379.34it/s] 63%|██████▎   | 252769/400000 [00:36<00:20, 7348.66it/s] 63%|██████▎   | 253531/400000 [00:36<00:19, 7426.32it/s] 64%|██████▎   | 254283/400000 [00:36<00:19, 7451.54it/s] 64%|██████▍   | 255029/400000 [00:36<00:19, 7431.22it/s] 64%|██████▍   | 255773/400000 [00:36<00:19, 7384.69it/s] 64%|██████▍   | 256525/400000 [00:37<00:19, 7422.53it/s] 64%|██████▍   | 257268/400000 [00:37<00:19, 7244.61it/s] 65%|██████▍   | 258001/400000 [00:37<00:19, 7269.11it/s] 65%|██████▍   | 258737/400000 [00:37<00:19, 7295.53it/s] 65%|██████▍   | 259471/400000 [00:37<00:19, 7307.29it/s] 65%|██████▌   | 260220/400000 [00:37<00:18, 7359.30it/s] 65%|██████▌   | 260973/400000 [00:37<00:18, 7407.15it/s] 65%|██████▌   | 261715/400000 [00:37<00:18, 7354.33it/s] 66%|██████▌   | 262461/400000 [00:37<00:18, 7383.22it/s] 66%|██████▌   | 263200/400000 [00:37<00:18, 7350.88it/s] 66%|██████▌   | 263951/400000 [00:38<00:18, 7396.02it/s] 66%|██████▌   | 264691/400000 [00:38<00:18, 7368.79it/s] 66%|██████▋   | 265429/400000 [00:38<00:18, 7349.79it/s] 67%|██████▋   | 266165/400000 [00:38<00:18, 7326.80it/s] 67%|██████▋   | 266898/400000 [00:38<00:18, 7312.78it/s] 67%|██████▋   | 267631/400000 [00:38<00:18, 7317.18it/s] 67%|██████▋   | 268363/400000 [00:38<00:18, 7239.65it/s] 67%|██████▋   | 269101/400000 [00:38<00:17, 7280.48it/s] 67%|██████▋   | 269830/400000 [00:38<00:17, 7256.29it/s] 68%|██████▊   | 270566/400000 [00:39<00:17, 7286.09it/s] 68%|██████▊   | 271307/400000 [00:39<00:17, 7322.26it/s] 68%|██████▊   | 272041/400000 [00:39<00:17, 7325.93it/s] 68%|██████▊   | 272780/400000 [00:39<00:17, 7344.82it/s] 68%|██████▊   | 273515/400000 [00:39<00:17, 7319.85it/s] 69%|██████▊   | 274248/400000 [00:39<00:17, 7114.89it/s] 69%|██████▊   | 274991/400000 [00:39<00:17, 7205.62it/s] 69%|██████▉   | 275713/400000 [00:39<00:17, 7176.57it/s] 69%|██████▉   | 276470/400000 [00:39<00:16, 7288.61it/s] 69%|██████▉   | 277224/400000 [00:39<00:16, 7360.11it/s] 69%|██████▉   | 277961/400000 [00:40<00:16, 7355.22it/s] 70%|██████▉   | 278698/400000 [00:40<00:16, 7229.06it/s] 70%|██████▉   | 279432/400000 [00:40<00:16, 7260.67it/s] 70%|███████   | 280161/400000 [00:40<00:16, 7268.44it/s] 70%|███████   | 280892/400000 [00:40<00:16, 7279.37it/s] 70%|███████   | 281622/400000 [00:40<00:16, 7283.40it/s] 71%|███████   | 282351/400000 [00:40<00:16, 7284.82it/s] 71%|███████   | 283080/400000 [00:40<00:16, 7257.84it/s] 71%|███████   | 283835/400000 [00:40<00:15, 7340.99it/s] 71%|███████   | 284572/400000 [00:40<00:15, 7347.40it/s] 71%|███████▏  | 285307/400000 [00:41<00:15, 7303.89it/s] 72%|███████▏  | 286038/400000 [00:41<00:15, 7283.10it/s] 72%|███████▏  | 286767/400000 [00:41<00:15, 7221.22it/s] 72%|███████▏  | 287496/400000 [00:41<00:15, 7239.38it/s] 72%|███████▏  | 288235/400000 [00:41<00:15, 7281.48it/s] 72%|███████▏  | 288967/400000 [00:41<00:15, 7292.79it/s] 72%|███████▏  | 289705/400000 [00:41<00:15, 7317.61it/s] 73%|███████▎  | 290437/400000 [00:41<00:14, 7306.82it/s] 73%|███████▎  | 291168/400000 [00:41<00:15, 7207.29it/s] 73%|███████▎  | 291903/400000 [00:41<00:14, 7247.96it/s] 73%|███████▎  | 292629/400000 [00:42<00:14, 7206.13it/s] 73%|███████▎  | 293365/400000 [00:42<00:14, 7251.45it/s] 74%|███████▎  | 294105/400000 [00:42<00:14, 7293.31it/s] 74%|███████▎  | 294851/400000 [00:42<00:14, 7341.00it/s] 74%|███████▍  | 295586/400000 [00:42<00:14, 7283.34it/s] 74%|███████▍  | 296315/400000 [00:42<00:14, 7281.69it/s] 74%|███████▍  | 297062/400000 [00:42<00:14, 7334.86it/s] 74%|███████▍  | 297807/400000 [00:42<00:13, 7366.53it/s] 75%|███████▍  | 298561/400000 [00:42<00:13, 7415.69it/s] 75%|███████▍  | 299312/400000 [00:42<00:13, 7440.84it/s] 75%|███████▌  | 300057/400000 [00:43<00:13, 7319.32it/s] 75%|███████▌  | 300792/400000 [00:43<00:13, 7326.11it/s] 75%|███████▌  | 301526/400000 [00:43<00:13, 7312.46it/s] 76%|███████▌  | 302258/400000 [00:43<00:13, 7309.78it/s] 76%|███████▌  | 302991/400000 [00:43<00:13, 7315.21it/s] 76%|███████▌  | 303723/400000 [00:43<00:13, 7278.83it/s] 76%|███████▌  | 304454/400000 [00:43<00:13, 7284.67it/s] 76%|███████▋  | 305183/400000 [00:43<00:13, 7261.37it/s] 76%|███████▋  | 305910/400000 [00:43<00:13, 7234.44it/s] 77%|███████▋  | 306637/400000 [00:43<00:12, 7243.47it/s] 77%|███████▋  | 307370/400000 [00:44<00:12, 7267.44it/s] 77%|███████▋  | 308097/400000 [00:44<00:12, 7207.40it/s] 77%|███████▋  | 308820/400000 [00:44<00:12, 7214.14it/s] 77%|███████▋  | 309551/400000 [00:44<00:12, 7242.14it/s] 78%|███████▊  | 310278/400000 [00:44<00:12, 7249.50it/s] 78%|███████▊  | 311004/400000 [00:44<00:12, 7248.99it/s] 78%|███████▊  | 311757/400000 [00:44<00:12, 7329.54it/s] 78%|███████▊  | 312491/400000 [00:44<00:12, 7258.55it/s] 78%|███████▊  | 313218/400000 [00:44<00:12, 7219.29it/s] 78%|███████▊  | 313957/400000 [00:44<00:11, 7267.77it/s] 79%|███████▊  | 314685/400000 [00:45<00:11, 7256.16it/s] 79%|███████▉  | 315411/400000 [00:45<00:11, 7080.01it/s] 79%|███████▉  | 316129/400000 [00:45<00:11, 7105.91it/s] 79%|███████▉  | 316842/400000 [00:45<00:11, 7112.75it/s] 79%|███████▉  | 317573/400000 [00:45<00:11, 7169.85it/s] 80%|███████▉  | 318300/400000 [00:45<00:11, 7199.39it/s] 80%|███████▉  | 319021/400000 [00:45<00:11, 7097.75it/s] 80%|███████▉  | 319744/400000 [00:45<00:11, 7136.87it/s] 80%|████████  | 320495/400000 [00:45<00:10, 7243.54it/s] 80%|████████  | 321221/400000 [00:45<00:10, 7224.97it/s] 80%|████████  | 321960/400000 [00:46<00:10, 7272.14it/s] 81%|████████  | 322688/400000 [00:46<00:10, 7242.96it/s] 81%|████████  | 323413/400000 [00:46<00:10, 7241.63it/s] 81%|████████  | 324145/400000 [00:46<00:10, 7264.70it/s] 81%|████████  | 324872/400000 [00:46<00:10, 7258.27it/s] 81%|████████▏ | 325601/400000 [00:46<00:10, 7267.72it/s] 82%|████████▏ | 326328/400000 [00:46<00:10, 7220.54it/s] 82%|████████▏ | 327063/400000 [00:46<00:10, 7257.54it/s] 82%|████████▏ | 327790/400000 [00:46<00:09, 7260.63it/s] 82%|████████▏ | 328521/400000 [00:46<00:09, 7273.32it/s] 82%|████████▏ | 329250/400000 [00:47<00:09, 7277.72it/s] 82%|████████▏ | 329978/400000 [00:47<00:09, 7167.71it/s] 83%|████████▎ | 330703/400000 [00:47<00:09, 7190.40it/s] 83%|████████▎ | 331431/400000 [00:47<00:09, 7216.08it/s] 83%|████████▎ | 332163/400000 [00:47<00:09, 7246.22it/s] 83%|████████▎ | 332897/400000 [00:47<00:09, 7272.15it/s] 83%|████████▎ | 333625/400000 [00:47<00:09, 7200.17it/s] 84%|████████▎ | 334346/400000 [00:47<00:09, 7183.42it/s] 84%|████████▍ | 335081/400000 [00:47<00:08, 7230.87it/s] 84%|████████▍ | 335805/400000 [00:47<00:08, 7215.86it/s] 84%|████████▍ | 336537/400000 [00:48<00:08, 7246.57it/s] 84%|████████▍ | 337262/400000 [00:48<00:08, 7170.60it/s] 84%|████████▍ | 337990/400000 [00:48<00:08, 7202.53it/s] 85%|████████▍ | 338735/400000 [00:48<00:08, 7274.09it/s] 85%|████████▍ | 339480/400000 [00:48<00:08, 7324.88it/s] 85%|████████▌ | 340228/400000 [00:48<00:08, 7368.43it/s] 85%|████████▌ | 340967/400000 [00:48<00:08, 7372.22it/s] 85%|████████▌ | 341705/400000 [00:48<00:08, 7278.49it/s] 86%|████████▌ | 342434/400000 [00:48<00:08, 7115.63it/s] 86%|████████▌ | 343147/400000 [00:49<00:08, 7091.44it/s] 86%|████████▌ | 343857/400000 [00:49<00:07, 7085.96it/s] 86%|████████▌ | 344567/400000 [00:49<00:07, 7053.22it/s] 86%|████████▋ | 345273/400000 [00:49<00:07, 7032.53it/s] 86%|████████▋ | 345990/400000 [00:49<00:07, 7071.08it/s] 87%|████████▋ | 346698/400000 [00:49<00:07, 7019.64it/s] 87%|████████▋ | 347416/400000 [00:49<00:07, 7065.84it/s] 87%|████████▋ | 348123/400000 [00:49<00:07, 7034.86it/s] 87%|████████▋ | 348841/400000 [00:49<00:07, 7076.98it/s] 87%|████████▋ | 349558/400000 [00:49<00:07, 7103.48it/s] 88%|████████▊ | 350269/400000 [00:50<00:07, 7095.77it/s] 88%|████████▊ | 350979/400000 [00:50<00:07, 6932.37it/s] 88%|████████▊ | 351674/400000 [00:50<00:07, 6901.19it/s] 88%|████████▊ | 352377/400000 [00:50<00:06, 6936.01it/s] 88%|████████▊ | 353090/400000 [00:50<00:06, 6992.69it/s] 88%|████████▊ | 353818/400000 [00:50<00:06, 7073.55it/s] 89%|████████▊ | 354535/400000 [00:50<00:06, 7098.90it/s] 89%|████████▉ | 355246/400000 [00:50<00:06, 6966.73it/s] 89%|████████▉ | 355973/400000 [00:50<00:06, 7052.39it/s] 89%|████████▉ | 356683/400000 [00:50<00:06, 7066.38it/s] 89%|████████▉ | 357391/400000 [00:51<00:06, 7017.62it/s] 90%|████████▉ | 358097/400000 [00:51<00:05, 7028.48it/s] 90%|████████▉ | 358801/400000 [00:51<00:05, 6870.23it/s] 90%|████████▉ | 359496/400000 [00:51<00:05, 6891.29it/s] 90%|█████████ | 360199/400000 [00:51<00:05, 6930.00it/s] 90%|█████████ | 360893/400000 [00:51<00:05, 6807.60it/s] 90%|█████████ | 361611/400000 [00:51<00:05, 6913.71it/s] 91%|█████████ | 362315/400000 [00:51<00:05, 6948.95it/s] 91%|█████████ | 363037/400000 [00:51<00:05, 7025.58it/s] 91%|█████████ | 363741/400000 [00:51<00:05, 7003.27it/s] 91%|█████████ | 364443/400000 [00:52<00:05, 7004.92it/s] 91%|█████████▏| 365152/400000 [00:52<00:04, 7029.45it/s] 91%|█████████▏| 365856/400000 [00:52<00:04, 6995.24it/s] 92%|█████████▏| 366568/400000 [00:52<00:04, 7031.18it/s] 92%|█████████▏| 367286/400000 [00:52<00:04, 7072.73it/s] 92%|█████████▏| 367994/400000 [00:52<00:04, 7048.97it/s] 92%|█████████▏| 368716/400000 [00:52<00:04, 7097.09it/s] 92%|█████████▏| 369426/400000 [00:52<00:04, 7075.53it/s] 93%|█████████▎| 370134/400000 [00:52<00:04, 7071.04it/s] 93%|█████████▎| 370858/400000 [00:52<00:04, 7120.21it/s] 93%|█████████▎| 371575/400000 [00:53<00:03, 7133.28it/s] 93%|█████████▎| 372289/400000 [00:53<00:03, 7088.50it/s] 93%|█████████▎| 372998/400000 [00:53<00:03, 7039.42it/s] 93%|█████████▎| 373720/400000 [00:53<00:03, 7090.37it/s] 94%|█████████▎| 374430/400000 [00:53<00:03, 6997.97it/s] 94%|█████████▍| 375146/400000 [00:53<00:03, 7044.51it/s] 94%|█████████▍| 375859/400000 [00:53<00:03, 7067.45it/s] 94%|█████████▍| 376567/400000 [00:53<00:03, 6819.61it/s] 94%|█████████▍| 377252/400000 [00:53<00:03, 6810.09it/s] 94%|█████████▍| 377959/400000 [00:53<00:03, 6884.59it/s] 95%|█████████▍| 378671/400000 [00:54<00:03, 6951.65it/s] 95%|█████████▍| 379373/400000 [00:54<00:02, 6970.06it/s] 95%|█████████▌| 380079/400000 [00:54<00:02, 6994.25it/s] 95%|█████████▌| 380779/400000 [00:54<00:02, 6962.74it/s] 95%|█████████▌| 381479/400000 [00:54<00:02, 6973.29it/s] 96%|█████████▌| 382182/400000 [00:54<00:02, 6989.20it/s] 96%|█████████▌| 382885/400000 [00:54<00:02, 6999.09it/s] 96%|█████████▌| 383586/400000 [00:54<00:02, 6874.37it/s] 96%|█████████▌| 384303/400000 [00:54<00:02, 6958.69it/s] 96%|█████████▋| 385000/400000 [00:54<00:02, 6935.09it/s] 96%|█████████▋| 385707/400000 [00:55<00:02, 6974.84it/s] 97%|█████████▋| 386409/400000 [00:55<00:01, 6985.92it/s] 97%|█████████▋| 387113/400000 [00:55<00:01, 6999.83it/s] 97%|█████████▋| 387817/400000 [00:55<00:01, 7009.18it/s] 97%|█████████▋| 388526/400000 [00:55<00:01, 7031.17it/s] 97%|█████████▋| 389230/400000 [00:55<00:01, 6989.54it/s] 97%|█████████▋| 389953/400000 [00:55<00:01, 7059.31it/s] 98%|█████████▊| 390660/400000 [00:55<00:01, 7038.89it/s] 98%|█████████▊| 391371/400000 [00:55<00:01, 7057.71it/s] 98%|█████████▊| 392081/400000 [00:55<00:01, 7069.23it/s] 98%|█████████▊| 392791/400000 [00:56<00:01, 7078.27it/s] 98%|█████████▊| 393499/400000 [00:56<00:00, 7032.78it/s] 99%|█████████▊| 394203/400000 [00:56<00:00, 6842.52it/s] 99%|█████████▊| 394899/400000 [00:56<00:00, 6875.17it/s] 99%|█████████▉| 395594/400000 [00:56<00:00, 6895.36it/s] 99%|█████████▉| 396299/400000 [00:56<00:00, 6939.31it/s] 99%|█████████▉| 397009/400000 [00:56<00:00, 6984.31it/s] 99%|█████████▉| 397708/400000 [00:56<00:00, 6924.83it/s]100%|█████████▉| 398423/400000 [00:56<00:00, 6987.86it/s]100%|█████████▉| 399142/400000 [00:57<00:00, 7046.11it/s]100%|█████████▉| 399864/400000 [00:57<00:00, 7095.73it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 7001.76it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdd3c6c5128>, <torchtext.data.dataset.TabularDataset object at 0x7fdd3c6c5278>, <torchtext.vocab.Vocab object at 0x7fdd3c6c5198>), {}) 

