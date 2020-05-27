
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a497d63f77051de8adc79ecd064bbf7f55ecb632', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a497d63f77051de8adc79ecd064bbf7f55ecb632

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
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
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
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
  "data_info": {
    "dataset": "mlmodels/dataset/text/imdb",
    "pass_data_pars": false,
    "train": true,
    "maxlen": 40,
    "max_features": 5
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:NumpyDataset",
      "args": {
        "numpy_loader_args": {
          "allow_pickle": true
        },
        "encoding": "'ISO-8859-1'"
      }
    },
    {
      "name": "imdb_process",
      "uri": "mlmodels.preprocess.text_keras:IMDBDataset",
      "args": {
        "num_words": 5
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
  "data_info": {
    "data_path": "dataset/text/",
    "dataset": "ner_dataset.csv",
    "pass_data_pars": false,
    "train": true
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels.preprocess.generic:pandasDataset",
      "args": {
        "read_csv_parm": {
          "encoding": "ISO-8859-1"
        },
        "colX": [],
        "coly": []
      }
    },
    {
      "uri": "mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args": {
        "max_len": 75
      },
      "internal_states": [
        "word_count"
      ]
    },
    {
      "name": "split_xy",
      "uri": "mlmodels.dataloader:split_xy_from_dict",
      "args": {
        "col_Xinput": [
          "X"
        ],
        "col_yinput": [
          "y"
        ]
      }
    },
    {
      "name": "split_train_test",
      "uri": "sklearn.model_selection:train_test_split",
      "args": {
        "test_size": 0.5
      }
    },
    {
      "name": "saver",
      "uri": "mlmodels.dataloader:pickle_dump",
      "args": {
        "path": "mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl"
      }
    }
  ],
  "output": {
    "shape": [
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len": 75
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe18a977620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe18a977620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe1f3ad1ae8> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe1f3ad1ae8> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe1a04a4378> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe1a04a4378> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/MNIST",
    "dataset": "MNIST",
    "data_type": "tch_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tch_dataset_start",
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "torchvision.datasets:MNIST",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars": false,
          "arg": {
            "fixed_size": 256,
            "path": "dataset/vision/MNIST/"
          }
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1a1357268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1a1357268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1a1357268> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3768320/9912422 [00:00<00:00, 37597093.30it/s]9920512it [00:00, 38388502.32it/s]                             
0it [00:00, ?it/s]32768it [00:00, 512879.76it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 3067326.72it/s]          
0it [00:00, ?it/s]8192it [00:00, 152637.36it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe18a91d9e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe178d60c18>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/cifar10/",
    "dataset": "CIFAR10",
    "data_type": "tf_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tf_dataset_start",
      "uri": "mlmodels.preprocess.generic:tf_dataset_download",
      "arg": {
        "train_samples": 2000,
        "test_samples": 500,
        "shuffle": true,
        "download": true
      }
    },
    {
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "mlmodels.preprocess.generic:NumpyDataset",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars": false,
          "arg": {
            "fixed_size": 256
          }
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe1a1352e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe1a1352e18> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe1a1352e18> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:16,  7.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:12, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 13.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 27.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 36.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 40.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 43.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 46.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 46.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 48.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 48.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 51.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 51.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 51.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 51.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 53.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 53.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 53.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 53.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 54.41 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 54.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 54.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 54.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 54.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.44s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.63 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 54.63 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.52s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.53s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.31 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.53s/ url]
0 examples [00:00, ? examples/s]2020-05-27 13:57:50.364748: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-27 13:57:50.368712: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-27 13:57:50.368838: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563310891cc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 13:57:50.368851: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
74 examples [00:00, 735.81 examples/s]181 examples [00:00, 810.44 examples/s]281 examples [00:00, 857.16 examples/s]383 examples [00:00, 898.85 examples/s]494 examples [00:00, 951.99 examples/s]606 examples [00:00, 994.60 examples/s]720 examples [00:00, 1032.02 examples/s]831 examples [00:00, 1053.07 examples/s]940 examples [00:00, 1063.44 examples/s]1052 examples [00:01, 1078.98 examples/s]1163 examples [00:01, 1084.93 examples/s]1272 examples [00:01, 1084.66 examples/s]1383 examples [00:01, 1089.89 examples/s]1492 examples [00:01, 1085.01 examples/s]1603 examples [00:01, 1091.62 examples/s]1712 examples [00:01, 1078.43 examples/s]1824 examples [00:01, 1088.80 examples/s]1935 examples [00:01, 1093.40 examples/s]2045 examples [00:01, 1062.73 examples/s]2156 examples [00:02, 1075.26 examples/s]2264 examples [00:02, 1061.44 examples/s]2375 examples [00:02, 1072.96 examples/s]2485 examples [00:02, 1080.49 examples/s]2594 examples [00:02, 1018.90 examples/s]2697 examples [00:02, 1012.51 examples/s]2808 examples [00:02, 1039.15 examples/s]2913 examples [00:02, 1036.47 examples/s]3020 examples [00:02, 1046.22 examples/s]3130 examples [00:02, 1059.49 examples/s]3238 examples [00:03, 1064.52 examples/s]3348 examples [00:03, 1074.67 examples/s]3459 examples [00:03, 1083.33 examples/s]3568 examples [00:03, 1047.69 examples/s]3678 examples [00:03, 1061.63 examples/s]3792 examples [00:03, 1081.69 examples/s]3906 examples [00:03, 1096.28 examples/s]4016 examples [00:03, 1052.68 examples/s]4128 examples [00:03, 1069.10 examples/s]4239 examples [00:03, 1080.36 examples/s]4348 examples [00:04, 1050.87 examples/s]4457 examples [00:04, 1059.81 examples/s]4564 examples [00:04, 1061.40 examples/s]4676 examples [00:04, 1076.06 examples/s]4784 examples [00:04, 1071.58 examples/s]4894 examples [00:04, 1078.09 examples/s]5004 examples [00:04, 1078.95 examples/s]5112 examples [00:04, 1076.08 examples/s]5225 examples [00:04, 1089.72 examples/s]5335 examples [00:05, 869.17 examples/s] 5447 examples [00:05, 931.63 examples/s]5558 examples [00:05, 977.99 examples/s]5670 examples [00:05, 1015.53 examples/s]5779 examples [00:05, 1036.08 examples/s]5890 examples [00:05, 1054.75 examples/s]6000 examples [00:05, 1065.29 examples/s]6109 examples [00:05, 1070.24 examples/s]6220 examples [00:05, 1079.64 examples/s]6329 examples [00:05, 1080.53 examples/s]6441 examples [00:06, 1090.01 examples/s]6551 examples [00:06, 1091.78 examples/s]6661 examples [00:06, 1092.00 examples/s]6771 examples [00:06, 1033.66 examples/s]6879 examples [00:06, 1045.03 examples/s]6988 examples [00:06, 1055.76 examples/s]7097 examples [00:06, 1065.27 examples/s]7208 examples [00:06, 1078.12 examples/s]7320 examples [00:06, 1087.56 examples/s]7430 examples [00:07, 1089.62 examples/s]7540 examples [00:07, 1080.84 examples/s]7649 examples [00:07, 1082.44 examples/s]7758 examples [00:07, 1065.01 examples/s]7869 examples [00:07, 1077.43 examples/s]7978 examples [00:07, 1080.88 examples/s]8087 examples [00:07, 1080.46 examples/s]8196 examples [00:07, 1073.37 examples/s]8304 examples [00:07, 1038.50 examples/s]8410 examples [00:07, 1028.24 examples/s]8514 examples [00:08, 1014.15 examples/s]8622 examples [00:08, 1030.80 examples/s]8730 examples [00:08, 1043.37 examples/s]8835 examples [00:08, 1037.47 examples/s]8945 examples [00:08, 1054.46 examples/s]9054 examples [00:08, 1051.11 examples/s]9164 examples [00:08, 1063.13 examples/s]9272 examples [00:08, 1068.13 examples/s]9381 examples [00:08, 1073.77 examples/s]9491 examples [00:08, 1081.37 examples/s]9602 examples [00:09, 1087.80 examples/s]9712 examples [00:09, 1088.74 examples/s]9821 examples [00:09, 1051.34 examples/s]9931 examples [00:09, 1063.14 examples/s]10038 examples [00:09, 990.52 examples/s]10146 examples [00:09, 1014.12 examples/s]10253 examples [00:09, 1030.09 examples/s]10364 examples [00:09, 1051.59 examples/s]10471 examples [00:09, 1056.63 examples/s]10581 examples [00:10, 1068.02 examples/s]10689 examples [00:10, 1062.26 examples/s]10797 examples [00:10, 1066.16 examples/s]10907 examples [00:10, 1073.74 examples/s]11016 examples [00:10, 1076.33 examples/s]11124 examples [00:10, 1074.37 examples/s]11234 examples [00:10, 1080.90 examples/s]11344 examples [00:10, 1086.35 examples/s]11456 examples [00:10, 1094.55 examples/s]11567 examples [00:10, 1098.92 examples/s]11678 examples [00:11, 1099.72 examples/s]11788 examples [00:11, 1099.27 examples/s]11898 examples [00:11, 1094.86 examples/s]12008 examples [00:11, 1090.24 examples/s]12118 examples [00:11, 1086.86 examples/s]12228 examples [00:11, 1089.44 examples/s]12338 examples [00:11, 1091.68 examples/s]12448 examples [00:11, 1087.17 examples/s]12557 examples [00:11, 1087.71 examples/s]12668 examples [00:11, 1092.02 examples/s]12780 examples [00:12, 1099.82 examples/s]12891 examples [00:12, 1098.93 examples/s]13001 examples [00:12, 1093.66 examples/s]13111 examples [00:12, 1082.80 examples/s]13221 examples [00:12, 1085.34 examples/s]13330 examples [00:12, 1066.42 examples/s]13439 examples [00:12, 1070.58 examples/s]13550 examples [00:12, 1081.06 examples/s]13661 examples [00:12, 1086.92 examples/s]13771 examples [00:12, 1090.21 examples/s]13881 examples [00:13, 1072.05 examples/s]13991 examples [00:13, 1078.18 examples/s]14104 examples [00:13, 1092.17 examples/s]14215 examples [00:13, 1095.60 examples/s]14326 examples [00:13, 1098.53 examples/s]14436 examples [00:13, 1098.62 examples/s]14547 examples [00:13, 1099.40 examples/s]14657 examples [00:13, 1085.55 examples/s]14766 examples [00:13, 1084.96 examples/s]14876 examples [00:13, 1087.02 examples/s]14985 examples [00:14, 1087.49 examples/s]15094 examples [00:14, 1087.59 examples/s]15203 examples [00:14, 1076.71 examples/s]15311 examples [00:14, 1058.62 examples/s]15421 examples [00:14, 1069.48 examples/s]15532 examples [00:14, 1079.44 examples/s]15641 examples [00:14, 1074.44 examples/s]15749 examples [00:14, 1075.23 examples/s]15858 examples [00:14, 1077.80 examples/s]15968 examples [00:14, 1084.10 examples/s]16079 examples [00:15, 1089.97 examples/s]16189 examples [00:15, 1087.95 examples/s]16298 examples [00:15, 1082.70 examples/s]16407 examples [00:15, 1045.51 examples/s]16517 examples [00:15, 1058.99 examples/s]16624 examples [00:15, 1028.30 examples/s]16730 examples [00:15, 1034.07 examples/s]16834 examples [00:15, 1030.48 examples/s]16944 examples [00:15, 1049.09 examples/s]17050 examples [00:16, 1052.00 examples/s]17159 examples [00:16, 1062.47 examples/s]17267 examples [00:16, 1067.08 examples/s]17377 examples [00:16, 1064.68 examples/s]17484 examples [00:16, 1060.68 examples/s]17594 examples [00:16, 1070.09 examples/s]17705 examples [00:16, 1081.43 examples/s]17814 examples [00:16, 1064.08 examples/s]17921 examples [00:16, 1032.70 examples/s]18031 examples [00:16, 1049.66 examples/s]18140 examples [00:17, 1060.10 examples/s]18250 examples [00:17, 1069.02 examples/s]18358 examples [00:17, 1068.83 examples/s]18465 examples [00:17, 1060.89 examples/s]18572 examples [00:17, 1062.92 examples/s]18679 examples [00:17, 1064.83 examples/s]18786 examples [00:17, 1050.37 examples/s]18892 examples [00:17, 1052.19 examples/s]18998 examples [00:17, 1042.38 examples/s]19104 examples [00:17, 1046.82 examples/s]19210 examples [00:18, 1048.47 examples/s]19315 examples [00:18, 1010.10 examples/s]19423 examples [00:18, 1028.84 examples/s]19531 examples [00:18, 1041.37 examples/s]19636 examples [00:18, 1025.06 examples/s]19744 examples [00:18, 1039.36 examples/s]19849 examples [00:18, 1006.75 examples/s]19954 examples [00:18, 1016.76 examples/s]20056 examples [00:18, 979.29 examples/s] 20166 examples [00:18, 1012.39 examples/s]20275 examples [00:19, 1032.38 examples/s]20383 examples [00:19, 1044.50 examples/s]20488 examples [00:19, 1042.24 examples/s]20597 examples [00:19, 1055.16 examples/s]20707 examples [00:19, 1065.91 examples/s]20817 examples [00:19, 1075.06 examples/s]20925 examples [00:19, 1071.74 examples/s]21033 examples [00:19, 1060.57 examples/s]21141 examples [00:19, 1065.02 examples/s]21250 examples [00:20, 1069.70 examples/s]21359 examples [00:20, 1074.85 examples/s]21467 examples [00:20, 1069.45 examples/s]21576 examples [00:20, 1073.11 examples/s]21687 examples [00:20, 1081.78 examples/s]21796 examples [00:20, 1066.44 examples/s]21905 examples [00:20, 1072.43 examples/s]22013 examples [00:20, 1073.21 examples/s]22121 examples [00:20, 1053.11 examples/s]22231 examples [00:20, 1065.70 examples/s]22342 examples [00:21, 1077.32 examples/s]22452 examples [00:21, 1082.44 examples/s]22561 examples [00:21, 1078.10 examples/s]22669 examples [00:21, 1061.38 examples/s]22776 examples [00:21, 1044.74 examples/s]22884 examples [00:21, 1054.37 examples/s]22994 examples [00:21, 1065.60 examples/s]23101 examples [00:21, 1062.67 examples/s]23208 examples [00:21, 1024.14 examples/s]23321 examples [00:21, 1053.54 examples/s]23434 examples [00:22, 1074.42 examples/s]23544 examples [00:22, 1080.29 examples/s]23654 examples [00:22, 1086.08 examples/s]23764 examples [00:22, 1088.69 examples/s]23874 examples [00:22, 1091.83 examples/s]23986 examples [00:22, 1098.27 examples/s]24098 examples [00:22, 1103.08 examples/s]24209 examples [00:22, 1103.43 examples/s]24320 examples [00:22, 1067.84 examples/s]24432 examples [00:22, 1081.62 examples/s]24544 examples [00:23, 1090.01 examples/s]24654 examples [00:23, 1084.00 examples/s]24763 examples [00:23, 1058.54 examples/s]24871 examples [00:23, 1063.54 examples/s]24980 examples [00:23, 1071.34 examples/s]25090 examples [00:23, 1077.19 examples/s]25198 examples [00:23, 1077.95 examples/s]25306 examples [00:23, 1075.41 examples/s]25414 examples [00:23, 1058.18 examples/s]25522 examples [00:23, 1064.34 examples/s]25629 examples [00:24, 1050.42 examples/s]25735 examples [00:24, 1047.78 examples/s]25843 examples [00:24, 1057.02 examples/s]25949 examples [00:24, 1024.09 examples/s]26052 examples [00:24, 1010.95 examples/s]26157 examples [00:24, 1020.84 examples/s]26266 examples [00:24, 1039.68 examples/s]26371 examples [00:24, 1038.68 examples/s]26476 examples [00:24, 1023.89 examples/s]26584 examples [00:25, 1039.73 examples/s]26694 examples [00:25, 1054.26 examples/s]26800 examples [00:25, 1047.35 examples/s]26905 examples [00:25, 1046.09 examples/s]27014 examples [00:25, 1058.63 examples/s]27123 examples [00:25, 1066.34 examples/s]27232 examples [00:25, 1071.88 examples/s]27341 examples [00:25, 1076.79 examples/s]27449 examples [00:25, 1071.72 examples/s]27557 examples [00:25, 1030.00 examples/s]27661 examples [00:26, 1026.49 examples/s]27769 examples [00:26, 1039.52 examples/s]27874 examples [00:26, 1022.49 examples/s]27980 examples [00:26, 1032.80 examples/s]28085 examples [00:26, 1037.83 examples/s]28189 examples [00:26, 1015.33 examples/s]28297 examples [00:26, 1030.70 examples/s]28401 examples [00:26, 1032.23 examples/s]28509 examples [00:26, 1044.05 examples/s]28617 examples [00:26, 1053.15 examples/s]28723 examples [00:27, 1054.46 examples/s]28833 examples [00:27, 1066.20 examples/s]28941 examples [00:27, 1070.12 examples/s]29054 examples [00:27, 1083.78 examples/s]29163 examples [00:27, 1050.64 examples/s]29269 examples [00:27, 1042.98 examples/s]29377 examples [00:27, 1053.62 examples/s]29487 examples [00:27, 1066.90 examples/s]29594 examples [00:27, 1050.74 examples/s]29700 examples [00:27, 1044.25 examples/s]29805 examples [00:28, 1032.87 examples/s]29913 examples [00:28, 1043.66 examples/s]30018 examples [00:28, 990.19 examples/s] 30125 examples [00:28, 1012.57 examples/s]30234 examples [00:28, 1033.86 examples/s]30342 examples [00:28, 1045.21 examples/s]30452 examples [00:28, 1058.45 examples/s]30559 examples [00:28, 1059.24 examples/s]30666 examples [00:28, 1058.14 examples/s]30775 examples [00:29, 1064.78 examples/s]30884 examples [00:29, 1070.07 examples/s]30992 examples [00:29, 1060.86 examples/s]31099 examples [00:29, 1052.04 examples/s]31209 examples [00:29, 1064.99 examples/s]31319 examples [00:29, 1072.92 examples/s]31427 examples [00:29, 1068.73 examples/s]31534 examples [00:29, 1063.25 examples/s]31641 examples [00:29, 1044.71 examples/s]31747 examples [00:29, 1047.34 examples/s]31852 examples [00:30, 1047.95 examples/s]31962 examples [00:30, 1058.72 examples/s]32071 examples [00:30, 1066.23 examples/s]32181 examples [00:30, 1074.84 examples/s]32290 examples [00:30, 1077.80 examples/s]32400 examples [00:30, 1081.82 examples/s]32509 examples [00:30, 1084.05 examples/s]32620 examples [00:30, 1089.20 examples/s]32731 examples [00:30, 1093.04 examples/s]32841 examples [00:30, 1092.99 examples/s]32952 examples [00:31, 1097.94 examples/s]33062 examples [00:31, 1093.69 examples/s]33172 examples [00:31, 1090.90 examples/s]33282 examples [00:31, 1082.38 examples/s]33391 examples [00:31, 1053.50 examples/s]33500 examples [00:31, 1063.97 examples/s]33609 examples [00:31, 1070.47 examples/s]33717 examples [00:31, 1054.96 examples/s]33826 examples [00:31, 1064.51 examples/s]33933 examples [00:31, 1043.11 examples/s]34042 examples [00:32, 1054.65 examples/s]34148 examples [00:32, 1054.86 examples/s]34254 examples [00:32, 1041.74 examples/s]34362 examples [00:32, 1050.50 examples/s]34468 examples [00:32, 1050.96 examples/s]34574 examples [00:32, 1053.06 examples/s]34683 examples [00:32, 1062.55 examples/s]34792 examples [00:32, 1067.98 examples/s]34901 examples [00:32, 1072.10 examples/s]35009 examples [00:33, 1009.40 examples/s]35120 examples [00:33, 1035.41 examples/s]35229 examples [00:33, 1050.69 examples/s]35339 examples [00:33, 1062.47 examples/s]35446 examples [00:33, 1059.31 examples/s]35553 examples [00:33, 1061.63 examples/s]35663 examples [00:33, 1071.62 examples/s]35772 examples [00:33, 1065.34 examples/s]35879 examples [00:33, 1039.54 examples/s]35989 examples [00:33, 1055.38 examples/s]36095 examples [00:34, 1054.49 examples/s]36203 examples [00:34, 1059.89 examples/s]36310 examples [00:34, 1060.36 examples/s]36417 examples [00:34, 1046.61 examples/s]36522 examples [00:34, 1001.16 examples/s]36629 examples [00:34, 1020.58 examples/s]36738 examples [00:34, 1038.26 examples/s]36843 examples [00:34, 1024.05 examples/s]36946 examples [00:34, 997.94 examples/s] 37050 examples [00:34, 1001.28 examples/s]37151 examples [00:35, 998.08 examples/s] 37262 examples [00:35, 1026.86 examples/s]37372 examples [00:35, 1045.72 examples/s]37481 examples [00:35, 1057.73 examples/s]37589 examples [00:35, 1063.53 examples/s]37696 examples [00:35, 1044.72 examples/s]37807 examples [00:35, 1062.40 examples/s]37914 examples [00:35, 1054.09 examples/s]38023 examples [00:35, 1062.42 examples/s]38130 examples [00:35, 1055.50 examples/s]38241 examples [00:36, 1070.18 examples/s]38351 examples [00:36, 1078.44 examples/s]38460 examples [00:36, 1079.58 examples/s]38569 examples [00:36, 1072.30 examples/s]38677 examples [00:36, 1048.46 examples/s]38787 examples [00:36, 1060.82 examples/s]38897 examples [00:36, 1071.35 examples/s]39005 examples [00:36, 1056.89 examples/s]39114 examples [00:36, 1064.97 examples/s]39222 examples [00:36, 1068.15 examples/s]39329 examples [00:37, 1065.53 examples/s]39437 examples [00:37, 1068.52 examples/s]39545 examples [00:37, 1069.25 examples/s]39652 examples [00:37, 994.62 examples/s] 39753 examples [00:37, 984.18 examples/s]39861 examples [00:37, 1010.67 examples/s]39969 examples [00:37, 1028.79 examples/s]40073 examples [00:37, 986.71 examples/s] 40183 examples [00:37, 1016.24 examples/s]40287 examples [00:38, 1021.29 examples/s]40396 examples [00:38, 1039.96 examples/s]40506 examples [00:38, 1055.89 examples/s]40612 examples [00:38, 1054.30 examples/s]40718 examples [00:38, 1036.92 examples/s]40828 examples [00:38, 1052.55 examples/s]40937 examples [00:38, 1062.17 examples/s]41047 examples [00:38, 1071.79 examples/s]41155 examples [00:38, 1038.48 examples/s]41263 examples [00:38, 1049.39 examples/s]41370 examples [00:39, 1053.13 examples/s]41476 examples [00:39, 996.15 examples/s] 41578 examples [00:39, 1002.36 examples/s]41681 examples [00:39, 1007.97 examples/s]41788 examples [00:39, 1025.08 examples/s]41898 examples [00:39, 1044.56 examples/s]42006 examples [00:39, 1052.28 examples/s]42112 examples [00:39, 1048.29 examples/s]42221 examples [00:39, 1058.15 examples/s]42327 examples [00:39, 1052.79 examples/s]42433 examples [00:40, 1052.73 examples/s]42539 examples [00:40, 1047.67 examples/s]42650 examples [00:40, 1063.27 examples/s]42761 examples [00:40, 1075.55 examples/s]42869 examples [00:40, 1070.68 examples/s]42977 examples [00:40, 1038.66 examples/s]43086 examples [00:40, 1051.68 examples/s]43195 examples [00:40, 1061.80 examples/s]43307 examples [00:40, 1078.03 examples/s]43419 examples [00:41, 1090.19 examples/s]43529 examples [00:41, 1077.63 examples/s]43637 examples [00:41, 1051.74 examples/s]43743 examples [00:41, 1008.60 examples/s]43850 examples [00:41, 1026.11 examples/s]43960 examples [00:41, 1045.23 examples/s]44066 examples [00:41, 1049.40 examples/s]44174 examples [00:41, 1055.65 examples/s]44282 examples [00:41, 1060.34 examples/s]44391 examples [00:41, 1067.46 examples/s]44501 examples [00:42, 1074.20 examples/s]44609 examples [00:42, 1067.90 examples/s]44716 examples [00:42, 1044.38 examples/s]44824 examples [00:42, 1053.58 examples/s]44930 examples [00:42, 1050.66 examples/s]45038 examples [00:42, 1057.05 examples/s]45147 examples [00:42, 1064.30 examples/s]45255 examples [00:42, 1067.35 examples/s]45364 examples [00:42, 1073.70 examples/s]45472 examples [00:42, 1062.82 examples/s]45579 examples [00:43, 1055.75 examples/s]45685 examples [00:43, 1027.74 examples/s]45788 examples [00:43, 1010.17 examples/s]45898 examples [00:43, 1034.99 examples/s]46006 examples [00:43, 1047.67 examples/s]46112 examples [00:43, 1037.43 examples/s]46220 examples [00:43, 1049.82 examples/s]46332 examples [00:43, 1068.24 examples/s]46440 examples [00:43, 1032.40 examples/s]46545 examples [00:44, 1035.07 examples/s]46649 examples [00:44, 1013.88 examples/s]46756 examples [00:44, 1028.52 examples/s]46864 examples [00:44, 1042.37 examples/s]46969 examples [00:44, 1040.05 examples/s]47075 examples [00:44, 1045.85 examples/s]47180 examples [00:44, 1024.17 examples/s]47287 examples [00:44, 1036.02 examples/s]47392 examples [00:44, 1039.50 examples/s]47500 examples [00:44, 1050.76 examples/s]47608 examples [00:45, 1057.92 examples/s]47716 examples [00:45, 1062.18 examples/s]47823 examples [00:45, 1060.45 examples/s]47930 examples [00:45, 1047.94 examples/s]48035 examples [00:45, 1040.97 examples/s]48141 examples [00:45, 1045.08 examples/s]48246 examples [00:45, 1036.74 examples/s]48350 examples [00:45, 1030.94 examples/s]48457 examples [00:45, 1041.63 examples/s]48562 examples [00:45, 1037.33 examples/s]48668 examples [00:46, 1041.80 examples/s]48774 examples [00:46, 1047.14 examples/s]48879 examples [00:46, 1042.98 examples/s]48986 examples [00:46, 1048.44 examples/s]49091 examples [00:46, 1039.64 examples/s]49198 examples [00:46, 1048.50 examples/s]49303 examples [00:46, 1021.43 examples/s]49410 examples [00:46, 1035.41 examples/s]49518 examples [00:46, 1047.87 examples/s]49626 examples [00:46, 1056.67 examples/s]49734 examples [00:47, 1062.82 examples/s]49841 examples [00:47, 1064.25 examples/s]49948 examples [00:47, 1045.45 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6403/50000 [00:00<00:00, 64028.08 examples/s] 39%|███▊      | 19292/50000 [00:00<00:00, 75412.88 examples/s] 65%|██████▍   | 32335/50000 [00:00<00:00, 86337.09 examples/s] 91%|█████████ | 45322/50000 [00:00<00:00, 95988.95 examples/s]                                                               0 examples [00:00, ? examples/s]91 examples [00:00, 903.38 examples/s]197 examples [00:00, 943.48 examples/s]307 examples [00:00, 984.36 examples/s]417 examples [00:00, 1014.25 examples/s]520 examples [00:00, 1016.09 examples/s]625 examples [00:00, 1024.35 examples/s]723 examples [00:00, 1009.17 examples/s]831 examples [00:00, 1027.66 examples/s]937 examples [00:00, 1036.02 examples/s]1048 examples [00:01, 1056.54 examples/s]1154 examples [00:01, 1054.79 examples/s]1263 examples [00:01, 1063.72 examples/s]1372 examples [00:01, 1069.16 examples/s]1481 examples [00:01, 1074.92 examples/s]1590 examples [00:01, 1077.49 examples/s]1698 examples [00:01, 1038.41 examples/s]1808 examples [00:01, 1054.83 examples/s]1916 examples [00:01, 1061.61 examples/s]2023 examples [00:01, 1054.92 examples/s]2129 examples [00:02, 1026.17 examples/s]2238 examples [00:02, 1043.28 examples/s]2348 examples [00:02, 1057.17 examples/s]2456 examples [00:02, 1061.95 examples/s]2563 examples [00:02, 1064.13 examples/s]2670 examples [00:02, 1046.92 examples/s]2775 examples [00:02, 1045.14 examples/s]2885 examples [00:02, 1060.51 examples/s]2994 examples [00:02, 1068.35 examples/s]3103 examples [00:02, 1073.70 examples/s]3215 examples [00:03, 1085.40 examples/s]3327 examples [00:03, 1093.18 examples/s]3438 examples [00:03, 1096.47 examples/s]3549 examples [00:03, 1099.71 examples/s]3660 examples [00:03, 1096.20 examples/s]3771 examples [00:03, 1099.20 examples/s]3882 examples [00:03, 1099.78 examples/s]3992 examples [00:03, 1077.95 examples/s]4102 examples [00:03, 1082.65 examples/s]4212 examples [00:03, 1086.43 examples/s]4321 examples [00:04, 1078.99 examples/s]4430 examples [00:04, 1079.44 examples/s]4543 examples [00:04, 1092.37 examples/s]4653 examples [00:04, 1094.55 examples/s]4764 examples [00:04, 1097.12 examples/s]4874 examples [00:04, 1092.19 examples/s]4985 examples [00:04, 1095.81 examples/s]5095 examples [00:04, 1088.25 examples/s]5206 examples [00:04, 1094.29 examples/s]5317 examples [00:04, 1097.14 examples/s]5427 examples [00:05, 1064.92 examples/s]5534 examples [00:05, 1055.31 examples/s]5645 examples [00:05, 1070.44 examples/s]5755 examples [00:05, 1078.43 examples/s]5866 examples [00:05, 1085.06 examples/s]5978 examples [00:05, 1095.26 examples/s]6090 examples [00:05, 1099.89 examples/s]6201 examples [00:05, 1094.62 examples/s]6313 examples [00:05, 1099.52 examples/s]6423 examples [00:05, 1089.96 examples/s]6533 examples [00:06, 1058.24 examples/s]6644 examples [00:06, 1072.38 examples/s]6755 examples [00:06, 1082.72 examples/s]6866 examples [00:06, 1090.07 examples/s]6976 examples [00:06, 1073.35 examples/s]7084 examples [00:06, 1069.40 examples/s]7192 examples [00:06, 1064.77 examples/s]7302 examples [00:06, 1074.31 examples/s]7412 examples [00:06, 1081.04 examples/s]7522 examples [00:07, 1085.99 examples/s]7631 examples [00:07, 1084.22 examples/s]7741 examples [00:07, 1087.80 examples/s]7853 examples [00:07, 1094.93 examples/s]7963 examples [00:07, 1087.95 examples/s]8072 examples [00:07, 1086.20 examples/s]8182 examples [00:07, 1088.01 examples/s]8294 examples [00:07, 1094.91 examples/s]8405 examples [00:07, 1098.99 examples/s]8516 examples [00:07, 1100.14 examples/s]8627 examples [00:08, 1078.59 examples/s]8735 examples [00:08, 1050.80 examples/s]8846 examples [00:08, 1065.53 examples/s]8958 examples [00:08, 1079.62 examples/s]9068 examples [00:08, 1084.14 examples/s]9177 examples [00:08, 1082.04 examples/s]9286 examples [00:08, 1064.81 examples/s]9396 examples [00:08, 1074.63 examples/s]9507 examples [00:08, 1082.15 examples/s]9616 examples [00:08, 1081.17 examples/s]9725 examples [00:09, 1078.96 examples/s]9834 examples [00:09, 1081.74 examples/s]9943 examples [00:09, 1080.33 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3PUDLV/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3PUDLV/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1a1357268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1a1357268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1a1357268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe18a9835f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe18230f470>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/vision/MNIST/",
    "dataset": "MNIST",
    "data_type": "tch_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tch_dataset_start",
      "uri": "mlmodels.preprocess.generic:get_dataset_torch",
      "args": {
        "dataloader": "torchvision.datasets:MNIST",
        "to_image": true,
        "transform": {
          "uri": "mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars": false,
          "arg": {}
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe1a1357268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe1a1357268> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe1a1357268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe1823bf320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe178d60978>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info": {
    "data_path": "dataset/recommender/",
    "dataset": "IMDB_sample.txt",
    "data_type": "csv_dataset",
    "batch_size": 64,
    "train": true
  },
  "preprocessors": [
    {
      "uri": "mlmodels.model_tch.textcnn:split_train_valid",
      "args": {
        "frac": 0.99
      }
    },
    {
      "uri": "mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args": {
        "lang": "en",
        "pretrained_emb": "glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe17b896a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe17b896a60> 

  function with postional parmater data_info <function split_train_valid at 0x7fe17b896a60> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe17b896b70> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe17b896b70> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe17b896b70> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=cbc97ce5c441e164ad1a551d9f3dfe99ce65a26fd742ac766ae5758adb021e47
  Stored in directory: /tmp/pip-ephem-wheel-cache-wkgu37kx/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:21:50, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:11:12, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:41:03, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:29:14, 32.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.49M/862M [00:01<5:13:44, 45.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<3:38:11, 65.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:32:21, 93.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:46:02, 133kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<1:13:48, 189kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<51:23, 270kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:01<35:59, 384kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:02<25:05, 547kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<17:38, 775kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<12:21, 1.10MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<08:43, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.0M/862M [00:03<08:00, 1.68MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:03<06:03, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<06:43, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<07:48, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<06:08, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:06<04:27, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<09:47, 1.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:07<08:14, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<06:03, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:09<07:19, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<07:51, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<06:26, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<05:52, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<04:26, 2.96MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<06:11, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<06:59, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<05:33, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:15<05:59, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<05:31, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<04:11, 3.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<05:58, 2.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:17<05:20, 2.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:17<03:55, 3.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:19<08:24, 1.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<07:13, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<05:23, 2.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:21<06:44, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:21<07:28, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:21<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<04:15, 3.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:23<1:25:04, 150kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<1:00:52, 210kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:23<42:51, 297kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:25<32:51, 387kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<24:19, 522kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<17:16, 734kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<15:02, 841kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<13:06, 964kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<09:48, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<06:58, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<1:35:53, 131kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<1:08:11, 184kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<47:58, 262kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<36:22, 344kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<27:59, 447kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<20:06, 621kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<14:12, 876kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<15:56, 780kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<12:25, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<08:59, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<09:10, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:42, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<06:52, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:25, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:43, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<04:11, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:57, 1.54MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<06:51, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:05, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:24, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<06:58, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:29, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:47, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:16, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<03:59, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:37, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<03:51, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:30, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:04, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<03:51, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:16, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:54, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<03:36, 3.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:20, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<04:44, 2.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:03, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:32, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<05:10, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:30, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:04, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<03:50, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:24, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:49, 2.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:28, 3.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<19:30, 592kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<14:50, 778kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<10:39, 1.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<10:06, 1.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:14, 1.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:02, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:54, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<07:09, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:30, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:03, 2.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:32, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:46, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:17, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:37, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:13, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:50, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<03:31, 3.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:57, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:42, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:55, 2.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:04, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:37, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:11, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:24, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:55, 2.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:43, 2.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:10, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:51, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<03:19, 3.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<09:40, 1.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:54, 1.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:48, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:36, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:56, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:20, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<03:50, 2.81MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<12:57, 833kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<10:11, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:23, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:39, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:26, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:43, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:49, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:14, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:54, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:08, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:40, 2.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:32, 2.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:55, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:32, 2.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:24, 3.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:30, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:18, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:10, 3.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:18, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:28, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:05, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:16, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:46, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:30, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:16, 3.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:24, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<07:37, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:35, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:17, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:23, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:59, 2.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:06, 1.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:37, 2.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:29, 2.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:46, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:23, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:13, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:03, 3.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:49, 778kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<09:58, 999kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:11, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:20, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:32, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:29, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:50, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:32, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:15, 3.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<38:14, 255kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<27:45, 352kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<19:35, 497kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<15:56, 608kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<13:07, 739kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<09:40, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:17, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:38, 1.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:52, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:40, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:54, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:33, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:16, 2.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<17:51, 532kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<13:28, 704kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<09:39, 980kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:56, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<07:12, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<05:14, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:53, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:01, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:37, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:21, 2.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:38, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:35, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:06, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:01, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:23, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:12, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:02, 3.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<8:49:06, 17.3kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<6:11:03, 24.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<4:19:14, 35.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<3:02:52, 49.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<2:09:53, 69.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<1:31:11, 99.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<1:03:39, 142kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<50:50, 177kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<36:21, 248kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<25:36, 351kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<19:56, 448kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<15:46, 567kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<11:24, 783kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<08:05, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:59, 987kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:11, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:12, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:42, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:47, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:29, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:13, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<8:28:24, 17.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<5:56:32, 24.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<4:09:02, 35.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<2:55:37, 49.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<2:04:47, 69.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<1:27:40, 98.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<1:01:06, 141kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<1:44:05, 82.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<1:13:42, 117kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<51:37, 166kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<38:00, 224kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<27:27, 310kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<19:22, 439kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<15:31, 545kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<12:35, 672kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<09:13, 915kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<06:31, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<8:07:17, 17.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<5:41:42, 24.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<3:58:40, 35.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<2:48:18, 49.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<1:59:28, 69.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<1:23:51, 99.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<58:32, 141kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<45:59, 180kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<33:00, 250kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<23:12, 354kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<18:06, 452kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<14:20, 571kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:23, 786kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<07:20, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<11:35, 700kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<08:56, 907kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<06:25, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:21, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:17, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:53, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:36, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<04:01, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:59, 2.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:58, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:33, 2.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:41, 2.93MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:43, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:24, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:34, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:38, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:31, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:11, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:20, 3.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:47, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:09, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:05, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:55, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:30, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:37, 2.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:36, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:04, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:13, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<02:19, 3.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<7:11:30, 17.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<5:02:33, 24.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<3:31:13, 35.0kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<2:28:51, 49.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<1:45:39, 69.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<1:14:12, 99.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<51:39, 141kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<7:36:57, 16.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<5:20:21, 22.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<3:43:38, 32.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<2:37:26, 45.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<1:50:51, 65.2kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<1:17:29, 92.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<55:40, 129kB/s]   .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<39:39, 180kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<27:50, 256kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<21:04, 337kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<15:27, 458kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<10:57, 644kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:18, 755kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:13, 972kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:12, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:16, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:06, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:55, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:50, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:24, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:33, 2.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:22, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:58, 2.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:08, 3.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<11:19, 596kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<08:36, 783kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:10, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<05:52, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:28, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:10, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:58, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:27, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:33, 2.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:19, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:39, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:50, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:03, 3.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<07:19, 885kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:47, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:12, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:25, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:44, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:45, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:25, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:02, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:17, 2.76MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:04, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:42, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:53, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:39, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:01, 3.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:49, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:36, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<01:58, 3.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:48, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:35, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:57, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:33, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<01:56, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:07, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:26, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<01:46, 3.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:02, 968kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:50, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<03:31, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:48, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:53, 1.49MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:00, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<02:09, 2.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<38:09, 150kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<27:16, 209kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<19:09, 297kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<14:45, 383kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<10:26, 539kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<07:19, 762kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:09:55, 79.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<49:27, 113kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<34:35, 160kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<25:22, 217kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<18:17, 301kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<12:53, 425kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<10:15, 531kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<08:16, 658kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:03, 897kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:04, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:06, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:59, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:19, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:27, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:38, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:54, 2.76MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:36, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:45, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:44, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:07, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:14, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:31, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:34, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:20, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:45, 2.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:23, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:38, 3.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:12, 4.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<43:06, 115kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<30:38, 162kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<21:27, 230kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<16:04, 305kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<12:15, 399kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<08:48, 554kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<06:52, 702kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:51, 823kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<04:18, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:04, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:50, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:09, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:18, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:42, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:50, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:13, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:17, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:04, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:34, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:09, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:28, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:05, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:53, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:01, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:47, 2.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:20, 3.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<00:59, 4.38MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<09:10, 474kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<07:18, 595kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:17, 819kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<03:43, 1.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:04, 842kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:59, 1.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:53, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:59, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:30, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:50, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:15, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:00, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:28, 2.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:00, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:21, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:53, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:08, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:40, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:12, 3.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<07:17, 540kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<05:30, 715kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:55, 995kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:37, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:55, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:07, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:22, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:25, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:20, 2.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:56, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:25, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:46, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:05, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:49, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:20, 2.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:47, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:59, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:34, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:39, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:31, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:08, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:40, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:14, 2.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:38, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:29, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:07, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:47, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:25, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:01, 3.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<23:54, 136kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<17:02, 190kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<11:55, 270kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<08:59, 354kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<06:56, 458kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<05:00, 633kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<03:29, 894kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<24:41, 126kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<17:34, 177kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<12:16, 251kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<09:12, 331kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<07:03, 431kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<05:03, 600kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:33, 844kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<03:31, 844kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:46, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:59, 1.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:03, 1.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:01, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:33, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:06, 2.58MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<21:14, 134kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<15:07, 187kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<10:33, 266kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<07:56, 349kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<06:07, 452kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<04:22, 629kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<03:03, 887kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:26, 785kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:40, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:55, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:56, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:37, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:11, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:25, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:15, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:55, 2.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:21, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:04, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<00:45, 3.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<2:21:18, 17.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<1:38:53, 24.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:08:32, 35.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<47:49, 49.4kB/s]  .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<33:24, 70.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<23:01, 100kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<17:12, 133kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<12:29, 183kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<08:48, 258kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<06:03, 367kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<2:13:00, 16.7kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<1:33:04, 23.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<1:04:26, 34.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<44:52, 48.0kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<31:47, 67.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<22:13, 96.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<15:17, 137kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<11:57, 174kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<08:33, 243kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<05:58, 344kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<04:34, 441kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<03:36, 558kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:36, 766kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:48, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<1:54:12, 17.1kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<1:19:52, 24.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<55:12, 34.7kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<38:21, 49.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<27:11, 69.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<18:59, 98.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<13:01, 140kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<10:10, 178kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<07:16, 248kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<05:04, 351kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:53, 448kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:51, 607kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<02:00, 848kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:46, 940kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:35, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:10, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:03, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:54, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:39, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:48, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:52, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:40, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:29, 3.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:23, 3.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:52, 786kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:06, 698kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:37, 899kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:10, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:50, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:24, 993kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:40, 833kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:19, 1.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:57, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:40, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:35, 840kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:42, 782kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:20, 991kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:57, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:40, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:50, 685kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<01:48, 700kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:22, 917kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:59, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:42, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:52, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:05, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:52, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:37, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:28, 2.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:43, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:57, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:46, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:23, 2.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<09:53, 107kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<07:18, 144kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<05:10, 202kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<03:34, 287kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<02:25, 407kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<03:42, 266kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<02:57, 332kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<02:08, 457kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:29, 640kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<01:01, 902kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<02:04, 443kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:47, 511kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:19, 683kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:55, 951kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:48, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:53, 952kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:40, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:20, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:08, 678kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:05, 711kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:49, 927kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:34, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:52, 815kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:52, 810kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:39, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:27, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:19, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<01:34, 406kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<01:19, 483kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:58, 648kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:39, 908kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:26, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<08:09, 70.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<05:54, 96.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<04:07, 136kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<02:46, 193kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<01:55, 260kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<01:31, 329kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<01:05, 453kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:43, 637kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:29, 897kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<02:02, 213kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<01:34, 273kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<01:07, 378kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:44, 532kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:33, 648kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:30, 700kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:23, 918kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:13, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:15, 1.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:12, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:08, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<12:24, 18.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<08:42, 25.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<05:54, 36.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<03:36, 52.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<02:07, 73.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<01:31, 101kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<01:01, 142kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:36, 202kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:18, 287kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:36, 142kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:26, 189kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:17, 265kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:08, 374kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 476kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 549kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 732kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 886/400000 [00:00<00:45, 8859.82it/s]  0%|          | 1801/400000 [00:00<00:44, 8943.14it/s]  1%|          | 2724/400000 [00:00<00:44, 9024.29it/s]  1%|          | 3601/400000 [00:00<00:44, 8944.36it/s]  1%|          | 4473/400000 [00:00<00:44, 8874.71it/s]  1%|▏         | 5374/400000 [00:00<00:44, 8914.19it/s]  2%|▏         | 6246/400000 [00:00<00:44, 8852.81it/s]  2%|▏         | 7059/400000 [00:00<00:45, 8611.82it/s]  2%|▏         | 7942/400000 [00:00<00:45, 8675.13it/s]  2%|▏         | 8816/400000 [00:01<00:45, 8691.55it/s]  2%|▏         | 9701/400000 [00:01<00:44, 8736.71it/s]  3%|▎         | 10630/400000 [00:01<00:43, 8893.05it/s]  3%|▎         | 11547/400000 [00:01<00:43, 8973.14it/s]  3%|▎         | 12472/400000 [00:01<00:42, 9051.91it/s]  3%|▎         | 13373/400000 [00:01<00:42, 9037.98it/s]  4%|▎         | 14274/400000 [00:01<00:43, 8894.57it/s]  4%|▍         | 15175/400000 [00:01<00:43, 8928.62it/s]  4%|▍         | 16087/400000 [00:01<00:42, 8982.74it/s]  4%|▍         | 16997/400000 [00:01<00:42, 9015.77it/s]  4%|▍         | 17898/400000 [00:02<00:42, 8892.84it/s]  5%|▍         | 18788/400000 [00:02<00:43, 8722.44it/s]  5%|▍         | 19693/400000 [00:02<00:43, 8816.44it/s]  5%|▌         | 20576/400000 [00:02<00:43, 8790.47it/s]  5%|▌         | 21491/400000 [00:02<00:42, 8894.81it/s]  6%|▌         | 22397/400000 [00:02<00:42, 8943.37it/s]  6%|▌         | 23321/400000 [00:02<00:41, 9029.94it/s]  6%|▌         | 24255/400000 [00:02<00:41, 9119.90it/s]  6%|▋         | 25171/400000 [00:02<00:41, 9131.01it/s]  7%|▋         | 26089/400000 [00:02<00:40, 9143.80it/s]  7%|▋         | 27004/400000 [00:03<00:41, 9090.07it/s]  7%|▋         | 27936/400000 [00:03<00:40, 9156.53it/s]  7%|▋         | 28858/400000 [00:03<00:40, 9173.85it/s]  7%|▋         | 29776/400000 [00:03<00:40, 9159.88it/s]  8%|▊         | 30693/400000 [00:03<00:40, 9125.05it/s]  8%|▊         | 31606/400000 [00:03<00:41, 8876.86it/s]  8%|▊         | 32496/400000 [00:03<00:41, 8839.65it/s]  8%|▊         | 33382/400000 [00:03<00:41, 8809.84it/s]  9%|▊         | 34264/400000 [00:03<00:41, 8777.60it/s]  9%|▉         | 35143/400000 [00:03<00:41, 8703.14it/s]  9%|▉         | 36014/400000 [00:04<00:41, 8701.43it/s]  9%|▉         | 36908/400000 [00:04<00:41, 8769.54it/s]  9%|▉         | 37789/400000 [00:04<00:41, 8780.02it/s] 10%|▉         | 38668/400000 [00:04<00:41, 8773.22it/s] 10%|▉         | 39546/400000 [00:04<00:41, 8751.27it/s] 10%|█         | 40423/400000 [00:04<00:41, 8756.41it/s] 10%|█         | 41304/400000 [00:04<00:40, 8769.84it/s] 11%|█         | 42197/400000 [00:04<00:40, 8817.27it/s] 11%|█         | 43091/400000 [00:04<00:40, 8851.06it/s] 11%|█         | 43977/400000 [00:04<00:40, 8799.55it/s] 11%|█         | 44858/400000 [00:05<00:40, 8775.07it/s] 11%|█▏        | 45746/400000 [00:05<00:40, 8803.94it/s] 12%|█▏        | 46627/400000 [00:05<00:40, 8723.47it/s] 12%|█▏        | 47512/400000 [00:05<00:40, 8759.25it/s] 12%|█▏        | 48389/400000 [00:05<00:40, 8754.30it/s] 12%|█▏        | 49281/400000 [00:05<00:39, 8800.47it/s] 13%|█▎        | 50191/400000 [00:05<00:39, 8887.49it/s] 13%|█▎        | 51090/400000 [00:05<00:39, 8916.09it/s] 13%|█▎        | 51985/400000 [00:05<00:38, 8925.45it/s] 13%|█▎        | 52878/400000 [00:05<00:39, 8806.60it/s] 13%|█▎        | 53770/400000 [00:06<00:39, 8839.05it/s] 14%|█▎        | 54662/400000 [00:06<00:38, 8862.90it/s] 14%|█▍        | 55549/400000 [00:06<00:38, 8848.93it/s] 14%|█▍        | 56435/400000 [00:06<00:38, 8830.35it/s] 14%|█▍        | 57335/400000 [00:06<00:38, 8878.21it/s] 15%|█▍        | 58223/400000 [00:06<00:38, 8821.10it/s] 15%|█▍        | 59112/400000 [00:06<00:38, 8838.82it/s] 15%|█▌        | 60008/400000 [00:06<00:38, 8872.34it/s] 15%|█▌        | 60913/400000 [00:06<00:37, 8923.40it/s] 15%|█▌        | 61810/400000 [00:06<00:37, 8934.97it/s] 16%|█▌        | 62704/400000 [00:07<00:38, 8774.74it/s] 16%|█▌        | 63583/400000 [00:07<00:38, 8720.92it/s] 16%|█▌        | 64456/400000 [00:07<00:38, 8643.01it/s] 16%|█▋        | 65343/400000 [00:07<00:38, 8709.24it/s] 17%|█▋        | 66215/400000 [00:07<00:39, 8548.05it/s] 17%|█▋        | 67071/400000 [00:07<00:39, 8532.94it/s] 17%|█▋        | 67942/400000 [00:07<00:38, 8582.67it/s] 17%|█▋        | 68833/400000 [00:07<00:38, 8678.06it/s] 17%|█▋        | 69748/400000 [00:07<00:37, 8813.20it/s] 18%|█▊        | 70684/400000 [00:07<00:36, 8969.01it/s] 18%|█▊        | 71583/400000 [00:08<00:37, 8851.65it/s] 18%|█▊        | 72498/400000 [00:08<00:36, 8936.83it/s] 18%|█▊        | 73407/400000 [00:08<00:36, 8982.13it/s] 19%|█▊        | 74307/400000 [00:08<00:37, 8781.51it/s] 19%|█▉        | 75247/400000 [00:08<00:36, 8955.77it/s] 19%|█▉        | 76173/400000 [00:08<00:35, 9044.55it/s] 19%|█▉        | 77080/400000 [00:08<00:35, 9016.50it/s] 19%|█▉        | 77985/400000 [00:08<00:35, 9025.60it/s] 20%|█▉        | 78889/400000 [00:08<00:35, 8999.04it/s] 20%|█▉        | 79790/400000 [00:08<00:35, 8939.75it/s] 20%|██        | 80685/400000 [00:09<00:35, 8913.87it/s] 20%|██        | 81577/400000 [00:09<00:36, 8799.90it/s] 21%|██        | 82458/400000 [00:09<00:36, 8652.88it/s] 21%|██        | 83374/400000 [00:09<00:35, 8797.09it/s] 21%|██        | 84285/400000 [00:09<00:35, 8887.14it/s] 21%|██▏       | 85198/400000 [00:09<00:35, 8957.13it/s] 22%|██▏       | 86097/400000 [00:09<00:35, 8963.55it/s] 22%|██▏       | 87012/400000 [00:09<00:34, 9018.51it/s] 22%|██▏       | 87915/400000 [00:09<00:35, 8868.54it/s] 22%|██▏       | 88803/400000 [00:10<00:35, 8798.51it/s] 22%|██▏       | 89715/400000 [00:10<00:34, 8890.25it/s] 23%|██▎       | 90625/400000 [00:10<00:34, 8950.10it/s] 23%|██▎       | 91521/400000 [00:10<00:34, 8835.18it/s] 23%|██▎       | 92422/400000 [00:10<00:34, 8884.93it/s] 23%|██▎       | 93333/400000 [00:10<00:34, 8950.65it/s] 24%|██▎       | 94233/400000 [00:10<00:34, 8963.56it/s] 24%|██▍       | 95130/400000 [00:10<00:34, 8861.90it/s] 24%|██▍       | 96027/400000 [00:10<00:34, 8892.62it/s] 24%|██▍       | 96924/400000 [00:10<00:33, 8915.17it/s] 24%|██▍       | 97831/400000 [00:11<00:33, 8960.66it/s] 25%|██▍       | 98737/400000 [00:11<00:33, 8988.35it/s] 25%|██▍       | 99647/400000 [00:11<00:33, 9019.43it/s] 25%|██▌       | 100550/400000 [00:11<00:33, 8890.09it/s] 25%|██▌       | 101440/400000 [00:11<00:34, 8714.45it/s] 26%|██▌       | 102320/400000 [00:11<00:34, 8739.86it/s] 26%|██▌       | 103195/400000 [00:11<00:35, 8414.03it/s] 26%|██▌       | 104067/400000 [00:11<00:34, 8501.32it/s] 26%|██▌       | 104954/400000 [00:11<00:34, 8606.19it/s] 26%|██▋       | 105833/400000 [00:11<00:33, 8659.15it/s] 27%|██▋       | 106719/400000 [00:12<00:33, 8717.15it/s] 27%|██▋       | 107605/400000 [00:12<00:33, 8758.88it/s] 27%|██▋       | 108493/400000 [00:12<00:33, 8792.18it/s] 27%|██▋       | 109387/400000 [00:12<00:32, 8834.88it/s] 28%|██▊       | 110271/400000 [00:12<00:33, 8696.87it/s] 28%|██▊       | 111142/400000 [00:12<00:33, 8549.90it/s] 28%|██▊       | 112027/400000 [00:12<00:33, 8636.29it/s] 28%|██▊       | 112892/400000 [00:12<00:33, 8570.35it/s] 28%|██▊       | 113776/400000 [00:12<00:33, 8648.22it/s] 29%|██▊       | 114670/400000 [00:12<00:32, 8733.69it/s] 29%|██▉       | 115550/400000 [00:13<00:32, 8751.32it/s] 29%|██▉       | 116427/400000 [00:13<00:32, 8756.03it/s] 29%|██▉       | 117310/400000 [00:13<00:32, 8776.46it/s] 30%|██▉       | 118205/400000 [00:13<00:31, 8825.82it/s] 30%|██▉       | 119103/400000 [00:13<00:31, 8871.20it/s] 30%|██▉       | 119991/400000 [00:13<00:31, 8854.56it/s] 30%|███       | 120877/400000 [00:13<00:31, 8751.20it/s] 30%|███       | 121767/400000 [00:13<00:31, 8794.74it/s] 31%|███       | 122664/400000 [00:13<00:31, 8845.04it/s] 31%|███       | 123549/400000 [00:13<00:31, 8756.28it/s] 31%|███       | 124442/400000 [00:14<00:31, 8806.28it/s] 31%|███▏      | 125323/400000 [00:14<00:31, 8732.72it/s] 32%|███▏      | 126197/400000 [00:14<00:31, 8702.82it/s] 32%|███▏      | 127068/400000 [00:14<00:31, 8690.35it/s] 32%|███▏      | 127952/400000 [00:14<00:31, 8734.69it/s] 32%|███▏      | 128856/400000 [00:14<00:30, 8822.69it/s] 32%|███▏      | 129739/400000 [00:14<00:31, 8602.41it/s] 33%|███▎      | 130614/400000 [00:14<00:31, 8643.21it/s] 33%|███▎      | 131501/400000 [00:14<00:30, 8709.66it/s] 33%|███▎      | 132391/400000 [00:14<00:30, 8765.58it/s] 33%|███▎      | 133276/400000 [00:15<00:30, 8788.18it/s] 34%|███▎      | 134156/400000 [00:15<00:30, 8747.69it/s] 34%|███▍      | 135045/400000 [00:15<00:30, 8789.56it/s] 34%|███▍      | 135925/400000 [00:15<00:30, 8718.40it/s] 34%|███▍      | 136817/400000 [00:15<00:29, 8777.23it/s] 34%|███▍      | 137725/400000 [00:15<00:29, 8865.14it/s] 35%|███▍      | 138612/400000 [00:15<00:29, 8852.29it/s] 35%|███▍      | 139507/400000 [00:15<00:29, 8880.08it/s] 35%|███▌      | 140396/400000 [00:15<00:29, 8860.19it/s] 35%|███▌      | 141309/400000 [00:15<00:28, 8936.93it/s] 36%|███▌      | 142203/400000 [00:16<00:28, 8909.53it/s] 36%|███▌      | 143095/400000 [00:16<00:29, 8767.32it/s] 36%|███▌      | 144013/400000 [00:16<00:28, 8887.03it/s] 36%|███▌      | 144925/400000 [00:16<00:28, 8955.03it/s] 36%|███▋      | 145822/400000 [00:16<00:28, 8938.00it/s] 37%|███▋      | 146717/400000 [00:16<00:28, 8885.19it/s] 37%|███▋      | 147606/400000 [00:16<00:28, 8738.56it/s] 37%|███▋      | 148490/400000 [00:16<00:28, 8766.76it/s] 37%|███▋      | 149391/400000 [00:16<00:28, 8837.75it/s] 38%|███▊      | 150276/400000 [00:17<00:28, 8828.63it/s] 38%|███▊      | 151195/400000 [00:17<00:27, 8933.61it/s] 38%|███▊      | 152098/400000 [00:17<00:27, 8961.59it/s] 38%|███▊      | 152995/400000 [00:17<00:27, 8892.69it/s] 38%|███▊      | 153917/400000 [00:17<00:27, 8987.03it/s] 39%|███▊      | 154817/400000 [00:17<00:27, 8984.03it/s] 39%|███▉      | 155716/400000 [00:17<00:27, 8960.30it/s] 39%|███▉      | 156613/400000 [00:17<00:27, 8721.37it/s] 39%|███▉      | 157487/400000 [00:17<00:28, 8488.59it/s] 40%|███▉      | 158413/400000 [00:17<00:27, 8705.96it/s] 40%|███▉      | 159330/400000 [00:18<00:27, 8838.01it/s] 40%|████      | 160217/400000 [00:18<00:27, 8840.82it/s] 40%|████      | 161130/400000 [00:18<00:26, 8922.32it/s] 41%|████      | 162039/400000 [00:18<00:26, 8970.19it/s] 41%|████      | 162947/400000 [00:18<00:26, 9001.94it/s] 41%|████      | 163854/400000 [00:18<00:26, 9019.53it/s] 41%|████      | 164762/400000 [00:18<00:26, 9036.88it/s] 41%|████▏     | 165667/400000 [00:18<00:25, 9014.99it/s] 42%|████▏     | 166577/400000 [00:18<00:26, 8870.85it/s] 42%|████▏     | 167472/400000 [00:18<00:26, 8893.46it/s] 42%|████▏     | 168364/400000 [00:19<00:26, 8899.23it/s] 42%|████▏     | 169255/400000 [00:19<00:25, 8881.89it/s] 43%|████▎     | 170144/400000 [00:19<00:26, 8773.07it/s] 43%|████▎     | 171022/400000 [00:19<00:26, 8679.45it/s] 43%|████▎     | 171913/400000 [00:19<00:26, 8745.38it/s] 43%|████▎     | 172801/400000 [00:19<00:25, 8783.61it/s] 43%|████▎     | 173688/400000 [00:19<00:25, 8808.81it/s] 44%|████▎     | 174570/400000 [00:19<00:25, 8684.47it/s] 44%|████▍     | 175462/400000 [00:19<00:25, 8753.54it/s] 44%|████▍     | 176356/400000 [00:19<00:25, 8808.62it/s] 44%|████▍     | 177238/400000 [00:20<00:25, 8771.51it/s] 45%|████▍     | 178132/400000 [00:20<00:25, 8819.48it/s] 45%|████▍     | 179015/400000 [00:20<00:25, 8786.53it/s] 45%|████▍     | 179910/400000 [00:20<00:24, 8831.48it/s] 45%|████▌     | 180814/400000 [00:20<00:24, 8890.34it/s] 45%|████▌     | 181704/400000 [00:20<00:24, 8755.63it/s] 46%|████▌     | 182593/400000 [00:20<00:24, 8793.51it/s] 46%|████▌     | 183473/400000 [00:20<00:24, 8752.11it/s] 46%|████▌     | 184354/400000 [00:20<00:24, 8768.21it/s] 46%|████▋     | 185232/400000 [00:20<00:24, 8760.32it/s] 47%|████▋     | 186126/400000 [00:21<00:24, 8811.26it/s] 47%|████▋     | 187028/400000 [00:21<00:24, 8871.75it/s] 47%|████▋     | 187916/400000 [00:21<00:23, 8844.60it/s] 47%|████▋     | 188801/400000 [00:21<00:23, 8828.34it/s] 47%|████▋     | 189695/400000 [00:21<00:23, 8860.47it/s] 48%|████▊     | 190582/400000 [00:21<00:23, 8780.59it/s] 48%|████▊     | 191461/400000 [00:21<00:24, 8614.06it/s] 48%|████▊     | 192324/400000 [00:21<00:24, 8517.01it/s] 48%|████▊     | 193200/400000 [00:21<00:24, 8586.20it/s] 49%|████▊     | 194099/400000 [00:21<00:23, 8702.27it/s] 49%|████▉     | 195004/400000 [00:22<00:23, 8802.45it/s] 49%|████▉     | 195897/400000 [00:22<00:23, 8839.82it/s] 49%|████▉     | 196782/400000 [00:22<00:23, 8750.25it/s] 49%|████▉     | 197658/400000 [00:22<00:24, 8400.72it/s] 50%|████▉     | 198541/400000 [00:22<00:23, 8523.38it/s] 50%|████▉     | 199397/400000 [00:22<00:23, 8431.77it/s] 50%|█████     | 200257/400000 [00:22<00:23, 8479.63it/s] 50%|█████     | 201107/400000 [00:22<00:23, 8437.52it/s] 50%|█████     | 201952/400000 [00:22<00:23, 8386.31it/s] 51%|█████     | 202826/400000 [00:23<00:23, 8488.90it/s] 51%|█████     | 203709/400000 [00:23<00:22, 8586.23it/s] 51%|█████     | 204569/400000 [00:23<00:22, 8576.43it/s] 51%|█████▏    | 205438/400000 [00:23<00:22, 8607.69it/s] 52%|█████▏    | 206320/400000 [00:23<00:22, 8667.48it/s] 52%|█████▏    | 207188/400000 [00:23<00:22, 8621.73it/s] 52%|█████▏    | 208060/400000 [00:23<00:22, 8649.35it/s] 52%|█████▏    | 208930/400000 [00:23<00:22, 8661.57it/s] 52%|█████▏    | 209797/400000 [00:23<00:22, 8413.25it/s] 53%|█████▎    | 210675/400000 [00:23<00:22, 8518.82it/s] 53%|█████▎    | 211566/400000 [00:24<00:21, 8630.14it/s] 53%|█████▎    | 212437/400000 [00:24<00:21, 8652.29it/s] 53%|█████▎    | 213304/400000 [00:24<00:21, 8640.77it/s] 54%|█████▎    | 214169/400000 [00:24<00:21, 8543.56it/s] 54%|█████▍    | 215051/400000 [00:24<00:21, 8622.31it/s] 54%|█████▍    | 215930/400000 [00:24<00:21, 8671.90it/s] 54%|█████▍    | 216798/400000 [00:24<00:21, 8642.81it/s] 54%|█████▍    | 217663/400000 [00:24<00:21, 8633.51it/s] 55%|█████▍    | 218527/400000 [00:24<00:21, 8600.17it/s] 55%|█████▍    | 219393/400000 [00:24<00:20, 8617.11it/s] 55%|█████▌    | 220259/400000 [00:25<00:20, 8629.80it/s] 55%|█████▌    | 221143/400000 [00:25<00:20, 8690.34it/s] 56%|█████▌    | 222013/400000 [00:25<00:20, 8652.10it/s] 56%|█████▌    | 222880/400000 [00:25<00:20, 8656.23it/s] 56%|█████▌    | 223766/400000 [00:25<00:20, 8713.53it/s] 56%|█████▌    | 224648/400000 [00:25<00:20, 8744.24it/s] 56%|█████▋    | 225525/400000 [00:25<00:19, 8749.31it/s] 57%|█████▋    | 226401/400000 [00:25<00:20, 8673.02it/s] 57%|█████▋    | 227269/400000 [00:25<00:20, 8463.63it/s] 57%|█████▋    | 228143/400000 [00:25<00:20, 8544.03it/s] 57%|█████▋    | 229013/400000 [00:26<00:19, 8588.91it/s] 57%|█████▋    | 229882/400000 [00:26<00:19, 8616.96it/s] 58%|█████▊    | 230745/400000 [00:26<00:19, 8609.12it/s] 58%|█████▊    | 231607/400000 [00:26<00:19, 8525.15it/s] 58%|█████▊    | 232483/400000 [00:26<00:19, 8592.69it/s] 58%|█████▊    | 233358/400000 [00:26<00:19, 8639.16it/s] 59%|█████▊    | 234245/400000 [00:26<00:19, 8706.09it/s] 59%|█████▉    | 235116/400000 [00:26<00:19, 8616.93it/s] 59%|█████▉    | 235979/400000 [00:26<00:19, 8385.98it/s] 59%|█████▉    | 236850/400000 [00:26<00:19, 8480.00it/s] 59%|█████▉    | 237714/400000 [00:27<00:19, 8525.99it/s] 60%|█████▉    | 238590/400000 [00:27<00:18, 8594.46it/s] 60%|█████▉    | 239473/400000 [00:27<00:18, 8663.65it/s] 60%|██████    | 240356/400000 [00:27<00:18, 8712.61it/s] 60%|██████    | 241232/400000 [00:27<00:18, 8726.05it/s] 61%|██████    | 242113/400000 [00:27<00:18, 8750.70it/s] 61%|██████    | 242989/400000 [00:27<00:17, 8725.97it/s] 61%|██████    | 243862/400000 [00:27<00:17, 8724.32it/s] 61%|██████    | 244735/400000 [00:27<00:18, 8604.42it/s] 61%|██████▏   | 245596/400000 [00:27<00:18, 8541.73it/s] 62%|██████▏   | 246451/400000 [00:28<00:18, 8445.42it/s] 62%|██████▏   | 247311/400000 [00:28<00:17, 8488.99it/s] 62%|██████▏   | 248188/400000 [00:28<00:17, 8571.05it/s] 62%|██████▏   | 249046/400000 [00:28<00:17, 8560.90it/s] 62%|██████▏   | 249903/400000 [00:28<00:17, 8529.63it/s] 63%|██████▎   | 250782/400000 [00:28<00:17, 8605.54it/s] 63%|██████▎   | 251643/400000 [00:28<00:17, 8596.87it/s] 63%|██████▎   | 252503/400000 [00:28<00:17, 8492.11it/s] 63%|██████▎   | 253369/400000 [00:28<00:17, 8541.76it/s] 64%|██████▎   | 254240/400000 [00:28<00:16, 8590.19it/s] 64%|██████▍   | 255100/400000 [00:29<00:17, 8522.63it/s] 64%|██████▍   | 255968/400000 [00:29<00:16, 8566.20it/s] 64%|██████▍   | 256859/400000 [00:29<00:16, 8664.24it/s] 64%|██████▍   | 257732/400000 [00:29<00:16, 8681.42it/s] 65%|██████▍   | 258605/400000 [00:29<00:16, 8693.67it/s] 65%|██████▍   | 259475/400000 [00:29<00:16, 8675.18it/s] 65%|██████▌   | 260343/400000 [00:29<00:16, 8446.96it/s] 65%|██████▌   | 261228/400000 [00:29<00:16, 8562.45it/s] 66%|██████▌   | 262086/400000 [00:29<00:16, 8487.51it/s] 66%|██████▌   | 262982/400000 [00:29<00:15, 8623.47it/s] 66%|██████▌   | 263856/400000 [00:30<00:15, 8657.23it/s] 66%|██████▌   | 264736/400000 [00:30<00:15, 8698.87it/s] 66%|██████▋   | 265642/400000 [00:30<00:15, 8801.86it/s] 67%|██████▋   | 266546/400000 [00:30<00:15, 8870.34it/s] 67%|██████▋   | 267448/400000 [00:30<00:14, 8912.30it/s] 67%|██████▋   | 268343/400000 [00:30<00:14, 8922.30it/s] 67%|██████▋   | 269236/400000 [00:30<00:14, 8822.35it/s] 68%|██████▊   | 270119/400000 [00:30<00:14, 8703.85it/s] 68%|██████▊   | 270991/400000 [00:30<00:14, 8704.26it/s] 68%|██████▊   | 271887/400000 [00:30<00:14, 8778.87it/s] 68%|██████▊   | 272777/400000 [00:31<00:14, 8814.72it/s] 68%|██████▊   | 273684/400000 [00:31<00:14, 8889.29it/s] 69%|██████▊   | 274580/400000 [00:31<00:14, 8910.30it/s] 69%|██████▉   | 275472/400000 [00:31<00:14, 8888.51it/s] 69%|██████▉   | 276362/400000 [00:31<00:13, 8874.74it/s] 69%|██████▉   | 277258/400000 [00:31<00:13, 8899.68it/s] 70%|██████▉   | 278149/400000 [00:31<00:13, 8891.53it/s] 70%|██████▉   | 279052/400000 [00:31<00:13, 8929.62it/s] 70%|██████▉   | 279946/400000 [00:31<00:13, 8872.76it/s] 70%|███████   | 280839/400000 [00:32<00:13, 8889.67it/s] 70%|███████   | 281739/400000 [00:32<00:13, 8919.91it/s] 71%|███████   | 282632/400000 [00:32<00:13, 8916.26it/s] 71%|███████   | 283526/400000 [00:32<00:13, 8920.77it/s] 71%|███████   | 284424/400000 [00:32<00:12, 8936.88it/s] 71%|███████▏  | 285334/400000 [00:32<00:12, 8982.68it/s] 72%|███████▏  | 286241/400000 [00:32<00:12, 9006.47it/s] 72%|███████▏  | 287142/400000 [00:32<00:12, 8923.76it/s] 72%|███████▏  | 288038/400000 [00:32<00:12, 8934.36it/s] 72%|███████▏  | 288943/400000 [00:32<00:12, 8966.72it/s] 72%|███████▏  | 289840/400000 [00:33<00:12, 8956.06it/s] 73%|███████▎  | 290736/400000 [00:33<00:12, 8899.18it/s] 73%|███████▎  | 291627/400000 [00:33<00:12, 8862.18it/s] 73%|███████▎  | 292543/400000 [00:33<00:12, 8947.78it/s] 73%|███████▎  | 293447/400000 [00:33<00:11, 8973.36it/s] 74%|███████▎  | 294345/400000 [00:33<00:11, 8915.92it/s] 74%|███████▍  | 295237/400000 [00:33<00:11, 8911.26it/s] 74%|███████▍  | 296129/400000 [00:33<00:11, 8871.59it/s] 74%|███████▍  | 297039/400000 [00:33<00:11, 8936.48it/s] 74%|███████▍  | 297933/400000 [00:33<00:11, 8852.91it/s] 75%|███████▍  | 298830/400000 [00:34<00:11, 8887.41it/s] 75%|███████▍  | 299720/400000 [00:34<00:11, 8830.34it/s] 75%|███████▌  | 300609/400000 [00:34<00:11, 8848.13it/s] 75%|███████▌  | 301522/400000 [00:34<00:11, 8929.88it/s] 76%|███████▌  | 302432/400000 [00:34<00:10, 8977.70it/s] 76%|███████▌  | 303331/400000 [00:34<00:11, 8768.43it/s] 76%|███████▌  | 304210/400000 [00:34<00:10, 8765.48it/s] 76%|███████▋  | 305088/400000 [00:34<00:10, 8757.95it/s] 76%|███████▋  | 305981/400000 [00:34<00:10, 8806.02it/s] 77%|███████▋  | 306863/400000 [00:34<00:10, 8641.73it/s] 77%|███████▋  | 307751/400000 [00:35<00:10, 8709.25it/s] 77%|███████▋  | 308623/400000 [00:35<00:10, 8502.37it/s] 77%|███████▋  | 309518/400000 [00:35<00:10, 8631.72it/s] 78%|███████▊  | 310423/400000 [00:35<00:10, 8750.95it/s] 78%|███████▊  | 311325/400000 [00:35<00:10, 8829.42it/s] 78%|███████▊  | 312233/400000 [00:35<00:09, 8902.97it/s] 78%|███████▊  | 313125/400000 [00:35<00:09, 8811.14it/s] 79%|███████▊  | 314016/400000 [00:35<00:09, 8838.77it/s] 79%|███████▊  | 314910/400000 [00:35<00:09, 8868.32it/s] 79%|███████▉  | 315803/400000 [00:35<00:09, 8883.74it/s] 79%|███████▉  | 316692/400000 [00:36<00:09, 8843.75it/s] 79%|███████▉  | 317589/400000 [00:36<00:09, 8878.42it/s] 80%|███████▉  | 318478/400000 [00:36<00:09, 8880.86it/s] 80%|███████▉  | 319389/400000 [00:36<00:09, 8946.38it/s] 80%|████████  | 320284/400000 [00:36<00:08, 8923.99it/s] 80%|████████  | 321186/400000 [00:36<00:08, 8951.94it/s] 81%|████████  | 322082/400000 [00:36<00:08, 8857.57it/s] 81%|████████  | 322986/400000 [00:36<00:08, 8908.88it/s] 81%|████████  | 323878/400000 [00:36<00:08, 8856.73it/s] 81%|████████  | 324764/400000 [00:36<00:08, 8593.79it/s] 81%|████████▏ | 325626/400000 [00:37<00:08, 8528.59it/s] 82%|████████▏ | 326481/400000 [00:37<00:08, 8303.91it/s] 82%|████████▏ | 327346/400000 [00:37<00:08, 8404.07it/s] 82%|████████▏ | 328227/400000 [00:37<00:08, 8519.68it/s] 82%|████████▏ | 329081/400000 [00:37<00:08, 8507.54it/s] 82%|████████▏ | 329961/400000 [00:37<00:08, 8591.11it/s] 83%|████████▎ | 330847/400000 [00:37<00:07, 8667.74it/s] 83%|████████▎ | 331743/400000 [00:37<00:07, 8750.57it/s] 83%|████████▎ | 332652/400000 [00:37<00:07, 8848.70it/s] 83%|████████▎ | 333538/400000 [00:37<00:07, 8782.95it/s] 84%|████████▎ | 334417/400000 [00:38<00:07, 8576.09it/s] 84%|████████▍ | 335286/400000 [00:38<00:07, 8607.41it/s] 84%|████████▍ | 336188/400000 [00:38<00:07, 8726.94it/s] 84%|████████▍ | 337098/400000 [00:38<00:07, 8833.89it/s] 84%|████████▍ | 337983/400000 [00:38<00:07, 8814.06it/s] 85%|████████▍ | 338866/400000 [00:38<00:07, 8507.90it/s] 85%|████████▍ | 339734/400000 [00:38<00:07, 8556.79it/s] 85%|████████▌ | 340600/400000 [00:38<00:06, 8586.90it/s] 85%|████████▌ | 341491/400000 [00:38<00:06, 8680.27it/s] 86%|████████▌ | 342376/400000 [00:39<00:06, 8728.08it/s] 86%|████████▌ | 343266/400000 [00:39<00:06, 8776.86it/s] 86%|████████▌ | 344145/400000 [00:39<00:06, 8720.84it/s] 86%|████████▋ | 345024/400000 [00:39<00:06, 8739.71it/s] 86%|████████▋ | 345901/400000 [00:39<00:06, 8747.03it/s] 87%|████████▋ | 346777/400000 [00:39<00:06, 8610.70it/s] 87%|████████▋ | 347681/400000 [00:39<00:05, 8734.26it/s] 87%|████████▋ | 348556/400000 [00:39<00:06, 8477.59it/s] 87%|████████▋ | 349427/400000 [00:39<00:05, 8545.23it/s] 88%|████████▊ | 350302/400000 [00:39<00:05, 8605.60it/s] 88%|████████▊ | 351168/400000 [00:40<00:05, 8621.83it/s] 88%|████████▊ | 352060/400000 [00:40<00:05, 8707.37it/s] 88%|████████▊ | 352933/400000 [00:40<00:05, 8671.04it/s] 88%|████████▊ | 353801/400000 [00:40<00:05, 8549.27it/s] 89%|████████▊ | 354687/400000 [00:40<00:05, 8639.75it/s] 89%|████████▉ | 355559/400000 [00:40<00:05, 8660.70it/s] 89%|████████▉ | 356426/400000 [00:40<00:05, 8587.92it/s] 89%|████████▉ | 357305/400000 [00:40<00:04, 8645.46it/s] 90%|████████▉ | 358190/400000 [00:40<00:04, 8705.00it/s] 90%|████████▉ | 359099/400000 [00:40<00:04, 8814.10it/s] 90%|█████████ | 360001/400000 [00:41<00:04, 8873.40it/s] 90%|█████████ | 360889/400000 [00:41<00:04, 8774.36it/s] 90%|█████████ | 361768/400000 [00:41<00:04, 8506.12it/s] 91%|█████████ | 362658/400000 [00:41<00:04, 8618.66it/s] 91%|█████████ | 363542/400000 [00:41<00:04, 8683.33it/s] 91%|█████████ | 364432/400000 [00:41<00:04, 8744.96it/s] 91%|█████████▏| 365340/400000 [00:41<00:03, 8842.64it/s] 92%|█████████▏| 366249/400000 [00:41<00:03, 8914.00it/s] 92%|█████████▏| 367146/400000 [00:41<00:03, 8929.80it/s] 92%|█████████▏| 368059/400000 [00:41<00:03, 8986.42it/s] 92%|█████████▏| 368959/400000 [00:42<00:03, 8962.20it/s] 92%|█████████▏| 369856/400000 [00:42<00:03, 8822.73it/s] 93%|█████████▎| 370745/400000 [00:42<00:03, 8841.27it/s] 93%|█████████▎| 371638/400000 [00:42<00:03, 8867.20it/s] 93%|█████████▎| 372527/400000 [00:42<00:03, 8873.10it/s] 93%|█████████▎| 373415/400000 [00:42<00:02, 8867.32it/s] 94%|█████████▎| 374308/400000 [00:42<00:02, 8884.34it/s] 94%|█████████▍| 375197/400000 [00:42<00:02, 8867.62it/s] 94%|█████████▍| 376111/400000 [00:42<00:02, 8947.57it/s] 94%|█████████▍| 377007/400000 [00:42<00:02, 8931.61it/s] 94%|█████████▍| 377911/400000 [00:43<00:02, 8961.67it/s] 95%|█████████▍| 378808/400000 [00:43<00:02, 8892.19it/s] 95%|█████████▍| 379706/400000 [00:43<00:02, 8916.80it/s] 95%|█████████▌| 380598/400000 [00:43<00:02, 8793.13it/s] 95%|█████████▌| 381513/400000 [00:43<00:02, 8895.30it/s] 96%|█████████▌| 382408/400000 [00:43<00:01, 8910.55it/s] 96%|█████████▌| 383300/400000 [00:43<00:01, 8815.36it/s] 96%|█████████▌| 384200/400000 [00:43<00:01, 8869.39it/s] 96%|█████████▋| 385091/400000 [00:43<00:01, 8878.73it/s] 96%|█████████▋| 385980/400000 [00:43<00:01, 8845.78it/s] 97%|█████████▋| 386885/400000 [00:44<00:01, 8905.79it/s] 97%|█████████▋| 387776/400000 [00:44<00:01, 8637.74it/s] 97%|█████████▋| 388654/400000 [00:44<00:01, 8678.70it/s] 97%|█████████▋| 389548/400000 [00:44<00:01, 8753.77it/s] 98%|█████████▊| 390461/400000 [00:44<00:01, 8862.13it/s] 98%|█████████▊| 391368/400000 [00:44<00:00, 8922.37it/s] 98%|█████████▊| 392262/400000 [00:44<00:00, 8834.14it/s] 98%|█████████▊| 393147/400000 [00:44<00:00, 8825.92it/s] 99%|█████████▊| 394037/400000 [00:44<00:00, 8847.47it/s] 99%|█████████▊| 394940/400000 [00:44<00:00, 8901.02it/s] 99%|█████████▉| 395850/400000 [00:45<00:00, 8958.84it/s] 99%|█████████▉| 396747/400000 [00:45<00:00, 8564.46it/s] 99%|█████████▉| 397657/400000 [00:45<00:00, 8717.33it/s]100%|█████████▉| 398561/400000 [00:45<00:00, 8808.95it/s]100%|█████████▉| 399473/400000 [00:45<00:00, 8897.94it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8779.94it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe20d35ab00>, <torchtext.data.dataset.TabularDataset object at 0x7fe20d35a9b0>, <torchtext.vocab.Vocab object at 0x7fe20d35a278>), {}) 

