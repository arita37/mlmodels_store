
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '62bdf6db8ad57c05ac9066eac348673bac127d5b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/62bdf6db8ad57c05ac9066eac348673bac127d5b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_CIFAR.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_KMNIST.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_FashionMNIST.json']





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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff425e7e730>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff425e7e730>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7ff48efdbb70>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7ff48efdbb70>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff425e98b70>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff425e98b70>
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



###### Test_run_model  #############################################################



 ####################################################################################################
model_keras/charcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.charcnn.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          10
        ],
        [
          256,
          7
        ],
        [
          256,
          5
        ],
        [
          256,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.1,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn/charcnn.h5",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.charcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 506, in test_dataloader
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 327, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 80, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3313: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
CharCNNKim model built: 
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sent_input (InputLayer)         (None, 1014)         0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1014, 128)    8960        sent_input[0][0]                 
__________________________________________________________________________________________________
Conv1D_256_10 (Conv1D)          (None, 1005, 256)    327936      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_7 (Conv1D)           (None, 1008, 256)    229632      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_5 (Conv1D)           (None, 1010, 256)    164096      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_3 (Conv1D)           (None, 1012, 256)    98560       embedding_1[0][0]                
__________________________________________________________________________________________________
MaxPoolingOverTime_256_10 (Glob (None, 256)          0           Conv1D_256_10[0][0]              
__________________________________________________________________________________________________
MaxPoolingOverTime_256_7 (Globa (None, 256)          0           Conv1D_256_7[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_5 (Globa (None, 256)          0           Conv1D_256_5[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_3 (Globa (None, 256)          0           Conv1D_256_3[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1024)         0           MaxPoolingOverTime_256_10[0][0]  
                                                                 MaxPoolingOverTime_256_7[0][0]   
                                                                 MaxPoolingOverTime_256_5[0][0]   
                                                                 MaxPoolingOverTime_256_3[0][0]   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         1049600     concatenate_1[0][0]              
__________________________________________________________________________________________________
alpha_dropout_1 (AlphaDropout)  (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         1049600     alpha_dropout_1[0][0]            
__________________________________________________________________________________________________
alpha_dropout_2 (AlphaDropout)  (None, 1024)         0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            4100        alpha_dropout_2[0][0]            
==================================================================================================
Total params: 2,932,484
Trainable params: 2,932,484
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.charcnn.Model object at 0x7ff41cf80eb8> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data
2020-05-25 11:28:29.495145: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 11:28:29.498994: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-25 11:28:29.499131: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560e3e0927f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 11:28:29.499145: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-25 11:28:30.084279: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132120576 exceeds 10% of system memory.
2020-05-25 11:28:30.085806: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132382720 exceeds 10% of system memory.
2020-05-25 11:28:31.115679: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132120576 exceeds 10% of system memory.
2020-05-25 11:28:31.217360: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132382720 exceeds 10% of system memory.
2020-05-25 11:28:31.328108: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 131727360 exceeds 10% of system memory.

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 13s - loss: 1.6815
256/354 [====================>.........] - ETA: 5s - loss: 1.2638 
354/354 [==============================] - 26s 74ms/step - loss: 1.1883 - val_loss: 0.6620

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[3.0134495e-05 3.4378132e-05 1.5815488e-01 8.4178060e-01]
 [2.6729755e-05 2.7237929e-05 1.3207301e-01 8.6787301e-01]
 [2.9390045e-05 3.2133954e-05 1.6103463e-01 8.3890384e-01]
 ...
 [5.3935870e-05 5.6254397e-05 1.5494391e-01 8.4494591e-01]
 [4.5704939e-05 5.1216255e-05 1.6853671e-01 8.3136636e-01]
 [4.2138279e-05 4.2360287e-05 1.5095575e-01 8.4895974e-01]]

  #### Get  metrics   ################################################ 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/charcnn_zhang.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.charcnn_zhang.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          7,
          3
        ],
        [
          256,
          7,
          3
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.5,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn_zhang/",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  <module 'mlmodels.model_keras.charcnn_zhang' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

CharCNNZhang model built: 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sent_input (InputLayer)      (None, 1014)              0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 1014, 128)         8960      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1008, 256)         229632    
_________________________________________________________________
thresholded_re_lu_1 (Thresho (None, 1008, 256)         0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 336, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 330, 256)          459008    
_________________________________________________________________
thresholded_re_lu_2 (Thresho (None, 330, 256)          0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 110, 256)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 108, 256)          196864    
_________________________________________________________________
thresholded_re_lu_3 (Thresho (None, 108, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 106, 256)          196864    
_________________________________________________________________
thresholded_re_lu_4 (Thresho (None, 106, 256)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 104, 256)          196864    
_________________________________________________________________
thresholded_re_lu_5 (Thresho (None, 104, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 102, 256)          196864    
_________________________________________________________________
thresholded_re_lu_6 (Thresho (None, 102, 256)          0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 34, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8704)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              8913920   
_________________________________________________________________
thresholded_re_lu_7 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
thresholded_re_lu_8 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 11,452,676
Trainable params: 11,452,676
Non-trainable params: 0
_________________________________________________________________

  <mlmodels.model_keras.charcnn_zhang.Model object at 0x7ff425633208> 

  #### Fit   ######################################################## 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 8s - loss: 1.3847
256/354 [====================>.........] - ETA: 3s - loss: 1.2692
354/354 [==============================] - 16s 46ms/step - loss: 1.2930 - val_loss: 0.7229

  #### Predict   #################################################### 

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels.model_keras.raw.char_cnn.data_utils:Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels.model_keras.raw.char_cnn.data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[0.05840052 0.07008284 0.27269873 0.59881794]
 [0.05837254 0.07006838 0.27273545 0.5988236 ]
 [0.05839446 0.07008235 0.27273878 0.5987844 ]
 ...
 [0.05892164 0.07064988 0.27310282 0.5973257 ]
 [0.05901353 0.07072373 0.27317798 0.5970848 ]
 [0.05899459 0.07070866 0.2731721  0.59712464]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/textcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.textcnn.py",
      "maxlen": 40,
      "max_features": 5,
      "embedding_dims": 50
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "engine": "adam",
      "loss": "binary_crossentropy",
      "metrics": [
        "accuracy"
      ],
      "batch_size": 1000,
      "epochs": 1
    },
    "out_pars": {
      "path": "mlmodels/output/textcnn_keras//model.h5",
      "model_path": "mlmodels/output/textcnn_keras/model.h5"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.textcnn.Model object at 0x7ff425e8b5c0> 

  #### Fit   ######################################################## 
Loading data...

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
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 9s - loss: 8.0270 - accuracy: 0.4765 
 3000/25000 [==>...........................] - ETA: 8s - loss: 8.0040 - accuracy: 0.4780
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.9388 - accuracy: 0.4823
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8844 - accuracy: 0.4858
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7816 - accuracy: 0.4925
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7827 - accuracy: 0.4924
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 4s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7350 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 3s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7107 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7207 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7067 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7102 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7073 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7148 - accuracy: 0.4969
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 367us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Predict   #################################################### 
Loading data...

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
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
model_keras/namentity_crm_bilstm.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.namentity_crm_bilstm.py",
      "embedding": 40,
      "optimizer": "rmsprop"
    },
    "data_pars": {
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
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 64
    },
    "out_pars": {
      "path": "ztest/ml_keras/namentity_crm_bilstm/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.namentity_crm_bilstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff425e7e730>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff425e7e730>

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7ff48efdbb70>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7ff48efdbb70>

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff425e98b70>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff425e98b70>
######## Error model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 408, in test_run_model
    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 260, in test_module
    model = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
    data_set, internal_states = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 183, in get_dataset
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 327, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 80, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
