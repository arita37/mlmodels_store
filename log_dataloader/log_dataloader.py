
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




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
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb8ea55eea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb8ea55eea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb955b15048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb955b15048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb96f843e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb96f843e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb9033948c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb9033948c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb9033948c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:22, 445494.63it/s] 96%|█████████▋| 9551872/9912422 [00:00<00:00, 635144.70it/s]9920512it [00:00, 46564267.48it/s]                           
0it [00:00, ?it/s]32768it [00:00, 632918.81it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464367.85it/s]1654784it [00:00, 11790961.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203653.09it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb9006b6518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb90235d7b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb903394510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb903394510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb903394510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:11,  1.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:11,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:10,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:09,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<02:08,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<02:08,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:30,  1.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:28,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:28,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:27,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:27,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:26,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:26,  1.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:25,  1.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:59,  2.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:59,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:59,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:59,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:58,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:58,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:57,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:57,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:57,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:56,  2.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:39,  3.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:39,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:39,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:39,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:38,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:38,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:38,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:38,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:37,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:37,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:37,  3.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:26,  4.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:26,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:26,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:25,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:25,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:25,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:25,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:25,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:24,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:24,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:24,  4.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:16,  6.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:10,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:10,  9.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 12.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 17.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 23.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 37.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 45.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 45.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 53.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.98 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.98 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.92 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ url]
0 examples [00:00, ? examples/s]2020-06-18 06:10:18.016962: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-18 06:10:18.032294: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-06-18 06:10:18.032583: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560e38dcea20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-18 06:10:18.032603: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
27 examples [00:00, 268.62 examples/s]153 examples [00:00, 351.46 examples/s]281 examples [00:00, 448.94 examples/s]396 examples [00:00, 549.18 examples/s]526 examples [00:00, 664.02 examples/s]651 examples [00:00, 772.20 examples/s]774 examples [00:00, 869.02 examples/s]905 examples [00:00, 966.27 examples/s]1023 examples [00:00, 1019.92 examples/s]1140 examples [00:01, 1051.29 examples/s]1256 examples [00:01, 1026.48 examples/s]1367 examples [00:01, 988.86 examples/s] 1475 examples [00:01, 1014.19 examples/s]1591 examples [00:01, 1053.15 examples/s]1708 examples [00:01, 1083.25 examples/s]1825 examples [00:01, 1106.14 examples/s]1947 examples [00:01, 1137.61 examples/s]2066 examples [00:01, 1150.85 examples/s]2184 examples [00:01, 1155.77 examples/s]2301 examples [00:02, 1157.85 examples/s]2418 examples [00:02, 1143.55 examples/s]2533 examples [00:02, 1116.60 examples/s]2646 examples [00:02, 1117.00 examples/s]2759 examples [00:02, 1103.56 examples/s]2870 examples [00:02, 1089.31 examples/s]2981 examples [00:02, 1094.51 examples/s]3091 examples [00:02, 1080.18 examples/s]3200 examples [00:02, 1075.10 examples/s]3308 examples [00:02, 1062.07 examples/s]3415 examples [00:03, 1060.63 examples/s]3524 examples [00:03, 1058.53 examples/s]3637 examples [00:03, 1077.47 examples/s]3754 examples [00:03, 1103.23 examples/s]3869 examples [00:03, 1116.77 examples/s]3981 examples [00:03, 1106.31 examples/s]4092 examples [00:03, 1103.32 examples/s]4208 examples [00:03, 1118.06 examples/s]4320 examples [00:03, 1111.41 examples/s]4435 examples [00:04, 1120.32 examples/s]4548 examples [00:04, 1121.90 examples/s]4661 examples [00:04, 1106.10 examples/s]4772 examples [00:04, 1101.02 examples/s]4891 examples [00:04, 1125.65 examples/s]5011 examples [00:04, 1143.63 examples/s]5126 examples [00:04, 1137.07 examples/s]5242 examples [00:04, 1143.77 examples/s]5357 examples [00:04, 1133.97 examples/s]5471 examples [00:04, 1134.22 examples/s]5585 examples [00:05, 1108.51 examples/s]5697 examples [00:05, 1093.69 examples/s]5807 examples [00:05, 1079.23 examples/s]5918 examples [00:05, 1085.94 examples/s]6039 examples [00:05, 1117.29 examples/s]6152 examples [00:05, 1099.01 examples/s]6266 examples [00:05, 1109.61 examples/s]6378 examples [00:05, 1110.92 examples/s]6490 examples [00:05, 1095.89 examples/s]6606 examples [00:05, 1114.23 examples/s]6721 examples [00:06, 1119.27 examples/s]6834 examples [00:06, 1101.18 examples/s]6945 examples [00:06, 1093.86 examples/s]7055 examples [00:06, 1089.44 examples/s]7168 examples [00:06, 1099.74 examples/s]7279 examples [00:06, 1086.39 examples/s]7388 examples [00:06, 1083.33 examples/s]7501 examples [00:06, 1094.39 examples/s]7611 examples [00:06, 1094.85 examples/s]7721 examples [00:06, 1088.87 examples/s]7837 examples [00:07, 1106.67 examples/s]7955 examples [00:07, 1126.04 examples/s]8068 examples [00:07, 1117.66 examples/s]8180 examples [00:07, 1088.81 examples/s]8290 examples [00:07, 1086.69 examples/s]8403 examples [00:07, 1099.16 examples/s]8520 examples [00:07, 1117.69 examples/s]8637 examples [00:07, 1130.59 examples/s]8758 examples [00:07, 1152.15 examples/s]8881 examples [00:07, 1173.09 examples/s]8999 examples [00:08, 1152.91 examples/s]9115 examples [00:08, 1131.42 examples/s]9231 examples [00:08, 1138.66 examples/s]9355 examples [00:08, 1164.40 examples/s]9484 examples [00:08, 1197.78 examples/s]9605 examples [00:08, 1174.13 examples/s]9723 examples [00:08, 1155.82 examples/s]9839 examples [00:08, 1135.52 examples/s]9965 examples [00:08, 1169.98 examples/s]10083 examples [00:09, 1117.90 examples/s]10196 examples [00:09, 1107.51 examples/s]10319 examples [00:09, 1139.48 examples/s]10434 examples [00:09, 1130.63 examples/s]10548 examples [00:09, 1099.45 examples/s]10666 examples [00:09, 1121.93 examples/s]10791 examples [00:09, 1155.76 examples/s]10913 examples [00:09, 1172.76 examples/s]11031 examples [00:09, 1172.26 examples/s]11149 examples [00:09, 1114.13 examples/s]11275 examples [00:10, 1153.20 examples/s]11392 examples [00:10, 1141.59 examples/s]11522 examples [00:10, 1183.16 examples/s]11646 examples [00:10, 1198.97 examples/s]11767 examples [00:10, 1181.62 examples/s]11886 examples [00:10, 1178.00 examples/s]12005 examples [00:10, 1177.20 examples/s]12125 examples [00:10, 1183.88 examples/s]12246 examples [00:10, 1191.06 examples/s]12367 examples [00:10, 1195.55 examples/s]12487 examples [00:11, 1159.66 examples/s]12610 examples [00:11, 1177.86 examples/s]12729 examples [00:11, 1173.70 examples/s]12848 examples [00:11, 1176.95 examples/s]12971 examples [00:11, 1191.84 examples/s]13091 examples [00:11, 1149.78 examples/s]13207 examples [00:11, 1124.35 examples/s]13326 examples [00:11, 1141.19 examples/s]13445 examples [00:11, 1153.40 examples/s]13563 examples [00:12, 1160.80 examples/s]13687 examples [00:12, 1181.48 examples/s]13809 examples [00:12, 1190.60 examples/s]13935 examples [00:12, 1210.43 examples/s]14060 examples [00:12, 1221.03 examples/s]14191 examples [00:12, 1245.34 examples/s]14316 examples [00:12, 1221.39 examples/s]14439 examples [00:12, 1195.50 examples/s]14559 examples [00:12, 1188.77 examples/s]14679 examples [00:12, 1175.32 examples/s]14797 examples [00:13, 1174.56 examples/s]14919 examples [00:13, 1185.82 examples/s]15045 examples [00:13, 1204.35 examples/s]15166 examples [00:13, 1191.91 examples/s]15286 examples [00:13, 1146.73 examples/s]15406 examples [00:13, 1161.47 examples/s]15535 examples [00:13, 1195.17 examples/s]15659 examples [00:13, 1207.27 examples/s]15787 examples [00:13, 1227.26 examples/s]15911 examples [00:13, 1224.30 examples/s]16034 examples [00:14, 1217.64 examples/s]16156 examples [00:14, 1190.62 examples/s]16276 examples [00:14, 1144.34 examples/s]16391 examples [00:14, 1134.54 examples/s]16517 examples [00:14, 1166.00 examples/s]16635 examples [00:14, 1151.47 examples/s]16751 examples [00:14, 1138.25 examples/s]16871 examples [00:14, 1153.70 examples/s]16988 examples [00:14, 1157.91 examples/s]17105 examples [00:15, 1159.88 examples/s]17228 examples [00:15, 1177.17 examples/s]17360 examples [00:15, 1214.86 examples/s]17488 examples [00:15, 1230.98 examples/s]17620 examples [00:15, 1255.54 examples/s]17755 examples [00:15, 1281.45 examples/s]17887 examples [00:15, 1291.41 examples/s]18017 examples [00:15, 1242.96 examples/s]18142 examples [00:15, 1193.28 examples/s]18263 examples [00:15, 1176.02 examples/s]18382 examples [00:16, 1143.33 examples/s]18497 examples [00:16, 1118.76 examples/s]18610 examples [00:16, 1101.21 examples/s]18731 examples [00:16, 1131.16 examples/s]18853 examples [00:16, 1152.72 examples/s]18969 examples [00:16, 1092.96 examples/s]19084 examples [00:16, 1107.76 examples/s]19196 examples [00:16, 1108.53 examples/s]19316 examples [00:16, 1133.39 examples/s]19430 examples [00:17, 1117.81 examples/s]19553 examples [00:17, 1146.92 examples/s]19669 examples [00:17, 1103.34 examples/s]19789 examples [00:17, 1129.45 examples/s]19908 examples [00:17, 1146.05 examples/s]20024 examples [00:17, 1073.85 examples/s]20151 examples [00:17, 1125.67 examples/s]20271 examples [00:17, 1146.90 examples/s]20390 examples [00:17, 1157.45 examples/s]20507 examples [00:17, 1151.58 examples/s]20623 examples [00:18, 1141.03 examples/s]20741 examples [00:18, 1149.60 examples/s]20864 examples [00:18, 1171.69 examples/s]20982 examples [00:18, 1173.00 examples/s]21106 examples [00:18, 1190.52 examples/s]21235 examples [00:18, 1215.59 examples/s]21357 examples [00:18, 1197.38 examples/s]21478 examples [00:18, 1199.82 examples/s]21602 examples [00:18, 1209.71 examples/s]21729 examples [00:18, 1225.76 examples/s]21859 examples [00:19, 1245.21 examples/s]21984 examples [00:19, 1226.86 examples/s]22111 examples [00:19, 1237.89 examples/s]22243 examples [00:19, 1260.52 examples/s]22370 examples [00:19, 1245.75 examples/s]22495 examples [00:19, 1220.31 examples/s]22618 examples [00:19, 1218.63 examples/s]22741 examples [00:19, 1209.74 examples/s]22863 examples [00:19, 1186.85 examples/s]22988 examples [00:20, 1204.41 examples/s]23109 examples [00:20, 1204.15 examples/s]23230 examples [00:20, 1134.19 examples/s]23354 examples [00:20, 1163.42 examples/s]23477 examples [00:20, 1182.45 examples/s]23608 examples [00:20, 1216.07 examples/s]23743 examples [00:20, 1250.53 examples/s]23871 examples [00:20, 1259.10 examples/s]23998 examples [00:20, 1239.81 examples/s]24127 examples [00:20, 1252.15 examples/s]24260 examples [00:21, 1272.51 examples/s]24388 examples [00:21, 1248.01 examples/s]24514 examples [00:21, 1246.61 examples/s]24639 examples [00:21, 1236.51 examples/s]24763 examples [00:21, 1220.35 examples/s]24888 examples [00:21, 1228.66 examples/s]25012 examples [00:21, 1222.59 examples/s]25142 examples [00:21, 1242.97 examples/s]25267 examples [00:21, 1206.24 examples/s]25388 examples [00:21, 1205.00 examples/s]25512 examples [00:22, 1212.85 examples/s]25639 examples [00:22, 1228.02 examples/s]25763 examples [00:22, 1231.21 examples/s]25887 examples [00:22, 1232.49 examples/s]26011 examples [00:22, 1223.92 examples/s]26136 examples [00:22, 1229.91 examples/s]26269 examples [00:22, 1257.43 examples/s]26395 examples [00:22, 1220.10 examples/s]26519 examples [00:22, 1224.83 examples/s]26643 examples [00:22, 1228.79 examples/s]26767 examples [00:23, 1210.42 examples/s]26889 examples [00:23, 1199.53 examples/s]27021 examples [00:23, 1232.11 examples/s]27148 examples [00:23, 1242.62 examples/s]27274 examples [00:23, 1247.57 examples/s]27405 examples [00:23, 1263.64 examples/s]27532 examples [00:23, 1237.93 examples/s]27662 examples [00:23, 1255.02 examples/s]27788 examples [00:23, 1250.13 examples/s]27914 examples [00:24, 1246.11 examples/s]28039 examples [00:24, 1241.17 examples/s]28164 examples [00:24, 1219.90 examples/s]28299 examples [00:24, 1254.09 examples/s]28425 examples [00:24, 1227.29 examples/s]28549 examples [00:24, 1221.57 examples/s]28672 examples [00:24, 1218.60 examples/s]28795 examples [00:24, 1219.03 examples/s]28920 examples [00:24, 1227.84 examples/s]29044 examples [00:24, 1228.28 examples/s]29169 examples [00:25, 1233.35 examples/s]29293 examples [00:25, 1234.86 examples/s]29417 examples [00:25, 1205.49 examples/s]29538 examples [00:25, 1181.48 examples/s]29658 examples [00:25, 1185.90 examples/s]29787 examples [00:25, 1213.36 examples/s]29913 examples [00:25, 1225.35 examples/s]30036 examples [00:25, 1159.74 examples/s]30153 examples [00:25, 1109.82 examples/s]30281 examples [00:25, 1155.19 examples/s]30407 examples [00:26, 1184.11 examples/s]30532 examples [00:26, 1200.51 examples/s]30662 examples [00:26, 1227.54 examples/s]30786 examples [00:26, 1214.57 examples/s]30919 examples [00:26, 1245.14 examples/s]31048 examples [00:26, 1255.92 examples/s]31176 examples [00:26, 1261.92 examples/s]31303 examples [00:26, 1264.16 examples/s]31430 examples [00:26, 1255.89 examples/s]31556 examples [00:26, 1254.19 examples/s]31682 examples [00:27, 1202.93 examples/s]31803 examples [00:27, 1186.29 examples/s]31923 examples [00:27, 1184.97 examples/s]32048 examples [00:27, 1201.63 examples/s]32178 examples [00:27, 1228.66 examples/s]32302 examples [00:27, 1231.69 examples/s]32427 examples [00:27, 1234.95 examples/s]32552 examples [00:27, 1238.69 examples/s]32676 examples [00:27, 1218.64 examples/s]32804 examples [00:28, 1235.37 examples/s]32933 examples [00:28, 1249.49 examples/s]33059 examples [00:28, 1204.93 examples/s]33180 examples [00:28, 1189.04 examples/s]33307 examples [00:28, 1211.02 examples/s]33436 examples [00:28, 1233.45 examples/s]33561 examples [00:28, 1235.19 examples/s]33685 examples [00:28, 1222.44 examples/s]33808 examples [00:28, 1220.51 examples/s]33931 examples [00:28, 1199.38 examples/s]34062 examples [00:29, 1228.41 examples/s]34186 examples [00:29, 1228.45 examples/s]34310 examples [00:29, 1224.38 examples/s]34443 examples [00:29, 1253.53 examples/s]34569 examples [00:29, 1230.94 examples/s]34703 examples [00:29, 1261.61 examples/s]34830 examples [00:29, 1250.20 examples/s]34956 examples [00:29, 1249.36 examples/s]35082 examples [00:29, 1235.60 examples/s]35214 examples [00:29, 1257.11 examples/s]35342 examples [00:30, 1262.71 examples/s]35475 examples [00:30, 1282.13 examples/s]35604 examples [00:30, 1265.71 examples/s]35731 examples [00:30, 1217.13 examples/s]35854 examples [00:30, 1208.76 examples/s]35985 examples [00:30, 1235.80 examples/s]36111 examples [00:30, 1241.27 examples/s]36236 examples [00:30, 1234.15 examples/s]36360 examples [00:30, 1199.73 examples/s]36483 examples [00:31, 1206.00 examples/s]36604 examples [00:31, 1166.19 examples/s]36724 examples [00:31, 1173.34 examples/s]36854 examples [00:31, 1207.75 examples/s]36976 examples [00:31, 1200.56 examples/s]37099 examples [00:31, 1208.22 examples/s]37226 examples [00:31, 1224.31 examples/s]37361 examples [00:31, 1257.77 examples/s]37494 examples [00:31, 1277.11 examples/s]37623 examples [00:31, 1216.96 examples/s]37746 examples [00:32, 1192.40 examples/s]37866 examples [00:32, 1187.12 examples/s]37990 examples [00:32, 1200.83 examples/s]38124 examples [00:32, 1238.36 examples/s]38249 examples [00:32, 1233.97 examples/s]38377 examples [00:32, 1244.70 examples/s]38502 examples [00:32, 1209.00 examples/s]38634 examples [00:32, 1238.52 examples/s]38763 examples [00:32, 1252.44 examples/s]38890 examples [00:32, 1255.38 examples/s]39016 examples [00:33, 1252.86 examples/s]39142 examples [00:33, 1252.27 examples/s]39268 examples [00:33, 1246.64 examples/s]39393 examples [00:33, 1246.52 examples/s]39518 examples [00:33, 1205.26 examples/s]39639 examples [00:33, 1184.92 examples/s]39758 examples [00:33, 1180.03 examples/s]39884 examples [00:33, 1200.86 examples/s]40005 examples [00:33, 1116.99 examples/s]40126 examples [00:34, 1139.91 examples/s]40257 examples [00:34, 1184.48 examples/s]40388 examples [00:34, 1217.98 examples/s]40519 examples [00:34, 1242.34 examples/s]40651 examples [00:34, 1263.12 examples/s]40779 examples [00:34, 1254.81 examples/s]40909 examples [00:34, 1266.72 examples/s]41041 examples [00:34, 1281.45 examples/s]41171 examples [00:34, 1286.44 examples/s]41300 examples [00:34, 1241.31 examples/s]41425 examples [00:35, 1178.29 examples/s]41559 examples [00:35, 1221.02 examples/s]41690 examples [00:35, 1244.64 examples/s]41824 examples [00:35, 1271.48 examples/s]41952 examples [00:35, 1242.09 examples/s]42077 examples [00:35, 1214.22 examples/s]42200 examples [00:35, 1189.73 examples/s]42326 examples [00:35, 1209.95 examples/s]42448 examples [00:35, 1210.27 examples/s]42572 examples [00:35, 1218.82 examples/s]42695 examples [00:36, 1184.16 examples/s]42814 examples [00:36, 1166.45 examples/s]42931 examples [00:36, 1115.83 examples/s]43055 examples [00:36, 1148.12 examples/s]43171 examples [00:36, 1088.31 examples/s]43291 examples [00:36, 1119.05 examples/s]43413 examples [00:36, 1146.62 examples/s]43529 examples [00:36, 1129.91 examples/s]43648 examples [00:36, 1147.23 examples/s]43764 examples [00:37, 1120.33 examples/s]43880 examples [00:37, 1131.61 examples/s]44000 examples [00:37, 1148.59 examples/s]44131 examples [00:37, 1190.66 examples/s]44268 examples [00:37, 1237.88 examples/s]44393 examples [00:37, 1224.68 examples/s]44522 examples [00:37, 1239.64 examples/s]44658 examples [00:37, 1272.60 examples/s]44794 examples [00:37, 1296.01 examples/s]44925 examples [00:37, 1300.00 examples/s]45056 examples [00:38, 1235.80 examples/s]45187 examples [00:38, 1256.86 examples/s]45314 examples [00:38, 1255.07 examples/s]45444 examples [00:38, 1266.71 examples/s]45578 examples [00:38, 1284.96 examples/s]45707 examples [00:38, 1268.89 examples/s]45835 examples [00:38, 1222.68 examples/s]45958 examples [00:38, 1203.13 examples/s]46079 examples [00:38, 1201.77 examples/s]46200 examples [00:39, 1198.58 examples/s]46326 examples [00:39, 1214.56 examples/s]46449 examples [00:39, 1216.15 examples/s]46571 examples [00:39, 1214.77 examples/s]46704 examples [00:39, 1238.23 examples/s]46829 examples [00:39, 1227.64 examples/s]46952 examples [00:39, 1214.57 examples/s]47074 examples [00:39, 1214.83 examples/s]47203 examples [00:39, 1234.83 examples/s]47334 examples [00:39, 1255.68 examples/s]47460 examples [00:40, 1250.14 examples/s]47586 examples [00:40, 1196.52 examples/s]47707 examples [00:40, 1152.17 examples/s]47835 examples [00:40, 1186.03 examples/s]47968 examples [00:40, 1224.59 examples/s]48092 examples [00:40, 1225.95 examples/s]48216 examples [00:40, 1190.18 examples/s]48340 examples [00:40, 1202.55 examples/s]48462 examples [00:40, 1206.22 examples/s]48588 examples [00:40, 1219.72 examples/s]48711 examples [00:41, 1207.17 examples/s]48832 examples [00:41, 1156.18 examples/s]48951 examples [00:41, 1159.82 examples/s]49080 examples [00:41, 1195.38 examples/s]49201 examples [00:41, 1187.59 examples/s]49330 examples [00:41, 1214.53 examples/s]49453 examples [00:41, 1217.61 examples/s]49581 examples [00:41, 1235.67 examples/s]49705 examples [00:41, 1229.48 examples/s]49833 examples [00:41, 1242.65 examples/s]49962 examples [00:42, 1254.91 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6345/50000 [00:00<00:00, 63447.19 examples/s] 40%|███▉      | 19909/50000 [00:00<00:00, 75501.71 examples/s] 69%|██████▊   | 34292/50000 [00:00<00:00, 88050.09 examples/s] 99%|█████████▊| 49279/50000 [00:00<00:00, 100484.42 examples/s]                                                                0 examples [00:00, ? examples/s]81 examples [00:00, 807.39 examples/s]205 examples [00:00, 900.39 examples/s]329 examples [00:00, 979.30 examples/s]453 examples [00:00, 1043.18 examples/s]573 examples [00:00, 1084.06 examples/s]684 examples [00:00, 1089.91 examples/s]799 examples [00:00, 1106.36 examples/s]922 examples [00:00, 1139.82 examples/s]1044 examples [00:00, 1160.81 examples/s]1162 examples [00:01, 1164.42 examples/s]1277 examples [00:01, 1141.37 examples/s]1404 examples [00:01, 1174.98 examples/s]1522 examples [00:01, 1146.92 examples/s]1637 examples [00:01, 1130.71 examples/s]1772 examples [00:01, 1187.89 examples/s]1892 examples [00:01, 1151.91 examples/s]2016 examples [00:01, 1174.48 examples/s]2141 examples [00:01, 1195.81 examples/s]2274 examples [00:01, 1231.65 examples/s]2410 examples [00:02, 1266.19 examples/s]2538 examples [00:02, 1239.28 examples/s]2664 examples [00:02, 1244.04 examples/s]2798 examples [00:02, 1269.53 examples/s]2934 examples [00:02, 1293.78 examples/s]3067 examples [00:02, 1303.37 examples/s]3198 examples [00:02, 1286.15 examples/s]3327 examples [00:02, 1271.79 examples/s]3463 examples [00:02, 1296.85 examples/s]3596 examples [00:02, 1305.77 examples/s]3727 examples [00:03, 1299.70 examples/s]3858 examples [00:03, 1262.63 examples/s]3986 examples [00:03, 1266.57 examples/s]4114 examples [00:03, 1269.43 examples/s]4242 examples [00:03, 1271.88 examples/s]4371 examples [00:03, 1273.89 examples/s]4499 examples [00:03, 1252.34 examples/s]4625 examples [00:03, 1243.75 examples/s]4757 examples [00:03, 1264.38 examples/s]4894 examples [00:03, 1292.09 examples/s]5027 examples [00:04, 1301.21 examples/s]5158 examples [00:04, 1300.15 examples/s]5289 examples [00:04, 1275.36 examples/s]5420 examples [00:04, 1283.69 examples/s]5549 examples [00:04, 1283.38 examples/s]5684 examples [00:04, 1300.19 examples/s]5815 examples [00:04, 1199.47 examples/s]5945 examples [00:04, 1225.21 examples/s]6069 examples [00:04, 1229.26 examples/s]6202 examples [00:05, 1256.26 examples/s]6335 examples [00:05, 1276.86 examples/s]6466 examples [00:05, 1284.81 examples/s]6600 examples [00:05, 1299.77 examples/s]6731 examples [00:05, 1258.52 examples/s]6864 examples [00:05, 1277.53 examples/s]6993 examples [00:05, 1272.07 examples/s]7121 examples [00:05, 1267.71 examples/s]7251 examples [00:05, 1275.12 examples/s]7383 examples [00:05, 1286.35 examples/s]7521 examples [00:06, 1311.64 examples/s]7653 examples [00:06, 1305.85 examples/s]7784 examples [00:06, 1277.74 examples/s]7913 examples [00:06, 1263.81 examples/s]8040 examples [00:06, 1226.18 examples/s]8164 examples [00:06, 1189.59 examples/s]8291 examples [00:06, 1211.33 examples/s]8419 examples [00:06, 1229.81 examples/s]8547 examples [00:06, 1242.34 examples/s]8678 examples [00:06, 1260.19 examples/s]8815 examples [00:07, 1290.86 examples/s]8945 examples [00:07, 1293.24 examples/s]9075 examples [00:07, 1277.53 examples/s]9203 examples [00:07, 1265.00 examples/s]9330 examples [00:07, 1225.04 examples/s]9453 examples [00:07, 1187.77 examples/s]9573 examples [00:07, 1156.55 examples/s]9690 examples [00:07, 1157.21 examples/s]9812 examples [00:07, 1173.99 examples/s]9937 examples [00:08, 1193.67 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ZFZOW/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ZFZOW/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb9033948c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb9033948c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb9033948c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb8888367f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb8898ee9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb9033948c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb9033948c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb9033948c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb9024d2ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb90235d208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb884f75400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb884f75400> 

  function with postional parmater data_info <function split_train_valid at 0x7fb884f75400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb884f75510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb884f75510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb884f75510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=9be7d4da0c4c1f6aa2265ff090f2d34c04ab35fb8e85439ffdbfdecac59564f5
  Stored in directory: /tmp/pip-ephem-wheel-cache-pr1uuj6o/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:06:16, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:24:37, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:32:32, 20.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 721k/862M [00:01<8:05:21, 29.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.82M/862M [00:01<5:39:44, 42.2kB/s].vector_cache/glove.6B.zip:   1%|          | 5.01M/862M [00:01<3:57:04, 60.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.87M/862M [00:01<2:45:22, 86.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:01<1:55:19, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:01<1:20:31, 175kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<56:11, 250kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 24.8M/862M [00:01<39:15, 355kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.9M/862M [00:02<27:27, 506kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:02<19:14, 719kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.6M/862M [00:02<13:30, 1.02MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<09:30, 1.44MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<08:17, 1.65MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.4M/862M [00:02<05:52, 2.32MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<04:11, 3.23MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<03:49, 3.53MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.8M/862M [00:03<02:54, 4.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<05:53, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<08:20, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<06:43, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<05:01, 2.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:24, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:33, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:06, 2.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<05:58, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<06:15, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:09<04:53, 2.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<05:48, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<06:06, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<04:42, 2.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<03:27, 3.81MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<17:41, 743kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<14:08, 929kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<10:27, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:13<07:32, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<09:38, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:15<08:48, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<04:48, 2.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<12:58, 1.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:18<18:20, 709kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:18<14:51, 875kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:18<10:54, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:50, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<19:32, 663kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<19:03, 679kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:20<14:45, 876kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:20<10:39, 1.21MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:40, 1.68MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<57:40, 223kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:21<45:43, 281kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:22<33:07, 388kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:22<23:42, 542kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<16:46, 764kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<16:48, 762kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<17:06, 748kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:24<13:18, 961kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:24<09:35, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<06:58, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:25<13:18, 957kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:25<14:13, 895kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:26<11:15, 1.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<08:08, 1.56MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:09, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<09:22, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<13:34, 933kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<11:15, 1.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<08:18, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:04, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:45, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<10:07, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<07:55, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:48, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<04:23, 2.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<12:18, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<15:35, 803kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<12:33, 997kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<09:09, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:40, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:01, 2.48MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:43, 3.33MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:01, 4.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<22:46, 544kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<26:21, 470kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<21:25, 579kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<15:43, 788kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<11:14, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:11, 1.51MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<17:08, 720kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<14:56, 826kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<11:03, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<08:03, 1.53MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:53, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<11:44, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<13:46, 891kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<11:06, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<08:09, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:57, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<12:34, 970kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<14:56, 817kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<11:54, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<08:39, 1.41MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:18, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<12:53, 941kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<14:33, 833kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<11:25, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<08:23, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:10, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<04:35, 2.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<21:34, 559kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<21:10, 569kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<16:15, 741kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<11:43, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<08:23, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<13:24, 894kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<14:50, 808kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<11:47, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<08:38, 1.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:18, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<12:17, 970kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<14:31, 821kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<11:34, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<08:28, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:10, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<13:11, 899kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<15:08, 783kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<11:47, 1.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<08:39, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:19, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<04:40, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<50:24, 234kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<40:37, 290kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<29:52, 394kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<21:15, 553kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<15:07, 776kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<24:39, 475kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<28:59, 404kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<22:58, 510kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<17:33, 667kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<12:34, 929kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<09:08, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<06:39, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<07:31, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<05:28, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<04:01, 2.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<18:12, 636kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<24:18, 476kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<19:22, 597kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<14:08, 818kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<10:17, 1.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<07:26, 1.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<09:57, 1.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<09:35, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:22, 1.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<05:23, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:01, 2.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<22:11, 515kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<20:44, 551kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<15:49, 723kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<11:23, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<08:09, 1.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<11:47, 964kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<13:27, 844kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<10:30, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<07:50, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:06<05:42, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<04:16, 2.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<17:41, 639kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<17:31, 645kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<13:34, 832kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<09:47, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<07:06, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<09:18, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<11:39, 964kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<09:25, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<05:04, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<13:01, 857kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<14:13, 785kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<11:09, 999kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<08:07, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<05:52, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<11:14, 986kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<12:56, 857kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<10:16, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<07:29, 1.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:26, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<09:45, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<11:25, 965kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<09:08, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<06:39, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<04:56, 2.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<07:31, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<09:50, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<07:47, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<05:50, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<04:18, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<06:52, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<09:20, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<07:48, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<06:01, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<04:41, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<04:20, 2.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<03:39, 2.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<03:04, 3.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<13:04, 828kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<20:05, 538kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<16:39, 649kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<12:17, 879kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<09:18, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<06:05, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:58, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<04:01, 2.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<03:22, 3.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<10:21, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<10:14, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<07:51, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<06:04, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<04:54, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<03:53, 2.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<03:56, 2.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<03:24, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<02:49, 3.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<1:02:34, 171kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<46:35, 229kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<33:17, 320kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<23:48, 447kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<17:10, 619kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<12:40, 838kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<09:28, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<41:57, 253kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<35:15, 301kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<25:49, 410kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<18:38, 568kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<13:34, 779kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<10:04, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<07:30, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<04:43, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<17:02, 619kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<14:41, 717kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<10:56, 962kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<08:09, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<06:22, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:04, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<03:22, 3.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<15:16, 686kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<16:30, 634kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<12:47, 818kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<09:36, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<07:10, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:32<05:24, 1.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<04:31, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<03:50, 2.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<10:13, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<09:55, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<08:10, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<06:15, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<05:10, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<04:14, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<04:03, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<03:30, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<03:23, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<03:12, 3.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<10:15, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<09:12, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<07:27, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<05:54, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<04:57, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<04:06, 2.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<03:53, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<03:29, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:36<03:13, 3.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<03:12, 3.20MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<02:56, 3.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<19:21, 530kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<23:47, 431kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<19:11, 535kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<14:36, 702kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<10:52, 943kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:40<08:23, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:40<06:35, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<05:16, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<04:22, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<03:42, 2.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<02:56, 3.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<45:44, 223kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<35:39, 286kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<26:31, 384kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<19:24, 525kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<14:10, 718kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<10:40, 953kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<08:03, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:42<06:23, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<05:03, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:42<03:36, 2.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<32:39, 310kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<30:02, 337kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<23:24, 432kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:44<17:17, 585kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<12:46, 791kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<09:36, 1.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<07:23, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<05:50, 1.73MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<04:45, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<03:58, 2.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<03:27, 2.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<10:54, 923kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<11:13, 895kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<09:24, 1.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<07:31, 1.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<06:01, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<04:48, 2.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<04:05, 2.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<03:27, 2.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<03:09, 3.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<02:47, 3.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:46<02:40, 3.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<12:35, 793kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<12:29, 800kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<10:27, 955kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:48<08:47, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<06:46, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<05:28, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<04:23, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<03:49, 2.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<03:13, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<03:00, 3.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<02:37, 3.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<13:14, 749kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<12:35, 788kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<09:50, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<07:35, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<05:50, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<04:46, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<03:52, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<03:23, 2.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<03:10, 3.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<02:45, 3.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<02:30, 3.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<02:19, 4.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<02:10, 4.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<02:04, 4.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:53<12:42, 770kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<18:37, 526kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<15:21, 638kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<11:19, 863kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<08:32, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:44, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<05:08, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:54<04:11, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<03:20, 2.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<05:01, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<09:21, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<10:17, 944kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<07:59, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<06:09, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:19, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:55<04:46, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<04:40, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<03:55, 2.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:56<03:21, 2.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<02:58, 3.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<02:40, 3.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<10:14, 942kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<10:25, 926kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<08:01, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<06:03, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<05:02, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<04:43, 2.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<03:58, 2.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<03:22, 2.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<02:51, 3.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<02:33, 3.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<15:14, 629kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<13:29, 711kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<10:12, 938kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<07:39, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<05:49, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<04:39, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<03:52, 2.46MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:59<03:08, 3.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<10:28, 909kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<12:34, 757kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<10:09, 936kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<07:48, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<05:55, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<04:40, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<03:45, 2.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:02<03:06, 3.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<02:40, 3.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<09:30, 993kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<09:16, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<07:15, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:44, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<04:32, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<03:36, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<03:02, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<02:33, 3.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<02:17, 4.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<55:00, 171kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<41:00, 229kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<29:19, 320kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<21:21, 438kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<15:32, 602kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<11:22, 822kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<08:25, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<06:20, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<14:50, 628kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<12:51, 724kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<09:37, 967kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<08:03, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<06:09, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<04:44, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<03:06, 2.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<02:39, 3.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<12:36, 733kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<11:16, 820kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<08:34, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<06:37, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<05:14, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<04:02, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<03:24, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<02:46, 3.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<08:41, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<08:29, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<06:34, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<05:32, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<04:22, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<03:30, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<02:55, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<02:28, 3.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<02:12, 4.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<23:09, 393kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<18:45, 485kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<13:47, 660kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<10:11, 891kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<07:38, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<05:44, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<04:30, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<03:32, 2.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<10:34, 855kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<12:28, 725kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<09:50, 918kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<08:04, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<06:10, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:43, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<03:45, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:15<03:01, 2.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<02:32, 3.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:17<08:44, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:17<08:16, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<06:21, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<05:21, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<04:23, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<03:29, 2.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<02:50, 3.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<02:22, 3.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:19<06:10, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:19<08:21, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<06:51, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:19<05:12, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<04:02, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<03:10, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<02:36, 3.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<02:09, 4.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<3:12:34, 45.9kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<2:18:48, 63.6kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<1:37:58, 90.1kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<1:09:14, 127kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<48:44, 181kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<34:26, 255kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<24:22, 360kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<17:22, 504kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:23<27:55, 314kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<23:10, 378kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<16:58, 516kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<12:16, 712kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<08:53, 982kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<06:30, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<14:21, 606kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<13:11, 659kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<10:07, 858kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<07:45, 1.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<05:42, 1.52MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<04:16, 2.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<03:15, 2.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:26<08:43, 988kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:27<08:59, 960kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:27<07:18, 1.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<06:05, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<04:28, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<03:32, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<02:51, 2.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<02:23, 3.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<17:35, 486kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:29<16:35, 516kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:29<12:41, 674kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<09:23, 910kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<06:53, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<05:13, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<03:58, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<03:09, 2.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<11:25, 743kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:31<11:31, 737kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:31<09:11, 923kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:31<06:52, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<05:08, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<03:04, 2.75MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<02:27, 3.42MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<55:32, 152kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:33<42:22, 199kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:33<30:29, 276kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<21:38, 388kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<15:24, 544kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<11:06, 753kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<08:02, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<1:10:05, 119kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<52:20, 159kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:35<37:54, 220kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:35<26:59, 309kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<19:14, 433kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:35<13:42, 606kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<09:53, 838kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:36<10:32, 785kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<10:29, 788kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:37<08:32, 969kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:37<06:28, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<04:50, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<03:39, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<02:51, 2.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<06:10, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<07:16, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:39<06:05, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<04:40, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<03:34, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<02:46, 2.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<02:25, 3.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:42<15:03, 541kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<18:41, 436kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<14:53, 547kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<11:24, 713kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:43<08:17, 978kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:43<06:17, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:43<04:45, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:43<03:43, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:43<02:56, 2.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:44<09:44, 829kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<08:51, 911kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<07:03, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<05:36, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:45<03:23, 2.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:45<02:41, 2.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:45<02:14, 3.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:46<06:33, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<08:19, 962kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<07:15, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<04:14, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<03:18, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:47<02:38, 3.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:47<02:11, 3.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:48<07:13, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:48<08:37, 919kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<07:32, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<05:49, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:49<04:23, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:49<03:25, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:49<02:40, 2.94MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:49<02:14, 3.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:49<01:50, 4.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:49<01:38, 4.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<01:26, 5.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<34:30, 227kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<32:03, 244kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:52<24:03, 325kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:52<17:16, 453kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:52<12:27, 627kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:52<08:59, 867kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:52<06:38, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:52<04:54, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:53<10:34, 734kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:53<10:55, 711kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:53<08:28, 915kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:54<06:12, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:54<04:38, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<03:35, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:56<11:16, 682kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:57<15:37, 493kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:57<13:00, 591kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:57<10:00, 768kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<07:20, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:57<05:27, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<03:11, 2.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:57<02:32, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<1:05:40, 116kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:58<46:44, 163kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:58<32:57, 231kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:58<23:22, 325kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:58<16:37, 456kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<11:56, 634kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:59<08:47, 860kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<33:58, 222kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<26:56, 281kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:00<19:39, 384kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:00<14:04, 536kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:00<10:05, 745kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<07:25, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<05:30, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<11:03, 677kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<11:09, 671kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:02<08:36, 868kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:02<06:22, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:02<04:43, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<03:40, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<02:55, 2.54MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<07:07, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:04<08:04, 919kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:04<06:24, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:04<04:48, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:04<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:05<02:54, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:05<02:22, 3.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:06<08:12, 896kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:06<08:49, 833kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:06<06:56, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:06<05:10, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:06<03:53, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:07<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:08<09:38, 755kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:08<10:47, 674kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:08<08:34, 849kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:08<06:19, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:08<04:46, 1.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:08<04:02, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:09<02:38, 2.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:10<05:16, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:10<05:23, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:10<04:10, 1.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:10<03:12, 2.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:10<02:41, 2.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:10<02:11, 3.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:10<02:00, 3.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<01:40, 4.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<29:30, 242kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<23:53, 299kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<17:31, 407kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:12<12:32, 568kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:12<09:00, 790kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:12<06:44, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:12<05:02, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:13<03:49, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<23:03, 307kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<19:21, 365kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<14:17, 494kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:14<10:16, 686kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:14<07:30, 938kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:14<05:33, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:14<04:12, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:15<03:13, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:16<32:30, 216kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:16<25:40, 273kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:16<18:41, 374kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:16<13:23, 522kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:16<09:35, 727kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:16<07:10, 971kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:16<05:19, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<07:05, 977kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<07:53, 879kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<06:07, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:18<04:35, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:18<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:18<02:44, 2.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:18<02:08, 3.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:20<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:20<06:59, 983kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:20<05:35, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:20<04:09, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:20<03:12, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:20<02:27, 2.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:20<02:00, 3.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:23<13:59, 486kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:23<15:26, 440kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:23<12:54, 526kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<09:47, 693kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:23<07:08, 949kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<05:15, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<03:55, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<03:00, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:25<05:51, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:25<06:41, 1.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:25<05:42, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:25<04:24, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:25<03:19, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:25<02:36, 2.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<02:02, 3.26MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<01:42, 3.92MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<16:59, 392kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<14:28, 460kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<11:08, 597kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:27<08:17, 803kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:27<06:01, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:27<04:30, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<03:21, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<02:37, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:28<12:47, 515kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<11:30, 573kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<09:07, 722kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:29<06:49, 964kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:29<05:00, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<03:47, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:29<02:51, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:29<02:16, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:30<13:43, 475kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:31<12:21, 527kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:31<09:18, 700kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:31<06:45, 961kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:31<04:56, 1.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:31<03:37, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<09:25, 685kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<12:03, 535kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<10:15, 629kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:33<07:54, 816kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:33<05:47, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<04:18, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:33<03:13, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:33<02:29, 2.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:35<06:39, 958kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:35<07:07, 896kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:35<05:31, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:35<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<03:08, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:35<02:29, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:35<01:54, 3.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:37<09:10, 689kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:37<08:41, 727kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:37<06:47, 930kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:37<05:02, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:37<03:44, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:37<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:37<02:10, 2.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:39<07:43, 810kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:39<07:38, 817kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:39<06:06, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:39<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<03:34, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:39<02:50, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:39<02:20, 2.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:39<01:59, 3.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:39<01:44, 3.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:40<06:14, 990kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:41<06:05, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:41<04:53, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:41<03:44, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:41<02:57, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<02:23, 2.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<01:59, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:41<01:42, 3.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:41<01:29, 4.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:42<14:50, 412kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:43<14:12, 430kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:43<10:47, 566kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:43<07:49, 779kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:43<05:43, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<04:21, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:43<03:17, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:43<02:41, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:44<05:34, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:45<05:29, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:45<04:16, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:45<03:17, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<02:34, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:45<02:05, 2.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:45<01:44, 3.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:46<37:26, 159kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<29:11, 205kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:47<21:13, 281kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:47<15:08, 393kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<10:52, 546kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:47<07:49, 757kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:47<05:48, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:47<04:18, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:48<2:02:10, 48.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:49<1:27:02, 67.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:49<1:01:15, 96.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<43:04, 136kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<30:20, 193kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:49<21:31, 272kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:49<15:17, 382kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:49<11:00, 530kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<1:29:10, 65.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<1:05:38, 88.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:51<46:34, 125kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:51<32:53, 177kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:51<23:19, 249kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<16:33, 350kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<11:46, 491kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:51<08:37, 670kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<10:31, 548kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<10:15, 562kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:53<07:56, 724kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:53<05:48, 988kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:53<04:20, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<03:18, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<02:36, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:53<02:07, 2.68MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:54<05:56, 960kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<05:39, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:55<04:21, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:55<03:17, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<02:29, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<02:09, 2.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:55<01:42, 3.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<05:34, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<06:43, 836kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:57<05:25, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:57<04:05, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<03:05, 1.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<02:21, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:57<02:00, 2.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<04:02, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<05:37, 988kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<04:37, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:59<03:31, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:59<02:41, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:59<02:08, 2.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:59<01:44, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<10:07, 542kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<09:50, 558kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<07:32, 727kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [04:01<05:30, 994kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<04:10, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<03:06, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:01<02:29, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:01<01:55, 2.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<20:46, 261kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<16:59, 319kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<12:31, 432kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:03<08:58, 601kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:03<06:27, 834kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<04:56, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<03:40, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:03<03:00, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:04<06:47, 787kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:04<06:19, 847kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<04:48, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:05<03:38, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:05<02:48, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<02:12, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<01:48, 2.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:06<09:53, 534kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:06<09:36, 550kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<07:19, 721kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<05:25, 972kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:07<04:02, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:07<03:02, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<02:23, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:07<01:52, 2.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:08<05:06, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:08<06:11, 842kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:08<04:57, 1.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<03:40, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:09<02:46, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<02:11, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:09<01:43, 2.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:09<01:27, 3.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:10<32:32, 158kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<25:07, 205kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:10<18:10, 283kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:10<12:54, 397kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:11<09:13, 553kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:11<06:39, 765kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<06:48, 745kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<07:04, 717kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:12<05:31, 918kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<04:05, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:13<03:01, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:13<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:13<01:49, 2.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<07:27, 671kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<07:19, 683kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<05:40, 882kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:14<04:12, 1.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<03:08, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:15<02:23, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<03:59, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<04:51, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<03:56, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:17<02:17, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<01:46, 2.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:17<01:28, 3.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:18<07:13, 673kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:18<07:05, 686kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:18<05:30, 883kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:18<04:04, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<03:02, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:19<02:18, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:20<04:01, 1.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:20<04:48, 998kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<03:52, 1.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:20<02:56, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<02:13, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<01:44, 2.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:22<03:30, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:22<04:37, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<03:45, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:22<02:50, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:22<02:10, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:23<01:41, 2.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:24<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<04:35, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:24<03:43, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<02:48, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:25<01:41, 2.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<03:28, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<04:32, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<03:40, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<02:45, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<02:03, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:26<01:41, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:27<01:20, 3.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<05:31, 820kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<05:45, 786kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<04:29, 1.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<03:19, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:28<02:31, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:28<01:54, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:30<03:55, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:30<04:36, 966kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:30<03:41, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<02:44, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<02:06, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:30<01:36, 2.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:30<01:19, 3.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:33<09:08, 480kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:33<10:06, 434kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:33<08:09, 537kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:34<05:55, 738kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:34<04:20, 1.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:34<03:10, 1.37MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:34<02:22, 1.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:34<01:48, 2.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:35<3:08:02, 23.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:35<2:13:06, 32.4kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:35<1:33:21, 46.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:35<1:05:26, 65.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:36<45:56, 93.4kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:36<32:11, 133kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:36<22:36, 188kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<18:11, 233kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<14:08, 300kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:37<10:15, 413kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:37<07:21, 575kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:38<05:25, 778kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:38<03:55, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:38<02:53, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:39<04:24, 948kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:39<04:27, 935kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:39<03:24, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:39<02:35, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:40<01:59, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:40<01:30, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:40<01:10, 3.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:41<08:28, 485kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:41<07:08, 575kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:41<05:14, 783kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:41<03:49, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<02:45, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:42<02:03, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<04:35, 881kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<05:27, 739kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<04:24, 917kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:43<03:14, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:43<02:23, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:44<01:45, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:45<03:25, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:45<03:23, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:45<02:34, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:45<01:57, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:45<01:32, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:46<01:14, 3.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:46<00:59, 3.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:47<08:15, 473kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:47<06:45, 578kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:47<04:56, 787kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:47<03:38, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:47<02:43, 1.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:48<02:03, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:48<01:31, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:49<24:07, 159kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:49<17:40, 217kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:49<12:30, 305kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:49<08:50, 430kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<06:13, 606kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:51<06:23, 589kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:51<05:55, 635kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:51<04:30, 833kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:51<03:14, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<02:20, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:53<03:08, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:53<03:37, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:53<02:49, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:53<02:07, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:33, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:53<01:10, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:55<08:06, 448kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:55<06:58, 520kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:55<05:11, 696kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:55<03:45, 958kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<02:43, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:55<02:01, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:57<03:03, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:57<02:42, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:57<02:02, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:57<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:59<02:00, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:59<02:30, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:59<01:58, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:59<01:30, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:59<01:06, 3.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:01<02:38, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:01<02:47, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:01<02:10, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [05:01<01:37, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:01<01:12, 2.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:03<02:01, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:03<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:03<01:45, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:03<01:18, 2.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<00:59, 3.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:05<13:12, 248kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:05<10:31, 311kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:05<07:38, 428kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:05<05:29, 593kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:05<03:53, 833kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<03:38, 880kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<03:43, 862kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<02:53, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:07<02:10, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:07<01:35, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<01:09, 2.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:09<10:14, 307kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:09<08:16, 379kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:09<06:03, 517kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:09<04:16, 726kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:11<03:44, 822kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:11<03:42, 828kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:11<02:49, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:11<02:03, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:11<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:14<05:03, 593kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:14<06:32, 459kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:14<05:17, 567kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:14<03:51, 773kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:14<02:43, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:15<02:40, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:15<02:03, 1.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:15<01:33, 1.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:15<01:08, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:17<01:49, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:17<02:10, 1.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:17<01:45, 1.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:17<01:21, 2.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:18<01:05, 2.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:19<01:17, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:19<02:18, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:19<02:00, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:19<01:33, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:20<01:08, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<00:51, 3.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:21<03:05, 881kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:21<03:24, 800kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:21<02:40, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:21<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:22<01:26, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<01:04, 2.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:23<03:58, 667kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:23<03:59, 666kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:23<03:05, 859kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:23<02:18, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:23<01:38, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:25<04:18, 601kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:25<04:06, 630kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:25<03:14, 798kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:25<02:23, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:25<01:47, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:26<01:16, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:27<02:43, 923kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:27<02:43, 925kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:27<02:10, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:27<01:36, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:27<01:11, 2.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:28<00:53, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:29<02:07, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:29<03:02, 804kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:29<02:25, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:29<01:55, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:29<01:23, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<01:02, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:31<02:23, 999kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:31<02:22, 999kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:31<01:50, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:31<01:22, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:31<01:00, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:31<00:46, 3.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:33<02:31, 914kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:33<03:04, 751kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:33<02:28, 933kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:33<01:48, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:33<01:18, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:35<01:53, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:35<02:34, 870kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:35<02:04, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:35<01:31, 1.46MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:36<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:36<00:50, 2.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:37<01:48, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:37<01:33, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:37<01:09, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:38<00:51, 2.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:39<01:25, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:39<02:10, 968kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:39<01:48, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:39<01:19, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:40<00:57, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:40<00:44, 2.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:41<03:11, 638kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:41<02:45, 736kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:41<02:03, 982kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:41<01:28, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:42<01:03, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:43<02:33, 768kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:43<02:52, 684kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:43<02:13, 881kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:43<01:36, 1.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:43<01:10, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:44<00:52, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:45<01:31, 1.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:45<01:33, 1.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:45<01:13, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:45<00:53, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:45<00:39, 2.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:47<02:05, 877kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:47<02:29, 736kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:47<01:56, 943kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:47<01:26, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:47<01:02, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:48<00:46, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:50<02:17, 769kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:51<03:21, 524kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:51<02:45, 635kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:51<02:01, 861kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:51<01:26, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:51<01:02, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:51<00:44, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:52<01:11, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:52<00:54, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:52<00:41, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:52<00:30, 3.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<00:24, 3.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:54<03:12, 495kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:54<03:07, 508kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:54<02:23, 662kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:54<01:42, 913kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:55<01:13, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:56<01:27, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:56<01:25, 1.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:56<01:04, 1.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:56<00:47, 1.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:56<00:34, 2.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:58<00:59, 1.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:58<01:29, 968kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:58<01:14, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:58<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:58<00:39, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:59<00:30, 2.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:00<01:28, 940kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:00<01:27, 949kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:00<01:07, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:00<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:00<00:35, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:02<01:05, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:02<01:29, 878kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:02<01:14, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:02<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:02<00:39, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:04<00:53, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:04<01:06, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:04<00:52, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:04<00:39, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:04<00:29, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:04<00:23, 3.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:05<00:18, 3.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:06<02:18, 510kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:06<02:01, 578kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:06<01:31, 767kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:06<01:04, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:06<00:46, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:06<00:35, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:08<01:07, 983kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:08<01:10, 935kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:08<00:54, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:08<00:39, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<00:29, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:08<00:22, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:10<00:43, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:10<01:14, 830kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:10<01:02, 985kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:10<00:46, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:10<00:33, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:10<00:25, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:12<00:43, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:12<00:49, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:12<00:39, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:29, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:12<00:21, 2.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:14<00:36, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:14<01:03, 842kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:14<00:53, 995kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:14<00:39, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:28, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:14<00:21, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:14<00:16, 3.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:16<09:23, 88.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:16<07:04, 117kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:16<05:03, 163kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:16<03:31, 231kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<02:27, 325kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:17<01:43, 456kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:17<01:11, 641kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:18<01:37, 468kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:18<01:23, 546kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:18<01:01, 733kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:43, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:18<00:30, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:19<00:22, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:20<01:05, 632kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:20<00:59, 694kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:20<00:44, 917kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:20<00:31, 1.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:20<00:22, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:22<00:31, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:22<00:34, 1.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:22<00:26, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:22<00:19, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:22<00:14, 2.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:22<00:10, 3.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:24<00:30, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:24<00:32, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:24<00:24, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:24<00:17, 1.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:24<00:12, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:26<00:21, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:26<00:24, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:26<00:19, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:26<00:13, 1.98MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:26<00:09, 2.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:28<00:17, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:28<00:20, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:28<00:15, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:28<00:11, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:28<00:08, 2.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:28<00:06, 3.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:30<00:16, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:30<00:18, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:30<00:13, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:30<00:09, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:30<00:07, 2.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:30<00:05, 3.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:32<00:25, 645kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:32<00:29, 555kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:32<00:23, 697kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:32<00:16, 951kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:32<00:10, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:07, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:34<00:43, 287kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:35<00:42, 291kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:35<00:32, 374kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:35<00:23, 503kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:35<00:16, 700kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:35<00:10, 967kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:35<00:06, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:36<00:08, 997kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:37<00:08, 984kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:37<00:06, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:37<00:04, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:37<00:02, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:37<00:01, 2.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:38<00:03, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:39<00:03, 1.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:39<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:39<00:01, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:39<00:00, 2.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:39<00:00, 3.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:40<00:00, 1.21MB/s].vector_cache/glove.6B.zip: 862MB [06:40, 2.15MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<16:35:21,  6.70it/s]  0%|          | 876/400000 [00:00<11:35:27,  9.57it/s]  0%|          | 1889/400000 [00:00<8:05:46, 13.66it/s]  1%|          | 2901/400000 [00:00<5:39:22, 19.50it/s]  1%|          | 3915/400000 [00:00<3:57:09, 27.84it/s]  1%|          | 4977/400000 [00:00<2:45:44, 39.72it/s]  2%|▏         | 6087/400000 [00:00<1:55:52, 56.66it/s]  2%|▏         | 7078/400000 [00:00<1:21:06, 80.74it/s]  2%|▏         | 8175/400000 [00:00<56:47, 114.98it/s]   2%|▏         | 9175/400000 [00:01<39:51, 163.45it/s]  3%|▎         | 10248/400000 [00:01<28:00, 231.99it/s]  3%|▎         | 11317/400000 [00:01<19:43, 328.36it/s]  3%|▎         | 12350/400000 [00:01<13:57, 462.76it/s]  3%|▎         | 13382/400000 [00:01<09:56, 648.20it/s]  4%|▎         | 14404/400000 [00:01<07:07, 901.44it/s]  4%|▍         | 15431/400000 [00:01<05:09, 1241.05it/s]  4%|▍         | 16498/400000 [00:01<03:47, 1688.73it/s]  4%|▍         | 17549/400000 [00:01<02:49, 2257.03it/s]  5%|▍         | 18590/400000 [00:01<02:12, 2879.38it/s]  5%|▍         | 19561/400000 [00:02<01:45, 3607.85it/s]  5%|▌         | 20715/400000 [00:02<01:23, 4544.84it/s]  5%|▌         | 21809/400000 [00:02<01:08, 5511.17it/s]  6%|▌         | 22849/400000 [00:02<00:58, 6415.70it/s]  6%|▌         | 23884/400000 [00:02<00:52, 7201.07it/s]  6%|▌         | 24933/400000 [00:02<00:47, 7947.11it/s]  6%|▋         | 25967/400000 [00:02<00:44, 8422.68it/s]  7%|▋         | 26984/400000 [00:02<00:42, 8807.80it/s]  7%|▋         | 27992/400000 [00:02<00:40, 9108.61it/s]  7%|▋         | 29046/400000 [00:03<00:39, 9493.25it/s]  8%|▊         | 30150/400000 [00:03<00:37, 9908.84it/s]  8%|▊         | 31252/400000 [00:03<00:36, 10217.35it/s]  8%|▊         | 32313/400000 [00:03<00:35, 10250.77it/s]  8%|▊         | 33370/400000 [00:03<00:35, 10341.93it/s]  9%|▊         | 34426/400000 [00:03<00:35, 10405.88it/s]  9%|▉         | 35481/400000 [00:03<00:35, 10179.55it/s]  9%|▉         | 36510/400000 [00:03<00:36, 10055.58it/s]  9%|▉         | 37535/400000 [00:03<00:35, 10112.95it/s] 10%|▉         | 38552/400000 [00:03<00:35, 10126.48it/s] 10%|▉         | 39569/400000 [00:04<00:36, 9968.67it/s]  10%|█         | 40570/400000 [00:04<00:36, 9734.48it/s] 10%|█         | 41610/400000 [00:04<00:36, 9924.61it/s] 11%|█         | 42701/400000 [00:04<00:35, 10198.93it/s] 11%|█         | 43790/400000 [00:04<00:34, 10396.36it/s] 11%|█         | 44880/400000 [00:04<00:33, 10539.07it/s] 11%|█▏        | 45938/400000 [00:04<00:33, 10440.52it/s] 12%|█▏        | 46985/400000 [00:04<00:34, 10306.18it/s] 12%|█▏        | 48117/400000 [00:04<00:33, 10590.68it/s] 12%|█▏        | 49180/400000 [00:04<00:33, 10420.33it/s] 13%|█▎        | 50226/400000 [00:05<00:34, 10029.88it/s] 13%|█▎        | 51315/400000 [00:05<00:33, 10272.84it/s] 13%|█▎        | 52429/400000 [00:05<00:33, 10517.75it/s] 13%|█▎        | 53490/400000 [00:05<00:32, 10544.49it/s] 14%|█▎        | 54548/400000 [00:05<00:33, 10467.98it/s] 14%|█▍        | 55598/400000 [00:05<00:33, 10177.32it/s] 14%|█▍        | 56620/400000 [00:05<00:34, 9882.00it/s]  14%|█▍        | 57694/400000 [00:05<00:33, 10124.60it/s] 15%|█▍        | 58776/400000 [00:05<00:33, 10322.25it/s] 15%|█▍        | 59956/400000 [00:05<00:31, 10723.21it/s] 15%|█▌        | 61036/400000 [00:06<00:32, 10444.46it/s] 16%|█▌        | 62106/400000 [00:06<00:32, 10518.71it/s] 16%|█▌        | 63176/400000 [00:06<00:31, 10571.82it/s] 16%|█▌        | 64237/400000 [00:06<00:31, 10501.98it/s] 16%|█▋        | 65290/400000 [00:06<00:31, 10471.84it/s] 17%|█▋        | 66339/400000 [00:06<00:32, 10289.54it/s] 17%|█▋        | 67370/400000 [00:06<00:32, 10138.33it/s] 17%|█▋        | 68390/400000 [00:06<00:32, 10155.51it/s] 17%|█▋        | 69407/400000 [00:06<00:32, 10128.52it/s] 18%|█▊        | 70436/400000 [00:07<00:32, 10174.74it/s] 18%|█▊        | 71455/400000 [00:07<00:32, 10026.98it/s] 18%|█▊        | 72496/400000 [00:07<00:32, 10138.11it/s] 18%|█▊        | 73547/400000 [00:07<00:31, 10245.96it/s] 19%|█▊        | 74620/400000 [00:07<00:31, 10385.74it/s] 19%|█▉        | 75660/400000 [00:07<00:31, 10345.03it/s] 19%|█▉        | 76696/400000 [00:07<00:32, 10067.68it/s] 19%|█▉        | 77787/400000 [00:07<00:31, 10305.84it/s] 20%|█▉        | 78849/400000 [00:07<00:30, 10396.53it/s] 20%|█▉        | 79996/400000 [00:07<00:29, 10695.15it/s] 20%|██        | 81070/400000 [00:08<00:30, 10586.46it/s] 21%|██        | 82132/400000 [00:08<00:30, 10534.69it/s] 21%|██        | 83246/400000 [00:08<00:29, 10706.86it/s] 21%|██        | 84319/400000 [00:08<00:29, 10678.70it/s] 21%|██▏       | 85389/400000 [00:08<00:29, 10590.44it/s] 22%|██▏       | 86451/400000 [00:08<00:29, 10598.78it/s] 22%|██▏       | 87512/400000 [00:08<00:30, 10192.11it/s] 22%|██▏       | 88536/400000 [00:08<00:30, 10072.22it/s] 22%|██▏       | 89594/400000 [00:08<00:30, 10219.30it/s] 23%|██▎       | 90745/400000 [00:08<00:29, 10574.95it/s] 23%|██▎       | 91906/400000 [00:09<00:28, 10864.93it/s] 23%|██▎       | 93004/400000 [00:09<00:28, 10898.89it/s] 24%|██▎       | 94098/400000 [00:09<00:28, 10651.81it/s] 24%|██▍       | 95168/400000 [00:09<00:29, 10351.32it/s] 24%|██▍       | 96291/400000 [00:09<00:28, 10599.24it/s] 24%|██▍       | 97429/400000 [00:09<00:27, 10820.73it/s] 25%|██▍       | 98516/400000 [00:09<00:28, 10666.59it/s] 25%|██▍       | 99620/400000 [00:09<00:27, 10775.84it/s] 25%|██▌       | 100747/400000 [00:09<00:27, 10918.31it/s] 25%|██▌       | 101842/400000 [00:10<00:28, 10356.02it/s] 26%|██▌       | 102896/400000 [00:10<00:28, 10408.41it/s] 26%|██▌       | 103943/400000 [00:10<00:29, 10205.81it/s] 26%|██▌       | 104969/400000 [00:10<00:29, 10041.38it/s] 26%|██▋       | 105988/400000 [00:10<00:29, 10081.75it/s] 27%|██▋       | 107044/400000 [00:10<00:28, 10218.40it/s] 27%|██▋       | 108092/400000 [00:10<00:28, 10294.08it/s] 27%|██▋       | 109147/400000 [00:10<00:28, 10367.42it/s] 28%|██▊       | 110186/400000 [00:10<00:28, 10124.58it/s] 28%|██▊       | 111213/400000 [00:10<00:28, 10166.11it/s] 28%|██▊       | 112339/400000 [00:11<00:27, 10468.06it/s] 28%|██▊       | 113426/400000 [00:11<00:27, 10585.25it/s] 29%|██▊       | 114491/400000 [00:11<00:26, 10603.77it/s] 29%|██▉       | 115554/400000 [00:11<00:27, 10507.85it/s] 29%|██▉       | 116633/400000 [00:11<00:26, 10589.98it/s] 29%|██▉       | 117694/400000 [00:11<00:26, 10595.22it/s] 30%|██▉       | 118797/400000 [00:11<00:26, 10720.98it/s] 30%|██▉       | 119871/400000 [00:11<00:26, 10677.55it/s] 30%|███       | 120993/400000 [00:11<00:25, 10833.71it/s] 31%|███       | 122078/400000 [00:11<00:25, 10723.87it/s] 31%|███       | 123152/400000 [00:12<00:26, 10527.46it/s] 31%|███       | 124207/400000 [00:12<00:26, 10267.89it/s] 31%|███▏      | 125237/400000 [00:12<00:27, 10041.46it/s] 32%|███▏      | 126332/400000 [00:12<00:26, 10296.55it/s] 32%|███▏      | 127468/400000 [00:12<00:25, 10593.72it/s] 32%|███▏      | 128533/400000 [00:12<00:25, 10523.65it/s] 32%|███▏      | 129622/400000 [00:12<00:25, 10629.88it/s] 33%|███▎      | 130688/400000 [00:12<00:25, 10619.35it/s] 33%|███▎      | 131752/400000 [00:12<00:25, 10600.25it/s] 33%|███▎      | 132820/400000 [00:12<00:25, 10606.34it/s] 33%|███▎      | 133882/400000 [00:13<00:25, 10387.95it/s] 34%|███▎      | 134925/400000 [00:13<00:25, 10399.56it/s] 34%|███▍      | 135967/400000 [00:13<00:25, 10328.47it/s] 34%|███▍      | 137001/400000 [00:13<00:26, 10039.29it/s] 35%|███▍      | 138058/400000 [00:13<00:25, 10191.61it/s] 35%|███▍      | 139097/400000 [00:13<00:25, 10250.28it/s] 35%|███▌      | 140139/400000 [00:13<00:25, 10286.12it/s] 35%|███▌      | 141169/400000 [00:13<00:25, 10119.06it/s] 36%|███▌      | 142267/400000 [00:13<00:24, 10360.59it/s] 36%|███▌      | 143440/400000 [00:13<00:23, 10735.85it/s] 36%|███▌      | 144519/400000 [00:14<00:24, 10593.84it/s] 36%|███▋      | 145592/400000 [00:14<00:23, 10632.18it/s] 37%|███▋      | 146667/400000 [00:14<00:23, 10663.72it/s] 37%|███▋      | 147774/400000 [00:14<00:23, 10780.50it/s] 37%|███▋      | 148854/400000 [00:14<00:23, 10563.92it/s] 37%|███▋      | 149913/400000 [00:14<00:24, 10108.06it/s] 38%|███▊      | 150930/400000 [00:14<00:24, 10008.25it/s] 38%|███▊      | 151936/400000 [00:14<00:25, 9888.47it/s]  38%|███▊      | 152976/400000 [00:14<00:24, 10034.72it/s] 38%|███▊      | 153983/400000 [00:15<00:24, 9975.27it/s]  39%|███▊      | 154983/400000 [00:15<00:25, 9586.74it/s] 39%|███▉      | 155988/400000 [00:15<00:25, 9718.49it/s] 39%|███▉      | 157125/400000 [00:15<00:23, 10159.88it/s] 40%|███▉      | 158149/400000 [00:15<00:25, 9572.11it/s]  40%|███▉      | 159119/400000 [00:15<00:25, 9352.65it/s] 40%|████      | 160176/400000 [00:15<00:24, 9685.88it/s] 40%|████      | 161217/400000 [00:15<00:24, 9890.64it/s] 41%|████      | 162361/400000 [00:15<00:23, 10308.81it/s] 41%|████      | 163426/400000 [00:15<00:22, 10407.84it/s] 41%|████      | 164475/400000 [00:16<00:23, 10086.50it/s] 41%|████▏     | 165491/400000 [00:16<00:23, 10024.64it/s] 42%|████▏     | 166499/400000 [00:16<00:23, 9917.96it/s]  42%|████▏     | 167544/400000 [00:16<00:23, 10068.70it/s] 42%|████▏     | 168555/400000 [00:16<00:22, 10071.38it/s] 42%|████▏     | 169613/400000 [00:16<00:22, 10216.89it/s] 43%|████▎     | 170693/400000 [00:16<00:22, 10384.90it/s] 43%|████▎     | 171734/400000 [00:16<00:22, 10234.46it/s] 43%|████▎     | 172760/400000 [00:16<00:22, 10177.88it/s] 43%|████▎     | 173850/400000 [00:16<00:21, 10383.39it/s] 44%|████▎     | 174891/400000 [00:17<00:22, 10114.21it/s] 44%|████▍     | 175906/400000 [00:17<00:22, 10023.74it/s] 44%|████▍     | 176930/400000 [00:17<00:22, 10087.45it/s] 45%|████▍     | 178055/400000 [00:17<00:21, 10409.28it/s] 45%|████▍     | 179184/400000 [00:17<00:20, 10657.93it/s] 45%|████▌     | 180258/400000 [00:17<00:20, 10679.96it/s] 45%|████▌     | 181329/400000 [00:17<00:20, 10433.03it/s] 46%|████▌     | 182376/400000 [00:17<00:21, 10186.51it/s] 46%|████▌     | 183467/400000 [00:17<00:20, 10392.08it/s] 46%|████▌     | 184510/400000 [00:18<00:20, 10296.65it/s] 46%|████▋     | 185658/400000 [00:18<00:20, 10623.36it/s] 47%|████▋     | 186725/400000 [00:18<00:21, 10101.93it/s] 47%|████▋     | 187747/400000 [00:18<00:20, 10136.17it/s] 47%|████▋     | 188891/400000 [00:18<00:20, 10493.83it/s] 48%|████▊     | 190013/400000 [00:18<00:19, 10700.84it/s] 48%|████▊     | 191123/400000 [00:18<00:19, 10817.23it/s] 48%|████▊     | 192210/400000 [00:18<00:19, 10764.08it/s] 48%|████▊     | 193290/400000 [00:18<00:20, 10268.62it/s] 49%|████▊     | 194387/400000 [00:18<00:19, 10468.53it/s] 49%|████▉     | 195556/400000 [00:19<00:18, 10806.06it/s] 49%|████▉     | 196644/400000 [00:19<00:18, 10727.52it/s] 49%|████▉     | 197760/400000 [00:19<00:18, 10852.16it/s] 50%|████▉     | 198850/400000 [00:19<00:18, 10810.01it/s] 50%|████▉     | 199934/400000 [00:19<00:18, 10772.97it/s] 50%|█████     | 201014/400000 [00:19<00:18, 10531.49it/s] 51%|█████     | 202070/400000 [00:19<00:19, 10019.09it/s] 51%|█████     | 203079/400000 [00:19<00:20, 9836.81it/s]  51%|█████     | 204069/400000 [00:19<00:20, 9697.25it/s] 51%|█████▏    | 205092/400000 [00:20<00:19, 9849.28it/s] 52%|█████▏    | 206105/400000 [00:20<00:19, 9930.04it/s] 52%|█████▏    | 207266/400000 [00:20<00:18, 10378.99it/s] 52%|█████▏    | 208312/400000 [00:20<00:18, 10390.38it/s] 52%|█████▏    | 209357/400000 [00:20<00:18, 10207.44it/s] 53%|█████▎    | 210382/400000 [00:20<00:18, 10041.69it/s] 53%|█████▎    | 211390/400000 [00:20<00:18, 9944.42it/s]  53%|█████▎    | 212388/400000 [00:20<00:19, 9629.60it/s] 53%|█████▎    | 213362/400000 [00:20<00:19, 9659.18it/s] 54%|█████▎    | 214411/400000 [00:20<00:18, 9893.56it/s] 54%|█████▍    | 215509/400000 [00:21<00:18, 10194.72it/s] 54%|█████▍    | 216561/400000 [00:21<00:17, 10290.12it/s] 54%|█████▍    | 217693/400000 [00:21<00:17, 10578.37it/s] 55%|█████▍    | 218756/400000 [00:21<00:17, 10452.69it/s] 55%|█████▍    | 219883/400000 [00:21<00:16, 10683.96it/s] 55%|█████▌    | 221013/400000 [00:21<00:16, 10861.25it/s] 56%|█████▌    | 222103/400000 [00:21<00:16, 10636.23it/s] 56%|█████▌    | 223202/400000 [00:21<00:16, 10738.78it/s] 56%|█████▌    | 224279/400000 [00:21<00:16, 10499.12it/s] 56%|█████▋    | 225332/400000 [00:21<00:16, 10436.20it/s] 57%|█████▋    | 226464/400000 [00:22<00:16, 10684.19it/s] 57%|█████▋    | 227536/400000 [00:22<00:16, 10618.92it/s] 57%|█████▋    | 228601/400000 [00:22<00:16, 10448.49it/s] 57%|█████▋    | 229648/400000 [00:22<00:16, 10326.14it/s] 58%|█████▊    | 230750/400000 [00:22<00:16, 10522.78it/s] 58%|█████▊    | 231820/400000 [00:22<00:15, 10573.01it/s] 58%|█████▊    | 232904/400000 [00:22<00:15, 10649.27it/s] 58%|█████▊    | 233993/400000 [00:22<00:15, 10720.19it/s] 59%|█████▉    | 235067/400000 [00:22<00:15, 10474.05it/s] 59%|█████▉    | 236117/400000 [00:22<00:15, 10331.90it/s] 59%|█████▉    | 237220/400000 [00:23<00:15, 10531.44it/s] 60%|█████▉    | 238354/400000 [00:23<00:15, 10759.10it/s] 60%|█████▉    | 239433/400000 [00:23<00:15, 10594.53it/s] 60%|██████    | 240495/400000 [00:23<00:15, 10111.16it/s] 60%|██████    | 241625/400000 [00:23<00:15, 10438.84it/s] 61%|██████    | 242677/400000 [00:23<00:15, 10367.57it/s] 61%|██████    | 243771/400000 [00:23<00:14, 10530.89it/s] 61%|██████    | 244829/400000 [00:23<00:14, 10423.24it/s] 61%|██████▏   | 245875/400000 [00:23<00:15, 9907.31it/s]  62%|██████▏   | 247013/400000 [00:24<00:14, 10304.97it/s] 62%|██████▏   | 248161/400000 [00:24<00:14, 10629.48it/s] 62%|██████▏   | 249295/400000 [00:24<00:13, 10831.53it/s] 63%|██████▎   | 250400/400000 [00:24<00:13, 10895.16it/s] 63%|██████▎   | 251515/400000 [00:24<00:13, 10970.25it/s] 63%|██████▎   | 252634/400000 [00:24<00:13, 11034.26it/s] 63%|██████▎   | 253747/400000 [00:24<00:13, 11062.04it/s] 64%|██████▎   | 254856/400000 [00:24<00:13, 10991.36it/s] 64%|██████▍   | 255957/400000 [00:24<00:13, 10596.19it/s] 64%|██████▍   | 257021/400000 [00:24<00:14, 10139.30it/s] 65%|██████▍   | 258042/400000 [00:25<00:14, 10026.88it/s] 65%|██████▍   | 259050/400000 [00:25<00:14, 9845.12it/s]  65%|██████▌   | 260109/400000 [00:25<00:13, 10055.66it/s] 65%|██████▌   | 261151/400000 [00:25<00:13, 10161.56it/s] 66%|██████▌   | 262171/400000 [00:25<00:14, 9781.56it/s]  66%|██████▌   | 263155/400000 [00:25<00:14, 9696.14it/s] 66%|██████▌   | 264198/400000 [00:25<00:13, 9905.06it/s] 66%|██████▋   | 265294/400000 [00:25<00:13, 10196.89it/s] 67%|██████▋   | 266372/400000 [00:25<00:12, 10365.08it/s] 67%|██████▋   | 267433/400000 [00:25<00:12, 10435.63it/s] 67%|██████▋   | 268550/400000 [00:26<00:12, 10645.47it/s] 67%|██████▋   | 269639/400000 [00:26<00:12, 10717.07it/s] 68%|██████▊   | 270776/400000 [00:26<00:11, 10903.33it/s] 68%|██████▊   | 271930/400000 [00:26<00:11, 11084.05it/s] 68%|██████▊   | 273041/400000 [00:26<00:11, 10852.08it/s] 69%|██████▊   | 274145/400000 [00:26<00:11, 10905.47it/s] 69%|██████▉   | 275238/400000 [00:26<00:11, 10752.71it/s] 69%|██████▉   | 276330/400000 [00:26<00:11, 10801.60it/s] 69%|██████▉   | 277412/400000 [00:26<00:11, 10467.69it/s] 70%|██████▉   | 278469/400000 [00:27<00:11, 10495.62it/s] 70%|██████▉   | 279598/400000 [00:27<00:11, 10721.38it/s] 70%|███████   | 280673/400000 [00:27<00:11, 10725.82it/s] 70%|███████   | 281767/400000 [00:27<00:10, 10788.97it/s] 71%|███████   | 282928/400000 [00:27<00:10, 11021.68it/s] 71%|███████   | 284033/400000 [00:27<00:10, 10652.10it/s] 71%|███████▏  | 285189/400000 [00:27<00:10, 10909.13it/s] 72%|███████▏  | 286304/400000 [00:27<00:10, 10979.14it/s] 72%|███████▏  | 287406/400000 [00:27<00:10, 10927.55it/s] 72%|███████▏  | 288548/400000 [00:27<00:10, 11068.31it/s] 72%|███████▏  | 289657/400000 [00:28<00:10, 10720.76it/s] 73%|███████▎  | 290786/400000 [00:28<00:10, 10883.69it/s] 73%|███████▎  | 291918/400000 [00:28<00:09, 11009.84it/s] 73%|███████▎  | 293034/400000 [00:28<00:09, 11052.80it/s] 74%|███████▎  | 294142/400000 [00:28<00:10, 10543.33it/s] 74%|███████▍  | 295203/400000 [00:28<00:10, 10367.90it/s] 74%|███████▍  | 296245/400000 [00:28<00:10, 10200.36it/s] 74%|███████▍  | 297270/400000 [00:28<00:10, 9735.56it/s]  75%|███████▍  | 298445/400000 [00:28<00:09, 10261.05it/s] 75%|███████▍  | 299564/400000 [00:28<00:09, 10521.13it/s] 75%|███████▌  | 300697/400000 [00:29<00:09, 10749.29it/s] 75%|███████▌  | 301805/400000 [00:29<00:09, 10845.05it/s] 76%|███████▌  | 302946/400000 [00:29<00:08, 11006.57it/s] 76%|███████▌  | 304052/400000 [00:29<00:09, 10651.17it/s] 76%|███████▋  | 305124/400000 [00:29<00:09, 10455.60it/s] 77%|███████▋  | 306256/400000 [00:29<00:08, 10699.91it/s] 77%|███████▋  | 307377/400000 [00:29<00:08, 10846.80it/s] 77%|███████▋  | 308504/400000 [00:29<00:08, 10968.06it/s] 77%|███████▋  | 309604/400000 [00:29<00:08, 10812.94it/s] 78%|███████▊  | 310688/400000 [00:30<00:08, 10144.99it/s] 78%|███████▊  | 311713/400000 [00:30<00:08, 10037.17it/s] 78%|███████▊  | 312811/400000 [00:30<00:08, 10300.36it/s] 78%|███████▊  | 313939/400000 [00:30<00:08, 10574.90it/s] 79%|███████▉  | 315021/400000 [00:30<00:07, 10645.33it/s] 79%|███████▉  | 316091/400000 [00:30<00:08, 10445.12it/s] 79%|███████▉  | 317140/400000 [00:30<00:08, 10226.52it/s] 80%|███████▉  | 318285/400000 [00:30<00:07, 10563.36it/s] 80%|███████▉  | 319347/400000 [00:30<00:07, 10571.91it/s] 80%|████████  | 320465/400000 [00:30<00:07, 10746.25it/s] 80%|████████  | 321544/400000 [00:31<00:07, 10466.88it/s] 81%|████████  | 322595/400000 [00:31<00:07, 10229.54it/s] 81%|████████  | 323697/400000 [00:31<00:07, 10453.99it/s] 81%|████████  | 324768/400000 [00:31<00:07, 10529.03it/s] 81%|████████▏ | 325901/400000 [00:31<00:06, 10755.70it/s] 82%|████████▏ | 326980/400000 [00:31<00:06, 10437.29it/s] 82%|████████▏ | 328029/400000 [00:31<00:07, 10213.79it/s] 82%|████████▏ | 329055/400000 [00:31<00:07, 9705.76it/s]  83%|████████▎ | 330133/400000 [00:31<00:06, 10002.99it/s] 83%|████████▎ | 331216/400000 [00:32<00:06, 10236.08it/s] 83%|████████▎ | 332247/400000 [00:32<00:06, 10197.89it/s] 83%|████████▎ | 333272/400000 [00:32<00:06, 10021.02it/s] 84%|████████▎ | 334331/400000 [00:32<00:06, 10182.82it/s] 84%|████████▍ | 335353/400000 [00:32<00:06, 10134.63it/s] 84%|████████▍ | 336432/400000 [00:32<00:06, 10322.19it/s] 84%|████████▍ | 337491/400000 [00:32<00:06, 10399.17it/s] 85%|████████▍ | 338533/400000 [00:32<00:06, 9739.73it/s]  85%|████████▍ | 339517/400000 [00:32<00:06, 9758.97it/s] 85%|████████▌ | 340579/400000 [00:32<00:05, 9999.80it/s] 85%|████████▌ | 341586/400000 [00:33<00:06, 9650.56it/s] 86%|████████▌ | 342733/400000 [00:33<00:05, 10132.02it/s] 86%|████████▌ | 343863/400000 [00:33<00:05, 10455.84it/s] 86%|████████▋ | 345027/400000 [00:33<00:05, 10784.70it/s] 87%|████████▋ | 346145/400000 [00:33<00:04, 10900.15it/s] 87%|████████▋ | 347243/400000 [00:33<00:04, 10827.98it/s] 87%|████████▋ | 348331/400000 [00:33<00:04, 10736.53it/s] 87%|████████▋ | 349453/400000 [00:33<00:04, 10875.67it/s] 88%|████████▊ | 350544/400000 [00:33<00:04, 10765.46it/s] 88%|████████▊ | 351636/400000 [00:33<00:04, 10808.34it/s] 88%|████████▊ | 352719/400000 [00:34<00:04, 10793.62it/s] 88%|████████▊ | 353800/400000 [00:34<00:04, 10619.93it/s] 89%|████████▊ | 354910/400000 [00:34<00:04, 10758.73it/s] 89%|████████▉ | 355988/400000 [00:34<00:04, 10703.08it/s] 89%|████████▉ | 357060/400000 [00:34<00:04, 10524.91it/s] 90%|████████▉ | 358114/400000 [00:34<00:04, 10355.04it/s] 90%|████████▉ | 359185/400000 [00:34<00:03, 10458.01it/s] 90%|█████████ | 360295/400000 [00:34<00:03, 10640.15it/s] 90%|█████████ | 361361/400000 [00:34<00:03, 10639.90it/s] 91%|█████████ | 362427/400000 [00:34<00:03, 10607.06it/s] 91%|█████████ | 363489/400000 [00:35<00:03, 10586.79it/s] 91%|█████████ | 364549/400000 [00:35<00:03, 10430.51it/s] 91%|█████████▏| 365594/400000 [00:35<00:03, 10328.46it/s] 92%|█████████▏| 366628/400000 [00:35<00:03, 10062.41it/s] 92%|█████████▏| 367697/400000 [00:35<00:03, 10240.34it/s] 92%|█████████▏| 368737/400000 [00:35<00:03, 10286.53it/s] 92%|█████████▏| 369768/400000 [00:35<00:02, 10182.90it/s] 93%|█████████▎| 370826/400000 [00:35<00:02, 10298.74it/s] 93%|█████████▎| 371940/400000 [00:35<00:02, 10535.39it/s] 93%|█████████▎| 373025/400000 [00:36<00:02, 10626.88it/s] 94%|█████████▎| 374090/400000 [00:36<00:02, 10269.77it/s] 94%|█████████▍| 375121/400000 [00:36<00:02, 10098.16it/s] 94%|█████████▍| 376238/400000 [00:36<00:02, 10394.97it/s] 94%|█████████▍| 377365/400000 [00:36<00:02, 10642.74it/s] 95%|█████████▍| 378495/400000 [00:36<00:01, 10830.45it/s] 95%|█████████▍| 379583/400000 [00:36<00:01, 10669.87it/s] 95%|█████████▌| 380654/400000 [00:36<00:01, 10527.99it/s] 95%|█████████▌| 381781/400000 [00:36<00:01, 10739.81it/s] 96%|█████████▌| 382862/400000 [00:36<00:01, 10758.94it/s] 96%|█████████▌| 383950/400000 [00:37<00:01, 10794.02it/s] 96%|█████████▋| 385031/400000 [00:37<00:01, 10632.67it/s] 97%|█████████▋| 386182/400000 [00:37<00:01, 10880.43it/s] 97%|█████████▋| 387305/400000 [00:37<00:01, 10981.88it/s] 97%|█████████▋| 388449/400000 [00:37<00:01, 11115.38it/s] 97%|█████████▋| 389580/400000 [00:37<00:00, 11172.21it/s] 98%|█████████▊| 390699/400000 [00:37<00:00, 10637.93it/s] 98%|█████████▊| 391770/400000 [00:37<00:00, 10205.11it/s] 98%|█████████▊| 392799/400000 [00:37<00:00, 10007.09it/s] 98%|█████████▊| 393830/400000 [00:37<00:00, 10094.75it/s] 99%|█████████▊| 394953/400000 [00:38<00:00, 10410.13it/s] 99%|█████████▉| 396009/400000 [00:38<00:00, 10453.81it/s] 99%|█████████▉| 397108/400000 [00:38<00:00, 10607.24it/s]100%|█████████▉| 398173/400000 [00:38<00:00, 10507.76it/s]100%|█████████▉| 399309/400000 [00:38<00:00, 10749.01it/s]100%|█████████▉| 399999/400000 [00:38<00:00, 10378.13it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb884f95ba8>, <torchtext.data.dataset.TabularDataset object at 0x7fb884f95cf8>, <torchtext.vocab.Vocab object at 0x7fb884f95c18>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.28 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.28 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.28 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.18 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.80 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.80 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.82 file/s]2020-06-18 06:19:21.713153: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-18 06:19:21.718348: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-06-18 06:19:21.718537: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c632fe19d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-18 06:19:21.718550: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161804.06it/s] 72%|███████▏  | 7159808/9912422 [00:00<00:11, 230914.42it/s]9920512it [00:00, 43824950.59it/s]                           
0it [00:00, ?it/s]32768it [00:00, 726994.06it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1041483.38it/s]1654784it [00:00, 12098541.60it/s]                           
0it [00:00, ?it/s]8192it [00:00, 257739.28it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz
