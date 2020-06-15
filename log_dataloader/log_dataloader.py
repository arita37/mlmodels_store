
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/da8e4697df697a7fe7561af9a7f9b4cf66bf6a3a', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'da8e4697df697a7fe7561af9a7f9b4cf66bf6a3a', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/da8e4697df697a7fe7561af9a7f9b4cf66bf6a3a

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/da8e4697df697a7fe7561af9a7f9b4cf66bf6a3a

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/da8e4697df697a7fe7561af9a7f9b4cf66bf6a3a

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4333b0aea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4333b0aea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f439f0c20d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f439f0c20d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f43b8deee18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f43b8deee18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f434c93d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f434c93d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f434c93d950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3784704/9912422 [00:00<00:00, 37846083.54it/s]9920512it [00:00, 37064200.77it/s]                             
0it [00:00, ?it/s]32768it [00:00, 675126.87it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481738.23it/s]1654784it [00:00, 12212818.97it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171859.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4349c00748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4349bf99e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f434c93d598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f434c93d598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f434c93d598> , (data_info, **args) 

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
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:48,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:48,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:47,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:47,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:46,  3.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:30,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:30,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:30,  4.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:19,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:19,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:13,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:12,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:12,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:09, 11.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:09, 11.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:09, 11.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:09, 11.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 11.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 16.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 16.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 21.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 28.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 36.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 45.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 71.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 71.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 79.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:01<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.02s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.28 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.50 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ url]
0 examples [00:00, ? examples/s]2020-06-15 09:02:54.559885: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 09:02:54.571253: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-15 09:02:54.571473: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5583b1ab75d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 09:02:54.571491: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
28 examples [00:00, 279.15 examples/s]135 examples [00:00, 358.47 examples/s]245 examples [00:00, 449.29 examples/s]355 examples [00:00, 546.16 examples/s]469 examples [00:00, 646.46 examples/s]575 examples [00:00, 731.17 examples/s]685 examples [00:00, 812.02 examples/s]784 examples [00:00, 856.25 examples/s]882 examples [00:00, 886.70 examples/s]986 examples [00:01, 925.95 examples/s]1094 examples [00:01, 965.74 examples/s]1205 examples [00:01, 1003.50 examples/s]1317 examples [00:01, 1035.05 examples/s]1427 examples [00:01, 1051.03 examples/s]1536 examples [00:01, 1061.00 examples/s]1646 examples [00:01, 1072.38 examples/s]1755 examples [00:01, 1070.85 examples/s]1867 examples [00:01, 1082.15 examples/s]1978 examples [00:01, 1086.31 examples/s]2088 examples [00:02, 1072.64 examples/s]2198 examples [00:02, 1079.20 examples/s]2309 examples [00:02, 1085.55 examples/s]2418 examples [00:02, 1067.44 examples/s]2525 examples [00:02, 1019.92 examples/s]2635 examples [00:02, 1042.63 examples/s]2741 examples [00:02, 1047.57 examples/s]2851 examples [00:02, 1062.65 examples/s]2959 examples [00:02, 1064.59 examples/s]3070 examples [00:02, 1076.67 examples/s]3178 examples [00:03, 1070.47 examples/s]3293 examples [00:03, 1091.10 examples/s]3403 examples [00:03, 1092.99 examples/s]3518 examples [00:03, 1108.73 examples/s]3630 examples [00:03, 1103.92 examples/s]3741 examples [00:03, 1090.40 examples/s]3851 examples [00:03, 1089.13 examples/s]3960 examples [00:03, 1083.20 examples/s]4071 examples [00:03, 1090.46 examples/s]4181 examples [00:03, 1046.67 examples/s]4295 examples [00:04, 1071.32 examples/s]4409 examples [00:04, 1088.79 examples/s]4519 examples [00:04, 1087.61 examples/s]4629 examples [00:04, 1090.16 examples/s]4739 examples [00:04, 1082.72 examples/s]4851 examples [00:04, 1092.82 examples/s]4963 examples [00:04, 1100.63 examples/s]5074 examples [00:04, 1102.97 examples/s]5185 examples [00:04, 1084.59 examples/s]5294 examples [00:04, 1071.54 examples/s]5403 examples [00:05, 1074.57 examples/s]5511 examples [00:05, 1067.31 examples/s]5618 examples [00:05, 1053.30 examples/s]5724 examples [00:05, 1048.22 examples/s]5831 examples [00:05, 1052.55 examples/s]5937 examples [00:05, 1047.62 examples/s]6046 examples [00:05, 1058.22 examples/s]6152 examples [00:05, 1049.28 examples/s]6262 examples [00:05, 1062.99 examples/s]6369 examples [00:06, 1060.20 examples/s]6480 examples [00:06, 1073.17 examples/s]6591 examples [00:06, 1083.28 examples/s]6700 examples [00:06, 1081.32 examples/s]6809 examples [00:06, 1080.51 examples/s]6918 examples [00:06, 1082.53 examples/s]7027 examples [00:06, 1084.39 examples/s]7137 examples [00:06, 1087.29 examples/s]7249 examples [00:06, 1095.97 examples/s]7359 examples [00:06, 1033.74 examples/s]7464 examples [00:07, 966.32 examples/s] 7570 examples [00:07, 991.95 examples/s]7673 examples [00:07, 1002.21 examples/s]7775 examples [00:07, 988.17 examples/s] 7884 examples [00:07, 1014.56 examples/s]7987 examples [00:07, 1013.09 examples/s]8096 examples [00:07, 1033.44 examples/s]8204 examples [00:07, 1045.25 examples/s]8313 examples [00:07, 1057.06 examples/s]8424 examples [00:07, 1070.39 examples/s]8532 examples [00:08, 1062.00 examples/s]8640 examples [00:08, 1065.34 examples/s]8750 examples [00:08, 1072.82 examples/s]8861 examples [00:08, 1081.99 examples/s]8970 examples [00:08, 1081.50 examples/s]9079 examples [00:08, 1047.55 examples/s]9188 examples [00:08, 1057.14 examples/s]9298 examples [00:08, 1068.38 examples/s]9410 examples [00:08, 1082.20 examples/s]9519 examples [00:08, 1084.43 examples/s]9628 examples [00:09, 1081.51 examples/s]9738 examples [00:09, 1086.17 examples/s]9851 examples [00:09, 1097.98 examples/s]9968 examples [00:09, 1116.65 examples/s]10080 examples [00:09, 1061.03 examples/s]10187 examples [00:09, 1059.59 examples/s]10300 examples [00:09, 1078.84 examples/s]10410 examples [00:09, 1083.81 examples/s]10521 examples [00:09, 1091.46 examples/s]10631 examples [00:10, 1089.07 examples/s]10741 examples [00:10, 1051.93 examples/s]10850 examples [00:10, 1061.65 examples/s]10958 examples [00:10, 1064.58 examples/s]11067 examples [00:10, 1070.88 examples/s]11181 examples [00:10, 1090.51 examples/s]11291 examples [00:10, 1093.11 examples/s]11403 examples [00:10, 1098.08 examples/s]11513 examples [00:10, 1078.71 examples/s]11627 examples [00:10, 1093.87 examples/s]11737 examples [00:11, 1092.51 examples/s]11847 examples [00:11, 1087.80 examples/s]11957 examples [00:11, 1089.60 examples/s]12067 examples [00:11, 1053.84 examples/s]12174 examples [00:11, 1058.53 examples/s]12285 examples [00:11, 1072.53 examples/s]12393 examples [00:11, 1059.31 examples/s]12507 examples [00:11, 1080.64 examples/s]12624 examples [00:11, 1104.17 examples/s]12740 examples [00:11, 1119.71 examples/s]12855 examples [00:12, 1126.36 examples/s]12968 examples [00:12, 1122.51 examples/s]13081 examples [00:12, 1122.85 examples/s]13194 examples [00:12, 1104.06 examples/s]13308 examples [00:12, 1112.21 examples/s]13421 examples [00:12, 1115.69 examples/s]13533 examples [00:12, 1100.06 examples/s]13649 examples [00:12, 1114.84 examples/s]13762 examples [00:12, 1119.17 examples/s]13876 examples [00:12, 1124.09 examples/s]13989 examples [00:13, 1113.62 examples/s]14101 examples [00:13, 1063.29 examples/s]14214 examples [00:13, 1081.18 examples/s]14329 examples [00:13, 1099.39 examples/s]14444 examples [00:13, 1113.49 examples/s]14559 examples [00:13, 1121.20 examples/s]14672 examples [00:13, 1121.36 examples/s]14785 examples [00:13, 1121.18 examples/s]14898 examples [00:13, 1103.06 examples/s]15009 examples [00:14, 1098.99 examples/s]15120 examples [00:14, 1035.47 examples/s]15227 examples [00:14, 1043.58 examples/s]15334 examples [00:14, 1049.21 examples/s]15446 examples [00:14, 1068.97 examples/s]15554 examples [00:14, 1070.68 examples/s]15662 examples [00:14, 1061.78 examples/s]15771 examples [00:14, 1069.95 examples/s]15884 examples [00:14, 1086.49 examples/s]15998 examples [00:14, 1099.70 examples/s]16113 examples [00:15, 1112.99 examples/s]16225 examples [00:15, 1101.10 examples/s]16336 examples [00:15, 1099.06 examples/s]16450 examples [00:15, 1108.89 examples/s]16561 examples [00:15, 1107.89 examples/s]16672 examples [00:15, 1083.11 examples/s]16781 examples [00:15, 1084.44 examples/s]16890 examples [00:15, 1081.74 examples/s]17003 examples [00:15, 1094.39 examples/s]17113 examples [00:15, 1091.56 examples/s]17223 examples [00:16, 1088.56 examples/s]17332 examples [00:16, 1075.65 examples/s]17441 examples [00:16, 1079.28 examples/s]17551 examples [00:16, 1082.78 examples/s]17660 examples [00:16, 1081.62 examples/s]17769 examples [00:16, 1082.86 examples/s]17878 examples [00:16, 1064.23 examples/s]17987 examples [00:16, 1069.76 examples/s]18100 examples [00:16, 1086.79 examples/s]18212 examples [00:16, 1095.47 examples/s]18323 examples [00:17, 1097.65 examples/s]18435 examples [00:17, 1103.36 examples/s]18546 examples [00:17, 1079.09 examples/s]18655 examples [00:17, 1059.01 examples/s]18762 examples [00:17, 1055.57 examples/s]18874 examples [00:17, 1073.61 examples/s]18982 examples [00:17, 1062.10 examples/s]19089 examples [00:17, 1043.48 examples/s]19202 examples [00:17, 1066.45 examples/s]19314 examples [00:18, 1081.05 examples/s]19423 examples [00:18, 1080.78 examples/s]19534 examples [00:18, 1088.45 examples/s]19644 examples [00:18, 1091.47 examples/s]19754 examples [00:18, 1090.78 examples/s]19867 examples [00:18, 1100.73 examples/s]19981 examples [00:18, 1111.04 examples/s]20093 examples [00:18, 1058.42 examples/s]20206 examples [00:18, 1076.41 examples/s]20324 examples [00:18, 1104.08 examples/s]20435 examples [00:19, 1064.19 examples/s]20543 examples [00:19, 1064.75 examples/s]20650 examples [00:19, 1052.10 examples/s]20758 examples [00:19, 1058.61 examples/s]20865 examples [00:19, 1016.04 examples/s]20971 examples [00:19, 1028.07 examples/s]21075 examples [00:19, 1024.84 examples/s]21182 examples [00:19, 1037.65 examples/s]21293 examples [00:19, 1057.83 examples/s]21407 examples [00:19, 1080.25 examples/s]21516 examples [00:20, 1059.04 examples/s]21623 examples [00:20, 1055.66 examples/s]21732 examples [00:20, 1063.95 examples/s]21839 examples [00:20, 1058.63 examples/s]21946 examples [00:20, 1059.51 examples/s]22056 examples [00:20, 1069.92 examples/s]22170 examples [00:20, 1088.62 examples/s]22282 examples [00:20, 1097.53 examples/s]22392 examples [00:20, 1087.92 examples/s]22501 examples [00:20, 1081.14 examples/s]22612 examples [00:21, 1086.86 examples/s]22722 examples [00:21, 1088.74 examples/s]22835 examples [00:21, 1097.97 examples/s]22945 examples [00:21, 1086.59 examples/s]23059 examples [00:21, 1100.39 examples/s]23170 examples [00:21, 1098.81 examples/s]23283 examples [00:21, 1106.24 examples/s]23398 examples [00:21, 1118.74 examples/s]23510 examples [00:21, 1113.12 examples/s]23626 examples [00:21, 1124.10 examples/s]23739 examples [00:22, 1113.68 examples/s]23851 examples [00:22, 1111.10 examples/s]23963 examples [00:22, 1070.23 examples/s]24071 examples [00:22, 1045.36 examples/s]24185 examples [00:22, 1069.71 examples/s]24293 examples [00:22, 1057.50 examples/s]24400 examples [00:22, 1054.26 examples/s]24512 examples [00:22, 1071.54 examples/s]24622 examples [00:22, 1079.18 examples/s]24738 examples [00:23, 1100.32 examples/s]24849 examples [00:23, 1090.34 examples/s]24962 examples [00:23, 1099.08 examples/s]25073 examples [00:23, 1084.58 examples/s]25184 examples [00:23, 1089.65 examples/s]25294 examples [00:23, 1092.39 examples/s]25405 examples [00:23, 1097.08 examples/s]25515 examples [00:23, 1085.39 examples/s]25626 examples [00:23, 1090.47 examples/s]25739 examples [00:23, 1101.10 examples/s]25853 examples [00:24, 1110.70 examples/s]25965 examples [00:24, 1111.60 examples/s]26077 examples [00:24, 1098.00 examples/s]26187 examples [00:24, 1091.39 examples/s]26297 examples [00:24, 1088.53 examples/s]26406 examples [00:24, 1088.57 examples/s]26515 examples [00:24, 1019.54 examples/s]26625 examples [00:24, 1042.34 examples/s]26736 examples [00:24, 1060.77 examples/s]26847 examples [00:24, 1073.78 examples/s]26958 examples [00:25, 1084.30 examples/s]27067 examples [00:25, 1084.24 examples/s]27180 examples [00:25, 1096.09 examples/s]27290 examples [00:25, 1063.64 examples/s]27402 examples [00:25, 1077.54 examples/s]27511 examples [00:25, 1078.17 examples/s]27620 examples [00:25, 1075.42 examples/s]27733 examples [00:25, 1089.11 examples/s]27844 examples [00:25, 1093.77 examples/s]27954 examples [00:25, 1091.51 examples/s]28064 examples [00:26, 1052.35 examples/s]28176 examples [00:26, 1071.06 examples/s]28286 examples [00:26, 1078.64 examples/s]28395 examples [00:26, 1073.63 examples/s]28508 examples [00:26, 1089.71 examples/s]28618 examples [00:26, 1091.19 examples/s]28730 examples [00:26, 1097.44 examples/s]28845 examples [00:26, 1110.30 examples/s]28957 examples [00:26, 1107.80 examples/s]29069 examples [00:27, 1111.40 examples/s]29185 examples [00:27, 1124.15 examples/s]29298 examples [00:27, 1124.36 examples/s]29411 examples [00:27, 1124.28 examples/s]29524 examples [00:27, 1118.00 examples/s]29636 examples [00:27, 1106.28 examples/s]29747 examples [00:27, 1105.20 examples/s]29858 examples [00:27, 1088.41 examples/s]29971 examples [00:27, 1099.83 examples/s]30082 examples [00:27, 1046.57 examples/s]30195 examples [00:28, 1069.93 examples/s]30304 examples [00:28, 1075.07 examples/s]30412 examples [00:28, 1060.03 examples/s]30522 examples [00:28, 1067.11 examples/s]30629 examples [00:28, 1040.74 examples/s]30736 examples [00:28, 1046.03 examples/s]30843 examples [00:28, 1051.11 examples/s]30954 examples [00:28, 1067.08 examples/s]31061 examples [00:28, 1061.04 examples/s]31170 examples [00:28, 1069.48 examples/s]31282 examples [00:29, 1081.32 examples/s]31391 examples [00:29, 1036.61 examples/s]31502 examples [00:29, 1055.29 examples/s]31608 examples [00:29, 1055.11 examples/s]31714 examples [00:29, 1053.10 examples/s]31820 examples [00:29, 1048.47 examples/s]31925 examples [00:29, 1021.67 examples/s]32032 examples [00:29, 1034.18 examples/s]32139 examples [00:29, 1042.31 examples/s]32245 examples [00:29, 1047.36 examples/s]32350 examples [00:30, 1020.23 examples/s]32461 examples [00:30, 1044.08 examples/s]32572 examples [00:30, 1062.94 examples/s]32683 examples [00:30, 1075.44 examples/s]32791 examples [00:30, 1070.47 examples/s]32901 examples [00:30, 1078.04 examples/s]33009 examples [00:30, 1065.07 examples/s]33120 examples [00:30, 1078.10 examples/s]33232 examples [00:30, 1084.81 examples/s]33341 examples [00:31, 1058.74 examples/s]33448 examples [00:31, 1061.02 examples/s]33555 examples [00:31, 1050.92 examples/s]33663 examples [00:31, 1058.83 examples/s]33769 examples [00:31, 1055.77 examples/s]33875 examples [00:31, 1021.98 examples/s]33982 examples [00:31, 1028.90 examples/s]34088 examples [00:31, 1036.94 examples/s]34192 examples [00:31, 1025.42 examples/s]34304 examples [00:31, 1051.86 examples/s]34414 examples [00:32, 1063.76 examples/s]34527 examples [00:32, 1080.55 examples/s]34636 examples [00:32, 1057.24 examples/s]34742 examples [00:32, 1044.85 examples/s]34847 examples [00:32, 1030.99 examples/s]34953 examples [00:32, 1038.74 examples/s]35065 examples [00:32, 1060.69 examples/s]35175 examples [00:32, 1069.87 examples/s]35283 examples [00:32, 1049.15 examples/s]35389 examples [00:32, 1048.56 examples/s]35499 examples [00:33, 1061.00 examples/s]35611 examples [00:33, 1076.56 examples/s]35719 examples [00:33, 1074.64 examples/s]35834 examples [00:33, 1096.15 examples/s]35949 examples [00:33, 1110.48 examples/s]36061 examples [00:33, 1099.98 examples/s]36172 examples [00:33, 1095.71 examples/s]36284 examples [00:33, 1101.84 examples/s]36400 examples [00:33, 1116.21 examples/s]36512 examples [00:33, 1108.00 examples/s]36623 examples [00:34, 1108.55 examples/s]36734 examples [00:34, 1095.61 examples/s]36849 examples [00:34, 1109.99 examples/s]36964 examples [00:34, 1119.65 examples/s]37077 examples [00:34, 1073.20 examples/s]37185 examples [00:34, 1041.78 examples/s]37293 examples [00:34, 1050.71 examples/s]37404 examples [00:34, 1066.93 examples/s]37518 examples [00:34, 1087.74 examples/s]37628 examples [00:35, 1062.78 examples/s]37737 examples [00:35, 1069.22 examples/s]37845 examples [00:35, 1055.25 examples/s]37955 examples [00:35, 1066.29 examples/s]38069 examples [00:35, 1087.37 examples/s]38180 examples [00:35, 1091.26 examples/s]38290 examples [00:35, 1080.35 examples/s]38402 examples [00:35, 1089.47 examples/s]38513 examples [00:35, 1095.01 examples/s]38626 examples [00:35, 1103.12 examples/s]38737 examples [00:36, 1099.27 examples/s]38847 examples [00:36, 1098.53 examples/s]38957 examples [00:36, 1082.15 examples/s]39066 examples [00:36, 1072.63 examples/s]39176 examples [00:36, 1079.65 examples/s]39289 examples [00:36, 1091.27 examples/s]39399 examples [00:36, 1080.08 examples/s]39509 examples [00:36, 1084.91 examples/s]39623 examples [00:36, 1100.00 examples/s]39739 examples [00:36, 1115.48 examples/s]39851 examples [00:37, 1115.38 examples/s]39963 examples [00:37, 1112.62 examples/s]40075 examples [00:37, 1056.27 examples/s]40185 examples [00:37, 1068.89 examples/s]40298 examples [00:37, 1084.89 examples/s]40409 examples [00:37, 1091.07 examples/s]40519 examples [00:37, 1021.04 examples/s]40629 examples [00:37, 1043.05 examples/s]40735 examples [00:37, 1040.02 examples/s]40840 examples [00:38, 1038.35 examples/s]40945 examples [00:38, 1039.96 examples/s]41056 examples [00:38, 1059.07 examples/s]41167 examples [00:38, 1072.59 examples/s]41275 examples [00:38, 1047.48 examples/s]41384 examples [00:38, 1059.08 examples/s]41496 examples [00:38, 1074.00 examples/s]41607 examples [00:38, 1082.07 examples/s]41719 examples [00:38, 1090.14 examples/s]41829 examples [00:38, 1093.02 examples/s]41939 examples [00:39, 1093.18 examples/s]42050 examples [00:39, 1097.67 examples/s]42160 examples [00:39, 1089.49 examples/s]42269 examples [00:39, 1084.30 examples/s]42383 examples [00:39, 1098.08 examples/s]42493 examples [00:39, 1096.14 examples/s]42603 examples [00:39, 1091.98 examples/s]42713 examples [00:39, 1077.21 examples/s]42824 examples [00:39, 1084.19 examples/s]42938 examples [00:39, 1099.25 examples/s]43053 examples [00:40, 1113.11 examples/s]43165 examples [00:40, 1101.06 examples/s]43276 examples [00:40, 1059.17 examples/s]43385 examples [00:40, 1065.61 examples/s]43499 examples [00:40, 1085.89 examples/s]43608 examples [00:40, 1081.29 examples/s]43717 examples [00:40, 1079.02 examples/s]43826 examples [00:40, 1038.05 examples/s]43935 examples [00:40, 1053.01 examples/s]44043 examples [00:40, 1060.70 examples/s]44159 examples [00:41, 1086.09 examples/s]44268 examples [00:41, 1045.69 examples/s]44379 examples [00:41, 1062.37 examples/s]44489 examples [00:41, 1071.77 examples/s]44597 examples [00:41, 1073.28 examples/s]44708 examples [00:41, 1081.58 examples/s]44817 examples [00:41, 1072.39 examples/s]44927 examples [00:41, 1078.92 examples/s]45036 examples [00:41, 1039.93 examples/s]45141 examples [00:42, 1037.36 examples/s]45252 examples [00:42, 1057.89 examples/s]45360 examples [00:42, 1061.92 examples/s]45473 examples [00:42, 1081.28 examples/s]45585 examples [00:42, 1091.61 examples/s]45696 examples [00:42, 1096.97 examples/s]45811 examples [00:42, 1111.24 examples/s]45923 examples [00:42, 1109.59 examples/s]46035 examples [00:42, 1090.28 examples/s]46145 examples [00:42, 1085.07 examples/s]46254 examples [00:43, 1083.14 examples/s]46363 examples [00:43, 1068.26 examples/s]46470 examples [00:43, 1047.04 examples/s]46583 examples [00:43, 1070.52 examples/s]46697 examples [00:43, 1088.06 examples/s]46808 examples [00:43, 1091.69 examples/s]46919 examples [00:43, 1094.25 examples/s]47029 examples [00:43, 1064.48 examples/s]47136 examples [00:43, 1029.82 examples/s]47246 examples [00:43, 1049.23 examples/s]47362 examples [00:44, 1079.47 examples/s]47477 examples [00:44, 1097.55 examples/s]47591 examples [00:44, 1108.29 examples/s]47707 examples [00:44, 1121.00 examples/s]47820 examples [00:44, 1117.16 examples/s]47932 examples [00:44, 1080.92 examples/s]48049 examples [00:44, 1103.61 examples/s]48160 examples [00:44, 1095.07 examples/s]48274 examples [00:44, 1106.03 examples/s]48385 examples [00:44, 1103.77 examples/s]48496 examples [00:45, 1096.57 examples/s]48606 examples [00:45, 1083.25 examples/s]48715 examples [00:45, 1069.88 examples/s]48823 examples [00:45, 1068.78 examples/s]48930 examples [00:45, 1031.74 examples/s]49041 examples [00:45, 1053.52 examples/s]49155 examples [00:45, 1078.00 examples/s]49264 examples [00:45, 1080.77 examples/s]49380 examples [00:45, 1101.62 examples/s]49493 examples [00:46, 1108.34 examples/s]49605 examples [00:46, 1100.01 examples/s]49716 examples [00:46, 1084.04 examples/s]49826 examples [00:46, 1087.23 examples/s]49935 examples [00:46, 1085.93 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 8076/50000 [00:00<00:00, 80759.31 examples/s] 44%|████▍     | 21989/50000 [00:00<00:00, 92383.89 examples/s] 70%|███████   | 35140/50000 [00:00<00:00, 101437.55 examples/s] 97%|█████████▋| 48682/50000 [00:00<00:00, 109695.41 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 892.58 examples/s]204 examples [00:00, 954.27 examples/s]320 examples [00:00, 1006.24 examples/s]418 examples [00:00, 995.94 examples/s] 528 examples [00:00, 1022.40 examples/s]644 examples [00:00, 1058.05 examples/s]758 examples [00:00, 1079.80 examples/s]874 examples [00:00, 1100.70 examples/s]980 examples [00:00, 1066.20 examples/s]1090 examples [00:01, 1074.32 examples/s]1204 examples [00:01, 1090.77 examples/s]1317 examples [00:01, 1100.45 examples/s]1428 examples [00:01, 1102.47 examples/s]1541 examples [00:01, 1109.21 examples/s]1655 examples [00:01, 1116.83 examples/s]1768 examples [00:01, 1120.48 examples/s]1880 examples [00:01, 1116.99 examples/s]1992 examples [00:01, 1110.30 examples/s]2103 examples [00:01, 1101.63 examples/s]2214 examples [00:02, 1102.66 examples/s]2326 examples [00:02, 1105.60 examples/s]2437 examples [00:02, 1095.55 examples/s]2550 examples [00:02, 1104.80 examples/s]2661 examples [00:02, 1102.00 examples/s]2772 examples [00:02, 1102.70 examples/s]2883 examples [00:02, 1104.31 examples/s]2994 examples [00:02, 1097.22 examples/s]3105 examples [00:02, 1090.03 examples/s]3215 examples [00:02, 1078.09 examples/s]3323 examples [00:03, 1057.72 examples/s]3438 examples [00:03, 1083.26 examples/s]3552 examples [00:03, 1099.26 examples/s]3667 examples [00:03, 1113.28 examples/s]3779 examples [00:03, 1107.35 examples/s]3891 examples [00:03, 1110.42 examples/s]4003 examples [00:03, 1112.18 examples/s]4116 examples [00:03, 1115.90 examples/s]4228 examples [00:03, 1111.97 examples/s]4340 examples [00:03, 1100.50 examples/s]4453 examples [00:04, 1107.92 examples/s]4567 examples [00:04, 1117.04 examples/s]4682 examples [00:04, 1124.16 examples/s]4795 examples [00:04, 1115.98 examples/s]4907 examples [00:04, 1111.74 examples/s]5019 examples [00:04, 1105.44 examples/s]5133 examples [00:04, 1115.53 examples/s]5246 examples [00:04, 1116.88 examples/s]5361 examples [00:04, 1124.83 examples/s]5474 examples [00:04, 1126.28 examples/s]5587 examples [00:05, 1122.93 examples/s]5703 examples [00:05, 1131.07 examples/s]5817 examples [00:05, 1124.58 examples/s]5930 examples [00:05, 1121.98 examples/s]6043 examples [00:05, 1106.50 examples/s]6156 examples [00:05, 1111.24 examples/s]6269 examples [00:05, 1113.87 examples/s]6383 examples [00:05, 1120.49 examples/s]6496 examples [00:05, 1121.04 examples/s]6609 examples [00:05, 1101.48 examples/s]6720 examples [00:06, 1074.70 examples/s]6834 examples [00:06, 1092.16 examples/s]6944 examples [00:06, 1051.12 examples/s]7050 examples [00:06, 1039.74 examples/s]7161 examples [00:06, 1057.57 examples/s]7275 examples [00:06, 1079.63 examples/s]7390 examples [00:06, 1097.49 examples/s]7504 examples [00:06, 1109.13 examples/s]7616 examples [00:06, 1080.73 examples/s]7725 examples [00:07, 1076.70 examples/s]7836 examples [00:07, 1085.89 examples/s]7945 examples [00:07, 1077.96 examples/s]8060 examples [00:07, 1096.72 examples/s]8174 examples [00:07, 1107.00 examples/s]8285 examples [00:07, 1104.59 examples/s]8396 examples [00:07, 1091.22 examples/s]8508 examples [00:07, 1097.26 examples/s]8625 examples [00:07, 1115.63 examples/s]8737 examples [00:07, 1103.08 examples/s]8848 examples [00:08, 1096.47 examples/s]8962 examples [00:08, 1108.09 examples/s]9076 examples [00:08, 1116.29 examples/s]9188 examples [00:08, 1078.22 examples/s]9297 examples [00:08, 1068.88 examples/s]9409 examples [00:08, 1081.38 examples/s]9521 examples [00:08, 1090.87 examples/s]9635 examples [00:08, 1104.51 examples/s]9746 examples [00:08, 1094.94 examples/s]9856 examples [00:08, 1087.39 examples/s]9965 examples [00:09, 1064.68 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSAHBSY/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSAHBSY/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f434c93d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f434c93d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f434c93d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42d1de4be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f434c92b080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f434c93d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f434c93d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f434c93d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f439832a940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4349bf96d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f42c43298c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f42c43298c8> 

  function with postional parmater data_info <function split_train_valid at 0x7f42c43298c8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f42c43299d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f42c43299d8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f42c43299d8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5f1686d7e2b6fa2061fe8ea71da7ab83929caa521c906b0f8178e82ab0a562bf
  Stored in directory: /tmp/pip-ephem-wheel-cache-ntmjcqlt/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<48:57:26, 4.89kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<34:30:15, 6.94kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<24:12:15, 9.89kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:56:38, 14.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:49:35, 20.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.64M/862M [00:02<8:13:18, 28.8kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:02<5:44:13, 41.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:02<3:59:27, 58.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:02<2:46:36, 83.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:02<1:55:54, 120kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.5M/862M [00:03<1:21:35, 170kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:03<56:46, 242kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:03<39:42, 345kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.5M/862M [00:03<27:43, 491kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:03<19:25, 697kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:04<14:31, 929kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<12:01, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:06<10:47, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:06<08:01, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:47, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:08<12:12, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:08<10:11, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:08<07:29, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:10<07:55, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<08:24, 1.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:10<06:29, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<04:43, 2.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:12<09:20, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:12<07:55, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:12<05:52, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:14<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:14<06:09, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:14<04:33, 2.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<03:22, 3.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<51:41, 253kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:16<37:30, 349kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:16<26:32, 492kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<21:36, 602kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:18<16:27, 790kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:18<11:49, 1.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<11:19, 1.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:20<09:14, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:20<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<07:47, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:22<06:46, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:22<05:03, 2.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<06:33, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:23<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:24<04:27, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<06:07, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:25<05:22, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:26<04:06, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<03:01, 4.19MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<53:07, 238kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<39:48, 318kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<28:24, 445kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<19:57, 632kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<23:51, 528kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<18:01, 699kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<12:54, 973kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<11:57, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<09:39, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<07:04, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<07:52, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<08:04, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:16, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:35<06:23, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:44, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<04:20, 2.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:54, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<04:05, 3.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:45, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:17, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:00, 3.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:41, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:29, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:03, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:41, 3.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<09:20, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<07:48, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:45, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:51, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:06, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<04:35, 2.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:53, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<06:41, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:14, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<03:50, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:26, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:31, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<04:49, 2.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:01, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:45, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:23, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:55, 3.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<20:42, 569kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<15:46, 746kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<11:20, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<10:31, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<10:00, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<07:38, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:27, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<18:21, 634kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<14:05, 825kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<10:07, 1.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<09:37, 1.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<08:00, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<05:51, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:01<06:39, 1.72MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<07:07, 1.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<05:30, 2.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:02, 2.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<06:41, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<04:28, 2.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<05:39, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<06:23, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<05:06, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:41, 3.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<16:22, 689kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:07<12:41, 888kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:07, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<08:52, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<08:43, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<07:26, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:30, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<06:43, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<08:30, 1.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<05:20, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<03:51, 2.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<16:04, 689kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<12:28, 887kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<09:01, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<08:42, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<08:33, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:15<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:49, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<03:34, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<14:49, 738kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<12:19, 887kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<09:04, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:26, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<1:24:17, 129kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<1:00:16, 180kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<42:26, 255kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<31:49, 339kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<24:41, 437kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<17:52, 603kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:36, 851kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<21:46, 493kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<16:23, 654kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<11:42, 914kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<10:32, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<08:19, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:06, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:24, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<14:54, 710kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<12:49, 826kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<09:27, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:45, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<08:36, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<07:08, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:13, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<05:58, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<06:31, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<05:08, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<03:42, 2.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<08:34, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<07:07, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:12, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<05:55, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<06:27, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<04:59, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<03:41, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:37<05:25, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<04:42, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<03:34, 2.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<02:38, 3.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<13:10, 772kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<11:23, 892kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<08:27, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:02, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<08:51, 1.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<07:18, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:22, 1.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<05:59, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<05:17, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:58, 2.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<04:59, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<05:39, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<04:29, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:15, 3.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<09:44, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<07:54, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:47, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<06:13, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<06:28, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<05:03, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:39, 2.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<31:44, 307kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<23:15, 419kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<16:27, 591kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<13:39, 709kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:53<11:45, 824kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<08:45, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:13, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<19:25, 495kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<14:37, 658kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<10:27, 916kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:57<09:26, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:57<08:39, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<06:30, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:42, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<06:13, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<05:22, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:00, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<04:57, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<05:34, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:10, 2.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:03<05:57, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:03<05:09, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<03:50, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:05<04:48, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:05<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<03:15, 2.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<05:06, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<04:00, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<02:55, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<05:54, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<05:06, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<03:49, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<04:41, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<05:14, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<04:05, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<02:59, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<05:22, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:13<04:44, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<03:33, 2.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<04:28, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<05:02, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<03:57, 2.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<02:52, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<07:05, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<05:56, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:23, 2.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<05:01, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<05:25, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:13, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:05, 2.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<05:13, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<04:37, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:27, 2.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<04:20, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<04:53, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<03:53, 2.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:48, 3.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<14:32, 590kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<11:07, 771kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<07:57, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<07:27, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<07:07, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<05:23, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:52, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<06:16, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:29<05:18, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<04:35, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<04:07, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:04, 2.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<04:00, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:35, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<02:35, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<45:01, 183kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<32:21, 254kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<22:46, 360kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<17:45, 460kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<13:19, 613kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<09:28, 858kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<08:24, 963kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<06:36, 1.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<04:47, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:27, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<12:43, 631kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<09:37, 834kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<06:54, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:55, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<13:50, 575kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<11:18, 703kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<08:19, 954kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:53, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<1:08:22, 115kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<48:40, 162kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<34:08, 230kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<25:32, 306kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<19:37, 398kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<14:05, 554kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<09:54, 784kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<11:37, 667kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<08:57, 864kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<06:27, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<06:12, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<06:59, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<05:28, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<03:57, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:17, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<04:49, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<03:39, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:55, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:47, 1.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<02:54, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:26, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<02:30, 2.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<03:16, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<03:57, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<03:10, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<02:17, 3.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<08:11, 896kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<06:30, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:44, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<04:58, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<04:15, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:10, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<03:51, 1.87MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:29, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<02:37, 2.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:25, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:01, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<02:15, 3.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<01:40, 4.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<16:56, 417kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<12:36, 560kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<08:58, 783kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<07:51, 889kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<06:15, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:33, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<04:43, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<04:47, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:40, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:42, 2.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<03:40, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<02:29, 2.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<03:15, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:00, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:16, 2.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:05, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:36, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<02:50, 2.36MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:04, 3.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<04:10, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:39, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:43, 2.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<03:21, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<03:42, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:53, 2.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:06, 3.09MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<04:02, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<04:02, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:05, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:56, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<03:10, 2.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:17, 2.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<04:46, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<04:01, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:58, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:28, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:47, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<02:59, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:09, 2.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<11:17, 552kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<08:26, 737kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<06:01, 1.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<16:15, 379kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<11:56, 515kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<08:41, 707kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<06:12, 986kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<05:52, 1.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<04:46, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:29, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:46, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:56, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:02, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:12, 2.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<03:53, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:22, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:29, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<03:02, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<03:23, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:37, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<01:55, 3.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:22, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:59, 1.94MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:14, 2.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:50, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:13, 1.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<02:33, 2.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<01:51, 3.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<10:12, 556kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<07:44, 731kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:48, 975kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:05, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<1:23:02, 67.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<58:40, 95.4kB/s]  .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<41:01, 136kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<29:47, 186kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<22:02, 251kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<15:38, 353kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<10:57, 500kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<10:03, 544kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<07:36, 717kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:27, 997kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<04:59, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:56<04:42, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<03:32, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:32, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<03:19, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:53, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<03:11, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<01:48, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<03:14, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<02:49, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:06, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:36, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:23, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<01:46, 2.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:46, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<02:13, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:36, 3.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<08:34, 581kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:08<06:32, 761kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<04:41, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<04:21, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<04:08, 1.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:10, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:15, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<09:09, 528kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<06:55, 697kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<04:57, 970kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<04:30, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<04:13, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<03:10, 1.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:16, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<03:20, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:05, 2.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:31, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<02:45, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:32, 2.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:50, 1.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<03:10, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:19, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:38, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:22<02:51, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:14, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:36, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<06:38, 667kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<05:06, 866kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<03:40, 1.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:32, 1.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:25, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:37, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:52, 2.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<07:51, 545kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<05:57, 718kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<04:15, 998kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<03:55, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<03:36, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<02:45, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:57, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<11:19, 366kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<08:21, 496kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<05:55, 695kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<05:01, 812kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<04:25, 920kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<03:18, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:20, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<07:38, 525kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<05:32, 721kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:58, 1.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:41, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:00, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:11, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:22, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:04, 1.86MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:32, 2.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<01:57, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:10, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:43, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:14, 3.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<06:44, 554kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<05:07, 729kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:38, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<03:20, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:45<03:07, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:21, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:41, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<02:55, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:26, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:46, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<02:00, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<02:08, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:39, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:11, 2.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<02:59, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<02:27, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:47, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<02:00, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<02:09, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:39, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:12, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<01:57, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:42, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:16, 2.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:37, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:50, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:26, 2.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:02, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<02:22, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:00, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:28, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:43, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:50, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:25, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:01, 2.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:01, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:44, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:33, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:43, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:21, 2.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:58, 3.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<02:22, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:58, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:26, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:37, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:43, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:21, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:57, 2.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<15:10, 182kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<10:53, 253kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<07:37, 359kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<05:53, 458kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<04:41, 574kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<03:23, 791kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:24, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:25, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:58, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:25, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:33, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:38, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:16, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:54, 2.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<03:44, 665kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:53, 860kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:04, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:57, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<02:01, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:41, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:17, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:55, 2.57MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<02:11, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:46, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:17, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:24, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:13, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<00:53, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<01:07, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:45, 2.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:09, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:54, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:39, 3.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:00, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:45, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<00:57, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:06, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:51, 2.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:37, 3.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<00:55, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:40, 2.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<00:52, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:48, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:36, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<00:50, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:44, 2.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:33, 3.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<00:47, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<00:44, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:33, 3.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:43<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:43<00:52, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:41, 2.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:29, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:12, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<01:00, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:44, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:50, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<00:45, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:33, 2.69MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:42, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<00:48, 1.78MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<00:38, 2.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:27, 3.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<00:57, 1.45MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:48, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:35, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:41, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:36, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:27, 2.84MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:19, 3.85MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:36, 770kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<01:24, 881kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<01:02, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:43, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:58, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:48, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:34, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:38, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<00:40, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:31, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:22, 2.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:53, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:44, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:31, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:34, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:22, 2.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<00:26, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:30, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:23, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:16, 3.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:36, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:30, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:21, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:27, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:21, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:14, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<01:11, 580kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:53, 761kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:37, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:33, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:26, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:18, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:19, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:17, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:12, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:14, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:12, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:08, 3.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:12, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:08, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:10, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:05, 3.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:29, 554kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:21, 731kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:14, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:11, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:08, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:05, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:01, 2.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:02, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.35MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<143:15:33,  1.29s/it]  0%|          | 787/400000 [00:01<100:05:19,  1.11it/s]  0%|          | 1581/400000 [00:01<69:55:37,  1.58it/s]  1%|          | 2419/400000 [00:01<48:50:59,  2.26it/s]  1%|          | 3249/400000 [00:01<34:07:39,  3.23it/s]  1%|          | 4094/400000 [00:01<23:50:32,  4.61it/s]  1%|          | 4927/400000 [00:01<16:39:30,  6.59it/s]  1%|▏         | 5780/400000 [00:01<11:38:22,  9.41it/s]  2%|▏         | 6629/400000 [00:02<8:08:02, 13.43it/s]   2%|▏         | 7426/400000 [00:02<5:41:10, 19.18it/s]  2%|▏         | 8270/400000 [00:02<3:58:32, 27.37it/s]  2%|▏         | 9116/400000 [00:02<2:46:51, 39.04it/s]  2%|▏         | 9963/400000 [00:02<1:56:46, 55.67it/s]  3%|▎         | 10803/400000 [00:02<1:21:47, 79.30it/s]  3%|▎         | 11635/400000 [00:02<57:22, 112.82it/s]   3%|▎         | 12476/400000 [00:02<40:18, 160.24it/s]  3%|▎         | 13314/400000 [00:02<28:23, 227.06it/s]  4%|▎         | 14154/400000 [00:02<20:03, 320.65it/s]  4%|▎         | 14989/400000 [00:03<14:14, 450.56it/s]  4%|▍         | 15821/400000 [00:03<10:10, 629.02it/s]  4%|▍         | 16652/400000 [00:03<07:21, 868.79it/s]  4%|▍         | 17517/400000 [00:03<05:21, 1189.90it/s]  5%|▍         | 18395/400000 [00:03<03:57, 1606.46it/s]  5%|▍         | 19241/400000 [00:03<02:59, 2117.65it/s]  5%|▌         | 20083/400000 [00:03<02:19, 2730.76it/s]  5%|▌         | 20951/400000 [00:03<01:50, 3437.23it/s]  5%|▌         | 21799/400000 [00:03<01:30, 4172.57it/s]  6%|▌         | 22643/400000 [00:04<01:17, 4899.23it/s]  6%|▌         | 23481/400000 [00:04<01:07, 5559.82it/s]  6%|▌         | 24347/400000 [00:04<01:00, 6228.74it/s]  6%|▋         | 25188/400000 [00:04<00:55, 6725.89it/s]  7%|▋         | 26054/400000 [00:04<00:51, 7207.95it/s]  7%|▋         | 26923/400000 [00:04<00:49, 7596.09it/s]  7%|▋         | 27832/400000 [00:04<00:46, 7989.26it/s]  7%|▋         | 28702/400000 [00:04<00:45, 8161.05it/s]  7%|▋         | 29569/400000 [00:04<00:44, 8305.95it/s]  8%|▊         | 30436/400000 [00:04<00:44, 8310.25it/s]  8%|▊         | 31327/400000 [00:05<00:43, 8478.73it/s]  8%|▊         | 32242/400000 [00:05<00:42, 8669.37it/s]  8%|▊         | 33123/400000 [00:05<00:42, 8651.73it/s]  8%|▊         | 33998/400000 [00:05<00:43, 8401.84it/s]  9%|▊         | 34924/400000 [00:05<00:42, 8640.84it/s]  9%|▉         | 35841/400000 [00:05<00:41, 8793.00it/s]  9%|▉         | 36764/400000 [00:05<00:40, 8917.18it/s]  9%|▉         | 37661/400000 [00:05<00:41, 8788.00it/s] 10%|▉         | 38551/400000 [00:05<00:40, 8818.94it/s] 10%|▉         | 39455/400000 [00:05<00:40, 8881.59it/s] 10%|█         | 40346/400000 [00:06<00:40, 8840.31it/s] 10%|█         | 41232/400000 [00:06<00:40, 8822.44it/s] 11%|█         | 42116/400000 [00:06<00:40, 8787.95it/s] 11%|█         | 43002/400000 [00:06<00:40, 8806.83it/s] 11%|█         | 43911/400000 [00:06<00:40, 8887.38it/s] 11%|█         | 44810/400000 [00:06<00:39, 8917.89it/s] 11%|█▏        | 45715/400000 [00:06<00:39, 8955.20it/s] 12%|█▏        | 46611/400000 [00:06<00:39, 8902.49it/s] 12%|█▏        | 47502/400000 [00:06<00:39, 8883.32it/s] 12%|█▏        | 48391/400000 [00:06<00:40, 8724.77it/s] 12%|█▏        | 49265/400000 [00:07<00:40, 8681.95it/s] 13%|█▎        | 50153/400000 [00:07<00:40, 8740.07it/s] 13%|█▎        | 51028/400000 [00:07<00:40, 8701.99it/s] 13%|█▎        | 51899/400000 [00:07<00:40, 8647.23it/s] 13%|█▎        | 52771/400000 [00:07<00:40, 8668.07it/s] 13%|█▎        | 53657/400000 [00:07<00:39, 8723.15it/s] 14%|█▎        | 54555/400000 [00:07<00:39, 8796.33it/s] 14%|█▍        | 55435/400000 [00:07<00:39, 8745.41it/s] 14%|█▍        | 56320/400000 [00:07<00:39, 8775.47it/s] 14%|█▍        | 57198/400000 [00:07<00:39, 8750.76it/s] 15%|█▍        | 58074/400000 [00:08<00:39, 8732.00it/s] 15%|█▍        | 58948/400000 [00:08<00:39, 8644.51it/s] 15%|█▍        | 59813/400000 [00:08<00:40, 8484.69it/s] 15%|█▌        | 60699/400000 [00:08<00:39, 8592.05it/s] 15%|█▌        | 61573/400000 [00:08<00:39, 8634.95it/s] 16%|█▌        | 62467/400000 [00:08<00:38, 8723.62it/s] 16%|█▌        | 63342/400000 [00:08<00:38, 8729.57it/s] 16%|█▌        | 64216/400000 [00:08<00:38, 8714.36it/s] 16%|█▋        | 65109/400000 [00:08<00:38, 8777.58it/s] 17%|█▋        | 66008/400000 [00:08<00:37, 8838.80it/s] 17%|█▋        | 66920/400000 [00:09<00:37, 8918.44it/s] 17%|█▋        | 67813/400000 [00:09<00:37, 8878.54it/s] 17%|█▋        | 68702/400000 [00:09<00:38, 8698.87it/s] 17%|█▋        | 69573/400000 [00:09<00:38, 8521.39it/s] 18%|█▊        | 70475/400000 [00:09<00:38, 8662.39it/s] 18%|█▊        | 71343/400000 [00:09<00:38, 8601.59it/s] 18%|█▊        | 72243/400000 [00:09<00:37, 8715.94it/s] 18%|█▊        | 73116/400000 [00:09<00:37, 8709.90it/s] 18%|█▊        | 73988/400000 [00:09<00:37, 8643.74it/s] 19%|█▊        | 74890/400000 [00:09<00:37, 8750.42it/s] 19%|█▉        | 75766/400000 [00:10<00:37, 8652.02it/s] 19%|█▉        | 76633/400000 [00:10<00:37, 8577.79it/s] 19%|█▉        | 77492/400000 [00:10<00:38, 8476.25it/s] 20%|█▉        | 78341/400000 [00:10<00:38, 8444.95it/s] 20%|█▉        | 79200/400000 [00:10<00:37, 8486.60it/s] 20%|██        | 80050/400000 [00:10<00:37, 8485.39it/s] 20%|██        | 80933/400000 [00:10<00:37, 8584.13it/s] 20%|██        | 81792/400000 [00:10<00:37, 8567.93it/s] 21%|██        | 82673/400000 [00:10<00:36, 8639.02it/s] 21%|██        | 83538/400000 [00:11<00:36, 8628.14it/s] 21%|██        | 84402/400000 [00:11<00:36, 8596.74it/s] 21%|██▏       | 85291/400000 [00:11<00:36, 8682.59it/s] 22%|██▏       | 86160/400000 [00:11<00:36, 8622.35it/s] 22%|██▏       | 87023/400000 [00:11<00:36, 8522.18it/s] 22%|██▏       | 87887/400000 [00:11<00:36, 8555.66it/s] 22%|██▏       | 88774/400000 [00:11<00:35, 8645.28it/s] 22%|██▏       | 89648/400000 [00:11<00:35, 8673.41it/s] 23%|██▎       | 90523/400000 [00:11<00:35, 8695.36it/s] 23%|██▎       | 91435/400000 [00:11<00:34, 8818.26it/s] 23%|██▎       | 92341/400000 [00:12<00:34, 8887.69it/s] 23%|██▎       | 93243/400000 [00:12<00:34, 8924.97it/s] 24%|██▎       | 94150/400000 [00:12<00:34, 8967.77it/s] 24%|██▍       | 95048/400000 [00:12<00:34, 8948.70it/s] 24%|██▍       | 95944/400000 [00:12<00:35, 8663.49it/s] 24%|██▍       | 96813/400000 [00:12<00:35, 8449.38it/s] 24%|██▍       | 97689/400000 [00:12<00:35, 8538.33it/s] 25%|██▍       | 98547/400000 [00:12<00:35, 8547.80it/s] 25%|██▍       | 99404/400000 [00:12<00:35, 8435.10it/s] 25%|██▌       | 100278/400000 [00:12<00:35, 8523.51it/s] 25%|██▌       | 101139/400000 [00:13<00:34, 8549.17it/s] 26%|██▌       | 102045/400000 [00:13<00:34, 8695.44it/s] 26%|██▌       | 102916/400000 [00:13<00:35, 8336.59it/s] 26%|██▌       | 103754/400000 [00:13<00:35, 8295.81it/s] 26%|██▌       | 104621/400000 [00:13<00:35, 8403.50it/s] 26%|██▋       | 105475/400000 [00:13<00:34, 8443.25it/s] 27%|██▋       | 106351/400000 [00:13<00:34, 8534.57it/s] 27%|██▋       | 107223/400000 [00:13<00:34, 8587.56it/s] 27%|██▋       | 108083/400000 [00:13<00:34, 8520.73it/s] 27%|██▋       | 108951/400000 [00:13<00:33, 8565.88it/s] 27%|██▋       | 109809/400000 [00:14<00:33, 8562.23it/s] 28%|██▊       | 110697/400000 [00:14<00:33, 8653.16it/s] 28%|██▊       | 111587/400000 [00:14<00:33, 8723.74it/s] 28%|██▊       | 112475/400000 [00:14<00:32, 8768.42it/s] 28%|██▊       | 113353/400000 [00:14<00:32, 8695.86it/s] 29%|██▊       | 114256/400000 [00:14<00:32, 8793.23it/s] 29%|██▉       | 115150/400000 [00:14<00:32, 8833.87it/s] 29%|██▉       | 116034/400000 [00:14<00:32, 8815.49it/s] 29%|██▉       | 116916/400000 [00:14<00:33, 8551.35it/s] 29%|██▉       | 117774/400000 [00:14<00:33, 8367.19it/s] 30%|██▉       | 118639/400000 [00:15<00:33, 8447.91it/s] 30%|██▉       | 119486/400000 [00:15<00:33, 8427.28it/s] 30%|███       | 120330/400000 [00:15<00:33, 8279.74it/s] 30%|███       | 121160/400000 [00:15<00:34, 8126.72it/s] 31%|███       | 122007/400000 [00:15<00:33, 8226.63it/s] 31%|███       | 122882/400000 [00:15<00:33, 8375.82it/s] 31%|███       | 123726/400000 [00:15<00:32, 8392.44it/s] 31%|███       | 124591/400000 [00:15<00:32, 8467.97it/s] 31%|███▏      | 125439/400000 [00:15<00:32, 8440.97it/s] 32%|███▏      | 126288/400000 [00:15<00:32, 8454.77it/s] 32%|███▏      | 127139/400000 [00:16<00:32, 8470.92it/s] 32%|███▏      | 128038/400000 [00:16<00:31, 8618.73it/s] 32%|███▏      | 128912/400000 [00:16<00:31, 8653.19it/s] 32%|███▏      | 129778/400000 [00:16<00:31, 8550.94it/s] 33%|███▎      | 130634/400000 [00:16<00:32, 8336.65it/s] 33%|███▎      | 131479/400000 [00:16<00:32, 8368.20it/s] 33%|███▎      | 132347/400000 [00:16<00:31, 8459.06it/s] 33%|███▎      | 133194/400000 [00:16<00:31, 8365.65it/s] 34%|███▎      | 134064/400000 [00:16<00:31, 8462.78it/s] 34%|███▎      | 134938/400000 [00:17<00:31, 8543.70it/s] 34%|███▍      | 135813/400000 [00:17<00:30, 8602.09it/s] 34%|███▍      | 136674/400000 [00:17<00:30, 8585.13it/s] 34%|███▍      | 137567/400000 [00:17<00:30, 8685.57it/s] 35%|███▍      | 138437/400000 [00:17<00:30, 8509.52it/s] 35%|███▍      | 139312/400000 [00:17<00:30, 8579.86it/s] 35%|███▌      | 140205/400000 [00:17<00:29, 8680.42it/s] 35%|███▌      | 141116/400000 [00:17<00:29, 8802.67it/s] 36%|███▌      | 142031/400000 [00:17<00:28, 8902.17it/s] 36%|███▌      | 142941/400000 [00:17<00:28, 8958.73it/s] 36%|███▌      | 143838/400000 [00:18<00:28, 8840.69it/s] 36%|███▌      | 144735/400000 [00:18<00:28, 8876.40it/s] 36%|███▋      | 145624/400000 [00:18<00:28, 8870.16it/s] 37%|███▋      | 146522/400000 [00:18<00:28, 8901.59it/s] 37%|███▋      | 147413/400000 [00:18<00:28, 8816.23it/s] 37%|███▋      | 148296/400000 [00:18<00:28, 8710.21it/s] 37%|███▋      | 149168/400000 [00:18<00:29, 8410.64it/s] 38%|███▊      | 150038/400000 [00:18<00:29, 8495.38it/s] 38%|███▊      | 150890/400000 [00:18<00:30, 8188.46it/s] 38%|███▊      | 151717/400000 [00:18<00:30, 8208.72it/s] 38%|███▊      | 152615/400000 [00:19<00:29, 8424.42it/s] 38%|███▊      | 153526/400000 [00:19<00:28, 8617.59it/s] 39%|███▊      | 154430/400000 [00:19<00:28, 8738.37it/s] 39%|███▉      | 155334/400000 [00:19<00:27, 8826.11it/s] 39%|███▉      | 156219/400000 [00:19<00:27, 8801.56it/s] 39%|███▉      | 157101/400000 [00:19<00:27, 8681.38it/s] 40%|███▉      | 158005/400000 [00:19<00:27, 8784.70it/s] 40%|███▉      | 158885/400000 [00:19<00:27, 8785.83it/s] 40%|███▉      | 159797/400000 [00:19<00:27, 8881.93it/s] 40%|████      | 160709/400000 [00:19<00:26, 8949.74it/s] 40%|████      | 161605/400000 [00:20<00:26, 8868.71it/s] 41%|████      | 162493/400000 [00:20<00:26, 8864.76it/s] 41%|████      | 163387/400000 [00:20<00:26, 8884.25it/s] 41%|████      | 164276/400000 [00:20<00:26, 8839.41it/s] 41%|████▏     | 165186/400000 [00:20<00:26, 8913.71it/s] 42%|████▏     | 166078/400000 [00:20<00:26, 8907.34it/s] 42%|████▏     | 166969/400000 [00:20<00:26, 8857.13it/s] 42%|████▏     | 167881/400000 [00:20<00:25, 8932.66it/s] 42%|████▏     | 168775/400000 [00:20<00:25, 8925.30it/s] 42%|████▏     | 169668/400000 [00:20<00:26, 8835.07it/s] 43%|████▎     | 170552/400000 [00:21<00:26, 8761.12it/s] 43%|████▎     | 171450/400000 [00:21<00:25, 8825.13it/s] 43%|████▎     | 172375/400000 [00:21<00:25, 8946.51it/s] 43%|████▎     | 173271/400000 [00:21<00:25, 8813.12it/s] 44%|████▎     | 174154/400000 [00:21<00:25, 8812.54it/s] 44%|████▍     | 175036/400000 [00:21<00:25, 8775.84it/s] 44%|████▍     | 175915/400000 [00:21<00:25, 8631.25it/s] 44%|████▍     | 176785/400000 [00:21<00:25, 8650.92it/s] 44%|████▍     | 177673/400000 [00:21<00:25, 8717.03it/s] 45%|████▍     | 178546/400000 [00:21<00:25, 8688.96it/s] 45%|████▍     | 179416/400000 [00:22<00:25, 8684.02it/s] 45%|████▌     | 180299/400000 [00:22<00:25, 8724.99it/s] 45%|████▌     | 181180/400000 [00:22<00:25, 8749.96it/s] 46%|████▌     | 182078/400000 [00:22<00:24, 8817.29it/s] 46%|████▌     | 182960/400000 [00:22<00:24, 8756.13it/s] 46%|████▌     | 183836/400000 [00:22<00:24, 8750.51it/s] 46%|████▌     | 184712/400000 [00:22<00:24, 8722.42it/s] 46%|████▋     | 185630/400000 [00:22<00:24, 8854.30it/s] 47%|████▋     | 186544/400000 [00:22<00:23, 8937.57it/s] 47%|████▋     | 187439/400000 [00:23<00:23, 8889.25it/s] 47%|████▋     | 188329/400000 [00:23<00:23, 8872.78it/s] 47%|████▋     | 189217/400000 [00:23<00:23, 8827.96it/s] 48%|████▊     | 190101/400000 [00:23<00:24, 8584.12it/s] 48%|████▊     | 190962/400000 [00:23<00:24, 8477.26it/s] 48%|████▊     | 191852/400000 [00:23<00:24, 8598.29it/s] 48%|████▊     | 192732/400000 [00:23<00:23, 8655.48it/s] 48%|████▊     | 193621/400000 [00:23<00:23, 8724.07it/s] 49%|████▊     | 194542/400000 [00:23<00:23, 8863.79it/s] 49%|████▉     | 195430/400000 [00:23<00:23, 8708.40it/s] 49%|████▉     | 196318/400000 [00:24<00:23, 8757.67it/s] 49%|████▉     | 197206/400000 [00:24<00:23, 8792.60it/s] 50%|████▉     | 198116/400000 [00:24<00:22, 8881.30it/s] 50%|████▉     | 199020/400000 [00:24<00:22, 8927.72it/s] 50%|████▉     | 199914/400000 [00:24<00:22, 8857.56it/s] 50%|█████     | 200801/400000 [00:24<00:22, 8843.64it/s] 50%|█████     | 201686/400000 [00:24<00:22, 8823.28it/s] 51%|█████     | 202569/400000 [00:24<00:22, 8771.20it/s] 51%|█████     | 203447/400000 [00:24<00:22, 8652.06it/s] 51%|█████     | 204313/400000 [00:24<00:22, 8600.40it/s] 51%|█████▏    | 205174/400000 [00:25<00:22, 8570.06it/s] 52%|█████▏    | 206032/400000 [00:25<00:22, 8542.53it/s] 52%|█████▏    | 206887/400000 [00:25<00:22, 8483.92it/s] 52%|█████▏    | 207736/400000 [00:25<00:22, 8480.83it/s] 52%|█████▏    | 208585/400000 [00:25<00:22, 8382.90it/s] 52%|█████▏    | 209424/400000 [00:25<00:23, 8268.08it/s] 53%|█████▎    | 210252/400000 [00:25<00:22, 8255.83it/s] 53%|█████▎    | 211084/400000 [00:25<00:22, 8274.69it/s] 53%|█████▎    | 211920/400000 [00:25<00:22, 8300.04it/s] 53%|█████▎    | 212769/400000 [00:25<00:22, 8354.30it/s] 53%|█████▎    | 213605/400000 [00:26<00:22, 8305.38it/s] 54%|█████▎    | 214438/400000 [00:26<00:22, 8311.89it/s] 54%|█████▍    | 215270/400000 [00:26<00:22, 8259.57it/s] 54%|█████▍    | 216097/400000 [00:26<00:22, 8228.16it/s] 54%|█████▍    | 216920/400000 [00:26<00:22, 8214.75it/s] 54%|█████▍    | 217758/400000 [00:26<00:22, 8261.11it/s] 55%|█████▍    | 218585/400000 [00:26<00:22, 7909.15it/s] 55%|█████▍    | 219414/400000 [00:26<00:22, 8019.61it/s] 55%|█████▌    | 220245/400000 [00:26<00:22, 8103.82it/s] 55%|█████▌    | 221078/400000 [00:26<00:21, 8168.73it/s] 55%|█████▌    | 221903/400000 [00:27<00:21, 8191.92it/s] 56%|█████▌    | 222724/400000 [00:27<00:21, 8124.60it/s] 56%|█████▌    | 223538/400000 [00:27<00:21, 8030.73it/s] 56%|█████▌    | 224352/400000 [00:27<00:21, 8060.37it/s] 56%|█████▋    | 225191/400000 [00:27<00:21, 8154.54it/s] 57%|█████▋    | 226019/400000 [00:27<00:21, 8190.94it/s] 57%|█████▋    | 226839/400000 [00:27<00:21, 8165.09it/s] 57%|█████▋    | 227656/400000 [00:27<00:21, 8137.69it/s] 57%|█████▋    | 228478/400000 [00:27<00:21, 8159.55it/s] 57%|█████▋    | 229307/400000 [00:27<00:20, 8195.93it/s] 58%|█████▊    | 230127/400000 [00:28<00:20, 8114.85it/s] 58%|█████▊    | 230939/400000 [00:28<00:20, 8072.91it/s] 58%|█████▊    | 231770/400000 [00:28<00:20, 8142.24it/s] 58%|█████▊    | 232585/400000 [00:28<00:20, 8100.62it/s] 58%|█████▊    | 233414/400000 [00:28<00:20, 8156.15it/s] 59%|█████▊    | 234244/400000 [00:28<00:20, 8197.29it/s] 59%|█████▉    | 235064/400000 [00:28<00:20, 8133.06it/s] 59%|█████▉    | 235888/400000 [00:28<00:20, 8162.11it/s] 59%|█████▉    | 236740/400000 [00:28<00:19, 8266.12it/s] 59%|█████▉    | 237586/400000 [00:28<00:19, 8320.79it/s] 60%|█████▉    | 238444/400000 [00:29<00:19, 8394.63it/s] 60%|█████▉    | 239284/400000 [00:29<00:19, 8278.22it/s] 60%|██████    | 240113/400000 [00:29<00:19, 8190.36it/s] 60%|██████    | 240933/400000 [00:29<00:19, 8032.72it/s] 60%|██████    | 241757/400000 [00:29<00:19, 8091.44it/s] 61%|██████    | 242601/400000 [00:29<00:19, 8191.99it/s] 61%|██████    | 243449/400000 [00:29<00:18, 8274.59it/s] 61%|██████    | 244309/400000 [00:29<00:18, 8369.05it/s] 61%|██████▏   | 245172/400000 [00:29<00:18, 8443.79it/s] 62%|██████▏   | 246018/400000 [00:30<00:18, 8447.50it/s] 62%|██████▏   | 246864/400000 [00:30<00:18, 8434.86it/s] 62%|██████▏   | 247708/400000 [00:30<00:18, 8145.80it/s] 62%|██████▏   | 248542/400000 [00:30<00:18, 8201.17it/s] 62%|██████▏   | 249401/400000 [00:30<00:18, 8312.97it/s] 63%|██████▎   | 250252/400000 [00:30<00:17, 8370.27it/s] 63%|██████▎   | 251110/400000 [00:30<00:17, 8430.11it/s] 63%|██████▎   | 251957/400000 [00:30<00:17, 8439.83it/s] 63%|██████▎   | 252808/400000 [00:30<00:17, 8459.20it/s] 63%|██████▎   | 253660/400000 [00:30<00:17, 8475.33it/s] 64%|██████▎   | 254519/400000 [00:31<00:17, 8507.93it/s] 64%|██████▍   | 255382/400000 [00:31<00:16, 8543.13it/s] 64%|██████▍   | 256246/400000 [00:31<00:16, 8570.09it/s] 64%|██████▍   | 257141/400000 [00:31<00:16, 8679.64it/s] 65%|██████▍   | 258010/400000 [00:31<00:16, 8576.69it/s] 65%|██████▍   | 258895/400000 [00:31<00:16, 8655.86it/s] 65%|██████▍   | 259785/400000 [00:31<00:16, 8726.06it/s] 65%|██████▌   | 260659/400000 [00:31<00:16, 8558.52it/s] 65%|██████▌   | 261516/400000 [00:31<00:16, 8471.02it/s] 66%|██████▌   | 262409/400000 [00:31<00:15, 8600.85it/s] 66%|██████▌   | 263307/400000 [00:32<00:15, 8710.35it/s] 66%|██████▌   | 264187/400000 [00:32<00:15, 8734.40it/s] 66%|██████▋   | 265062/400000 [00:32<00:15, 8679.08it/s] 66%|██████▋   | 265931/400000 [00:32<00:15, 8674.47it/s] 67%|██████▋   | 266816/400000 [00:32<00:15, 8726.27it/s] 67%|██████▋   | 267690/400000 [00:32<00:15, 8682.67it/s] 67%|██████▋   | 268559/400000 [00:32<00:15, 8616.50it/s] 67%|██████▋   | 269421/400000 [00:32<00:15, 8570.10it/s] 68%|██████▊   | 270298/400000 [00:32<00:15, 8626.70it/s] 68%|██████▊   | 271188/400000 [00:32<00:14, 8705.54it/s] 68%|██████▊   | 272072/400000 [00:33<00:14, 8742.73it/s] 68%|██████▊   | 272947/400000 [00:33<00:14, 8727.58it/s] 68%|██████▊   | 273820/400000 [00:33<00:14, 8646.67it/s] 69%|██████▊   | 274718/400000 [00:33<00:14, 8742.08it/s] 69%|██████▉   | 275593/400000 [00:33<00:14, 8594.74it/s] 69%|██████▉   | 276454/400000 [00:33<00:14, 8582.95it/s] 69%|██████▉   | 277344/400000 [00:33<00:14, 8675.02it/s] 70%|██████▉   | 278213/400000 [00:33<00:14, 8466.29it/s] 70%|██████▉   | 279092/400000 [00:33<00:14, 8559.84it/s] 70%|██████▉   | 279954/400000 [00:33<00:13, 8575.43it/s] 70%|███████   | 280841/400000 [00:34<00:13, 8661.22it/s] 70%|███████   | 281748/400000 [00:34<00:13, 8778.07it/s] 71%|███████   | 282627/400000 [00:34<00:13, 8723.36it/s] 71%|███████   | 283505/400000 [00:34<00:13, 8737.33it/s] 71%|███████   | 284380/400000 [00:34<00:13, 8584.87it/s] 71%|███████▏  | 285240/400000 [00:34<00:13, 8546.91it/s] 72%|███████▏  | 286118/400000 [00:34<00:13, 8614.65it/s] 72%|███████▏  | 286981/400000 [00:34<00:13, 8584.93it/s] 72%|███████▏  | 287877/400000 [00:34<00:12, 8691.40it/s] 72%|███████▏  | 288774/400000 [00:34<00:12, 8771.50it/s] 72%|███████▏  | 289679/400000 [00:35<00:12, 8852.17it/s] 73%|███████▎  | 290576/400000 [00:35<00:12, 8886.89it/s] 73%|███████▎  | 291466/400000 [00:35<00:12, 8556.08it/s] 73%|███████▎  | 292325/400000 [00:35<00:12, 8559.91it/s] 73%|███████▎  | 293184/400000 [00:35<00:12, 8511.04it/s] 74%|███████▎  | 294064/400000 [00:35<00:12, 8593.23it/s] 74%|███████▎  | 294940/400000 [00:35<00:12, 8640.42it/s] 74%|███████▍  | 295805/400000 [00:35<00:12, 8528.06it/s] 74%|███████▍  | 296693/400000 [00:35<00:11, 8628.46it/s] 74%|███████▍  | 297573/400000 [00:35<00:11, 8676.49it/s] 75%|███████▍  | 298442/400000 [00:36<00:11, 8669.03it/s] 75%|███████▍  | 299325/400000 [00:36<00:11, 8713.84it/s] 75%|███████▌  | 300197/400000 [00:36<00:11, 8626.62it/s] 75%|███████▌  | 301063/400000 [00:36<00:11, 8635.97it/s] 75%|███████▌  | 301961/400000 [00:36<00:11, 8736.23it/s] 76%|███████▌  | 302853/400000 [00:36<00:11, 8787.64it/s] 76%|███████▌  | 303751/400000 [00:36<00:10, 8843.75it/s] 76%|███████▌  | 304636/400000 [00:36<00:10, 8747.36it/s] 76%|███████▋  | 305512/400000 [00:36<00:10, 8634.63it/s] 77%|███████▋  | 306377/400000 [00:37<00:10, 8607.38it/s] 77%|███████▋  | 307250/400000 [00:37<00:10, 8643.15it/s] 77%|███████▋  | 308126/400000 [00:37<00:10, 8675.42it/s] 77%|███████▋  | 308994/400000 [00:37<00:10, 8594.21it/s] 77%|███████▋  | 309859/400000 [00:37<00:10, 8609.00it/s] 78%|███████▊  | 310721/400000 [00:37<00:10, 8602.39it/s] 78%|███████▊  | 311582/400000 [00:37<00:10, 8589.18it/s] 78%|███████▊  | 312442/400000 [00:37<00:10, 8506.22it/s] 78%|███████▊  | 313293/400000 [00:37<00:10, 8433.22it/s] 79%|███████▊  | 314173/400000 [00:37<00:10, 8538.08it/s] 79%|███████▉  | 315080/400000 [00:38<00:09, 8690.33it/s] 79%|███████▉  | 315965/400000 [00:38<00:09, 8737.33it/s] 79%|███████▉  | 316862/400000 [00:38<00:09, 8803.52it/s] 79%|███████▉  | 317744/400000 [00:38<00:09, 8669.98it/s] 80%|███████▉  | 318619/400000 [00:38<00:09, 8693.13it/s] 80%|███████▉  | 319489/400000 [00:38<00:09, 8685.55it/s] 80%|████████  | 320359/400000 [00:38<00:09, 8294.14it/s] 80%|████████  | 321240/400000 [00:38<00:09, 8442.38it/s] 81%|████████  | 322088/400000 [00:38<00:09, 8451.08it/s] 81%|████████  | 322980/400000 [00:38<00:08, 8586.08it/s] 81%|████████  | 323895/400000 [00:39<00:08, 8745.86it/s] 81%|████████  | 324792/400000 [00:39<00:08, 8809.32it/s] 81%|████████▏ | 325710/400000 [00:39<00:08, 8916.73it/s] 82%|████████▏ | 326604/400000 [00:39<00:08, 8693.73it/s] 82%|████████▏ | 327476/400000 [00:39<00:08, 8636.24it/s] 82%|████████▏ | 328391/400000 [00:39<00:08, 8782.32it/s] 82%|████████▏ | 329272/400000 [00:39<00:08, 8635.61it/s] 83%|████████▎ | 330141/400000 [00:39<00:08, 8649.48it/s] 83%|████████▎ | 331008/400000 [00:39<00:07, 8625.87it/s] 83%|████████▎ | 331914/400000 [00:39<00:07, 8751.45it/s] 83%|████████▎ | 332791/400000 [00:40<00:07, 8593.27it/s] 83%|████████▎ | 333658/400000 [00:40<00:07, 8614.30it/s] 84%|████████▎ | 334521/400000 [00:40<00:07, 8598.58it/s] 84%|████████▍ | 335396/400000 [00:40<00:07, 8641.46it/s] 84%|████████▍ | 336296/400000 [00:40<00:07, 8745.26it/s] 84%|████████▍ | 337172/400000 [00:40<00:07, 8673.05it/s] 85%|████████▍ | 338040/400000 [00:40<00:07, 8651.37it/s] 85%|████████▍ | 338967/400000 [00:40<00:06, 8826.96it/s] 85%|████████▍ | 339854/400000 [00:40<00:06, 8838.11it/s] 85%|████████▌ | 340739/400000 [00:40<00:06, 8829.46it/s] 85%|████████▌ | 341647/400000 [00:41<00:06, 8901.86it/s] 86%|████████▌ | 342538/400000 [00:41<00:06, 8890.76it/s] 86%|████████▌ | 343428/400000 [00:41<00:06, 8584.03it/s] 86%|████████▌ | 344289/400000 [00:41<00:06, 8478.38it/s] 86%|████████▋ | 345139/400000 [00:41<00:06, 8408.78it/s] 87%|████████▋ | 346031/400000 [00:41<00:06, 8553.42it/s] 87%|████████▋ | 346921/400000 [00:41<00:06, 8652.21it/s] 87%|████████▋ | 347792/400000 [00:41<00:06, 8666.57it/s] 87%|████████▋ | 348660/400000 [00:41<00:06, 8531.91it/s] 87%|████████▋ | 349574/400000 [00:42<00:05, 8703.76it/s] 88%|████████▊ | 350494/400000 [00:42<00:05, 8845.55it/s] 88%|████████▊ | 351404/400000 [00:42<00:05, 8919.41it/s] 88%|████████▊ | 352298/400000 [00:42<00:05, 8892.45it/s] 88%|████████▊ | 353189/400000 [00:42<00:05, 8756.27it/s] 89%|████████▊ | 354070/400000 [00:42<00:05, 8770.91it/s] 89%|████████▊ | 354956/400000 [00:42<00:05, 8797.37it/s] 89%|████████▉ | 355837/400000 [00:42<00:05, 8725.07it/s] 89%|████████▉ | 356711/400000 [00:42<00:05, 8593.56it/s] 89%|████████▉ | 357585/400000 [00:42<00:04, 8635.42it/s] 90%|████████▉ | 358473/400000 [00:43<00:04, 8705.08it/s] 90%|████████▉ | 359346/400000 [00:43<00:04, 8712.27it/s] 90%|█████████ | 360229/400000 [00:43<00:04, 8746.80it/s] 90%|█████████ | 361120/400000 [00:43<00:04, 8793.85it/s] 90%|█████████ | 362000/400000 [00:43<00:04, 8745.62it/s] 91%|█████████ | 362875/400000 [00:43<00:04, 8735.30it/s] 91%|█████████ | 363749/400000 [00:43<00:04, 8499.31it/s] 91%|█████████ | 364629/400000 [00:43<00:04, 8585.25it/s] 91%|█████████▏| 365548/400000 [00:43<00:03, 8756.62it/s] 92%|█████████▏| 366426/400000 [00:43<00:03, 8758.13it/s] 92%|█████████▏| 367311/400000 [00:44<00:03, 8782.04it/s] 92%|█████████▏| 368214/400000 [00:44<00:03, 8851.92it/s] 92%|█████████▏| 369100/400000 [00:44<00:03, 8852.22it/s] 92%|█████████▏| 369986/400000 [00:44<00:03, 8752.04it/s] 93%|█████████▎| 370862/400000 [00:44<00:03, 8682.51it/s] 93%|█████████▎| 371731/400000 [00:44<00:03, 8546.44it/s] 93%|█████████▎| 372610/400000 [00:44<00:03, 8614.60it/s] 93%|█████████▎| 373494/400000 [00:44<00:03, 8679.22it/s] 94%|█████████▎| 374363/400000 [00:44<00:03, 8306.70it/s] 94%|█████████▍| 375198/400000 [00:44<00:02, 8296.64it/s] 94%|█████████▍| 376031/400000 [00:45<00:02, 8298.94it/s] 94%|█████████▍| 376903/400000 [00:45<00:02, 8420.09it/s] 94%|█████████▍| 377771/400000 [00:45<00:02, 8495.63it/s] 95%|█████████▍| 378665/400000 [00:45<00:02, 8622.26it/s] 95%|█████████▍| 379529/400000 [00:45<00:02, 8471.80it/s] 95%|█████████▌| 380426/400000 [00:45<00:02, 8614.93it/s] 95%|█████████▌| 381347/400000 [00:45<00:02, 8782.50it/s] 96%|█████████▌| 382263/400000 [00:45<00:01, 8889.61it/s] 96%|█████████▌| 383177/400000 [00:45<00:01, 8962.62it/s] 96%|█████████▌| 384075/400000 [00:45<00:01, 8866.09it/s] 96%|█████████▌| 384963/400000 [00:46<00:01, 8862.70it/s] 96%|█████████▋| 385851/400000 [00:46<00:01, 8834.28it/s] 97%|█████████▋| 386762/400000 [00:46<00:01, 8915.00it/s] 97%|█████████▋| 387665/400000 [00:46<00:01, 8946.42it/s] 97%|█████████▋| 388561/400000 [00:46<00:01, 8837.92it/s] 97%|█████████▋| 389456/400000 [00:46<00:01, 8869.84it/s] 98%|█████████▊| 390344/400000 [00:46<00:01, 8779.89it/s] 98%|█████████▊| 391223/400000 [00:46<00:01, 8616.18it/s] 98%|█████████▊| 392086/400000 [00:46<00:00, 8572.49it/s] 98%|█████████▊| 392945/400000 [00:46<00:00, 8449.97it/s] 98%|█████████▊| 393791/400000 [00:47<00:00, 8413.76it/s] 99%|█████████▊| 394694/400000 [00:47<00:00, 8587.51it/s] 99%|█████████▉| 395602/400000 [00:47<00:00, 8728.02it/s] 99%|█████████▉| 396520/400000 [00:47<00:00, 8856.59it/s] 99%|█████████▉| 397408/400000 [00:47<00:00, 8609.31it/s]100%|█████████▉| 398272/400000 [00:47<00:00, 8514.74it/s]100%|█████████▉| 399141/400000 [00:47<00:00, 8564.71it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8367.50it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f43b8b70828>, <torchtext.data.dataset.TabularDataset object at 0x7f43b8b70978>, <torchtext.vocab.Vocab object at 0x7f43b8b70898>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 37.85 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 47.78 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.87 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.98 file/s]2020-06-15 09:12:00.499881: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 09:12:00.504128: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-15 09:12:00.504289: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cf126c7530 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 09:12:00.504303: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3768320/9912422 [00:00<00:00, 36919978.34it/s]9920512it [00:00, 38903187.32it/s]                             
0it [00:00, ?it/s]32768it [00:00, 642562.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485829.69it/s]1654784it [00:00, 12055713.49it/s]                         
0it [00:00, ?it/s]8192it [00:00, 221029.75it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
