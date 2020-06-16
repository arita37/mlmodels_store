
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/70f1a18a5306e0e1f742c0397a3d2aeed0ee5b84

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7faa6ef85f28> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7faa6ef85f28> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7faada53d0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7faada53d0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7faaf4269ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7faaf4269ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7faa87dba950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7faa87dba950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7faa87dba950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3055616/9912422 [00:00<00:00, 30549706.71it/s]9920512it [00:00, 34374441.89it/s]                             
0it [00:00, ?it/s]32768it [00:00, 940653.98it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463058.85it/s]1654784it [00:00, 11206771.13it/s]                         
0it [00:00, ?it/s]8192it [00:00, 239250.62it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa85405710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa853fb978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7faa87dba598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7faa87dba598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7faa87dba598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:43,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:13,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:09,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  3.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:33,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:32,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:32,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:32,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:32,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:22,  5.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:15,  8.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 42.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 47.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 52.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 51.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 49.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 45.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 45.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 40.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 35.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 35.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 35.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 35.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 35.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 35.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 32.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 32.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 32.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 32.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 32.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 30.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 30.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 30.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 30.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 30.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 28.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 28.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 28.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 28.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 28.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 27.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 27.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 27.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 27.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 26.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 26.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 26.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 26.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 25.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 25.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 25.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 25.78 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 25.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 25.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 25.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 25.35 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 25.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.22 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.42s/ url]
0 examples [00:00, ? examples/s]2020-06-16 18:09:52.925847: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-16 18:09:52.939368: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-16 18:09:52.939575: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566be2162d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 18:09:52.939596: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 198.49 examples/s]104 examples [00:00, 257.37 examples/s]192 examples [00:00, 325.92 examples/s]279 examples [00:00, 400.76 examples/s]373 examples [00:00, 483.57 examples/s]452 examples [00:00, 547.06 examples/s]544 examples [00:00, 622.05 examples/s]630 examples [00:00, 678.34 examples/s]723 examples [00:00, 736.87 examples/s]816 examples [00:01, 784.31 examples/s]913 examples [00:01, 830.21 examples/s]1007 examples [00:01, 859.64 examples/s]1098 examples [00:01, 855.81 examples/s]1193 examples [00:01, 880.09 examples/s]1287 examples [00:01, 894.45 examples/s]1384 examples [00:01, 914.50 examples/s]1477 examples [00:01, 905.33 examples/s]1569 examples [00:01, 898.10 examples/s]1664 examples [00:01, 910.62 examples/s]1756 examples [00:02, 899.42 examples/s]1850 examples [00:02, 908.84 examples/s]1942 examples [00:02, 910.97 examples/s]2034 examples [00:02, 911.56 examples/s]2126 examples [00:02, 913.00 examples/s]2218 examples [00:02, 914.82 examples/s]2310 examples [00:02, 896.07 examples/s]2400 examples [00:02, 894.59 examples/s]2492 examples [00:02, 895.70 examples/s]2585 examples [00:02, 904.03 examples/s]2677 examples [00:03, 908.48 examples/s]2768 examples [00:03, 880.53 examples/s]2858 examples [00:03, 885.11 examples/s]2951 examples [00:03, 896.88 examples/s]3046 examples [00:03, 909.86 examples/s]3138 examples [00:03, 883.92 examples/s]3227 examples [00:03, 866.95 examples/s]3315 examples [00:03, 867.55 examples/s]3406 examples [00:03, 877.88 examples/s]3494 examples [00:03, 876.14 examples/s]3582 examples [00:04, 875.66 examples/s]3670 examples [00:04, 845.67 examples/s]3755 examples [00:04, 817.39 examples/s]3838 examples [00:04, 820.75 examples/s]3921 examples [00:04, 818.10 examples/s]4009 examples [00:04, 833.62 examples/s]4099 examples [00:04, 851.78 examples/s]4185 examples [00:04, 835.78 examples/s]4269 examples [00:04, 817.52 examples/s]4362 examples [00:05, 846.79 examples/s]4458 examples [00:05, 877.27 examples/s]4552 examples [00:05, 894.55 examples/s]4643 examples [00:05, 897.41 examples/s]4738 examples [00:05, 910.88 examples/s]4832 examples [00:05, 918.19 examples/s]4926 examples [00:05, 923.26 examples/s]5019 examples [00:05, 914.35 examples/s]5111 examples [00:05, 899.70 examples/s]5203 examples [00:05, 904.20 examples/s]5294 examples [00:06, 902.35 examples/s]5390 examples [00:06, 916.33 examples/s]5485 examples [00:06, 925.66 examples/s]5578 examples [00:06, 905.53 examples/s]5672 examples [00:06, 914.76 examples/s]5767 examples [00:06, 923.01 examples/s]5863 examples [00:06, 932.85 examples/s]5957 examples [00:06, 904.83 examples/s]6049 examples [00:06, 906.80 examples/s]6141 examples [00:06, 909.59 examples/s]6233 examples [00:07, 903.26 examples/s]6327 examples [00:07, 912.31 examples/s]6419 examples [00:07, 911.59 examples/s]6511 examples [00:07, 911.99 examples/s]6603 examples [00:07, 890.82 examples/s]6693 examples [00:07, 887.61 examples/s]6787 examples [00:07, 893.21 examples/s]6881 examples [00:07, 904.39 examples/s]6972 examples [00:07, 894.71 examples/s]7064 examples [00:07, 900.98 examples/s]7155 examples [00:08, 896.03 examples/s]7248 examples [00:08, 904.63 examples/s]7340 examples [00:08, 907.84 examples/s]7431 examples [00:08, 906.08 examples/s]7523 examples [00:08, 909.39 examples/s]7614 examples [00:08, 901.90 examples/s]7706 examples [00:08, 907.03 examples/s]7800 examples [00:08, 914.38 examples/s]7892 examples [00:08, 877.76 examples/s]7981 examples [00:09, 865.98 examples/s]8071 examples [00:09, 874.03 examples/s]8161 examples [00:09, 879.96 examples/s]8253 examples [00:09, 889.11 examples/s]8343 examples [00:09, 888.19 examples/s]8433 examples [00:09, 891.46 examples/s]8524 examples [00:09, 895.06 examples/s]8614 examples [00:09, 882.34 examples/s]8703 examples [00:09, 867.62 examples/s]8793 examples [00:09, 876.60 examples/s]8883 examples [00:10, 883.09 examples/s]8972 examples [00:10, 859.65 examples/s]9059 examples [00:10, 857.23 examples/s]9150 examples [00:10, 872.12 examples/s]9238 examples [00:10, 855.17 examples/s]9324 examples [00:10, 850.63 examples/s]9412 examples [00:10, 857.86 examples/s]9498 examples [00:10, 856.28 examples/s]9588 examples [00:10, 868.88 examples/s]9675 examples [00:10, 868.63 examples/s]9764 examples [00:11, 874.54 examples/s]9857 examples [00:11, 890.40 examples/s]9947 examples [00:11, 890.94 examples/s]10037 examples [00:11, 844.93 examples/s]10131 examples [00:11, 871.22 examples/s]10224 examples [00:11, 887.59 examples/s]10316 examples [00:11, 895.84 examples/s]10406 examples [00:11, 892.50 examples/s]10496 examples [00:11, 891.46 examples/s]10588 examples [00:11, 897.23 examples/s]10678 examples [00:12, 871.68 examples/s]10766 examples [00:12, 865.77 examples/s]10853 examples [00:12, 864.93 examples/s]10940 examples [00:12, 833.53 examples/s]11026 examples [00:12, 840.47 examples/s]11113 examples [00:12, 849.02 examples/s]11209 examples [00:12, 879.16 examples/s]11300 examples [00:12, 885.53 examples/s]11389 examples [00:12, 871.39 examples/s]11477 examples [00:13, 805.84 examples/s]11562 examples [00:13, 816.51 examples/s]11650 examples [00:13, 833.28 examples/s]11737 examples [00:13, 842.18 examples/s]11823 examples [00:13, 845.62 examples/s]11914 examples [00:13, 861.73 examples/s]12009 examples [00:13, 886.06 examples/s]12099 examples [00:13, 862.15 examples/s]12186 examples [00:13, 857.25 examples/s]12273 examples [00:13, 843.76 examples/s]12363 examples [00:14, 858.17 examples/s]12452 examples [00:14, 865.68 examples/s]12542 examples [00:14, 874.52 examples/s]12630 examples [00:14, 872.27 examples/s]12720 examples [00:14, 877.55 examples/s]12808 examples [00:14, 852.96 examples/s]12895 examples [00:14, 857.64 examples/s]12983 examples [00:14, 862.60 examples/s]13071 examples [00:14, 865.30 examples/s]13159 examples [00:14, 868.05 examples/s]13246 examples [00:15, 862.69 examples/s]13333 examples [00:15, 850.12 examples/s]13423 examples [00:15, 862.73 examples/s]13512 examples [00:15, 869.40 examples/s]13602 examples [00:15, 877.82 examples/s]13691 examples [00:15, 879.21 examples/s]13779 examples [00:15, 850.53 examples/s]13866 examples [00:15, 855.81 examples/s]13952 examples [00:15, 847.69 examples/s]14039 examples [00:16, 853.07 examples/s]14128 examples [00:16, 863.42 examples/s]14215 examples [00:16, 863.75 examples/s]14305 examples [00:16, 873.18 examples/s]14395 examples [00:16, 879.28 examples/s]14486 examples [00:16, 887.74 examples/s]14578 examples [00:16, 896.47 examples/s]14668 examples [00:16, 884.30 examples/s]14757 examples [00:16, 872.93 examples/s]14845 examples [00:16, 868.73 examples/s]14932 examples [00:17, 844.01 examples/s]15017 examples [00:17, 838.24 examples/s]15101 examples [00:17, 831.84 examples/s]15185 examples [00:17, 828.57 examples/s]15275 examples [00:17, 847.23 examples/s]15368 examples [00:17, 868.52 examples/s]15463 examples [00:17, 890.36 examples/s]15556 examples [00:17, 901.24 examples/s]15648 examples [00:17, 904.09 examples/s]15740 examples [00:17, 906.05 examples/s]15831 examples [00:18, 881.28 examples/s]15920 examples [00:18, 869.70 examples/s]16008 examples [00:18, 867.38 examples/s]16096 examples [00:18, 869.00 examples/s]16183 examples [00:18, 862.00 examples/s]16270 examples [00:18, 843.43 examples/s]16358 examples [00:18, 853.76 examples/s]16445 examples [00:18, 857.96 examples/s]16531 examples [00:18, 842.47 examples/s]16616 examples [00:19, 786.42 examples/s]16696 examples [00:19, 780.45 examples/s]16780 examples [00:19, 794.98 examples/s]16860 examples [00:19, 795.83 examples/s]16940 examples [00:19, 781.65 examples/s]17019 examples [00:19, 772.64 examples/s]17106 examples [00:19, 799.18 examples/s]17193 examples [00:19, 818.85 examples/s]17280 examples [00:19, 831.60 examples/s]17366 examples [00:19, 839.00 examples/s]17456 examples [00:20, 854.35 examples/s]17542 examples [00:20, 854.01 examples/s]17628 examples [00:20, 844.32 examples/s]17713 examples [00:20, 845.02 examples/s]17798 examples [00:20, 840.39 examples/s]17883 examples [00:20, 804.88 examples/s]17966 examples [00:20, 809.92 examples/s]18051 examples [00:20, 820.19 examples/s]18137 examples [00:20, 830.77 examples/s]18221 examples [00:20, 831.67 examples/s]18305 examples [00:21, 822.10 examples/s]18390 examples [00:21, 829.98 examples/s]18474 examples [00:21, 830.73 examples/s]18558 examples [00:21, 829.89 examples/s]18643 examples [00:21, 832.99 examples/s]18727 examples [00:21, 823.81 examples/s]18810 examples [00:21, 825.09 examples/s]18898 examples [00:21, 838.91 examples/s]18989 examples [00:21, 857.83 examples/s]19081 examples [00:21, 870.20 examples/s]19169 examples [00:22, 852.61 examples/s]19258 examples [00:22, 862.22 examples/s]19345 examples [00:22, 859.89 examples/s]19435 examples [00:22, 869.27 examples/s]19527 examples [00:22, 882.49 examples/s]19618 examples [00:22, 888.38 examples/s]19707 examples [00:22, 880.46 examples/s]19798 examples [00:22, 888.11 examples/s]19888 examples [00:22, 889.46 examples/s]19981 examples [00:22, 899.17 examples/s]20071 examples [00:23, 856.31 examples/s]20160 examples [00:23, 863.55 examples/s]20249 examples [00:23, 869.46 examples/s]20337 examples [00:23, 871.81 examples/s]20431 examples [00:23, 888.73 examples/s]20522 examples [00:23, 892.92 examples/s]20612 examples [00:23, 890.49 examples/s]20702 examples [00:23, 888.95 examples/s]20791 examples [00:23, 879.42 examples/s]20883 examples [00:24, 889.88 examples/s]20973 examples [00:24, 852.85 examples/s]21063 examples [00:24, 865.86 examples/s]21153 examples [00:24, 874.00 examples/s]21241 examples [00:24, 873.85 examples/s]21329 examples [00:24, 868.45 examples/s]21419 examples [00:24, 875.94 examples/s]21510 examples [00:24, 885.64 examples/s]21602 examples [00:24, 895.53 examples/s]21693 examples [00:24, 899.37 examples/s]21784 examples [00:25, 899.94 examples/s]21875 examples [00:25, 896.64 examples/s]21965 examples [00:25, 858.87 examples/s]22052 examples [00:25, 812.57 examples/s]22135 examples [00:25, 814.96 examples/s]22224 examples [00:25, 836.10 examples/s]22312 examples [00:25, 846.61 examples/s]22398 examples [00:25, 831.40 examples/s]22488 examples [00:25, 848.64 examples/s]22578 examples [00:25, 856.95 examples/s]22667 examples [00:26, 865.24 examples/s]22755 examples [00:26, 867.79 examples/s]22846 examples [00:26, 879.97 examples/s]22938 examples [00:26, 891.06 examples/s]23029 examples [00:26, 895.38 examples/s]23119 examples [00:26, 895.44 examples/s]23209 examples [00:26, 861.92 examples/s]23296 examples [00:26, 854.57 examples/s]23382 examples [00:26, 851.83 examples/s]23478 examples [00:27, 880.76 examples/s]23567 examples [00:27, 861.60 examples/s]23654 examples [00:27, 849.59 examples/s]23745 examples [00:27, 865.52 examples/s]23834 examples [00:27, 871.30 examples/s]23923 examples [00:27, 874.88 examples/s]24012 examples [00:27, 878.80 examples/s]24100 examples [00:27, 874.19 examples/s]24188 examples [00:27, 859.04 examples/s]24279 examples [00:27, 871.66 examples/s]24367 examples [00:28, 864.95 examples/s]24454 examples [00:28, 848.86 examples/s]24544 examples [00:28, 862.46 examples/s]24631 examples [00:28, 864.45 examples/s]24723 examples [00:28, 878.52 examples/s]24811 examples [00:28, 868.40 examples/s]24899 examples [00:28, 871.80 examples/s]24987 examples [00:28, 866.00 examples/s]25074 examples [00:28, 861.57 examples/s]25163 examples [00:28, 867.26 examples/s]25254 examples [00:29, 879.08 examples/s]25342 examples [00:29, 845.89 examples/s]25427 examples [00:29, 790.52 examples/s]25511 examples [00:29, 802.66 examples/s]25598 examples [00:29, 821.38 examples/s]25685 examples [00:29, 833.70 examples/s]25777 examples [00:29, 857.40 examples/s]25868 examples [00:29, 871.21 examples/s]25958 examples [00:29, 877.65 examples/s]26047 examples [00:30, 879.79 examples/s]26136 examples [00:30, 866.54 examples/s]26225 examples [00:30, 872.69 examples/s]26318 examples [00:30, 887.41 examples/s]26408 examples [00:30, 890.61 examples/s]26502 examples [00:30, 904.08 examples/s]26593 examples [00:30, 889.90 examples/s]26684 examples [00:30, 892.54 examples/s]26778 examples [00:30, 906.14 examples/s]26871 examples [00:30, 912.19 examples/s]26963 examples [00:31, 909.15 examples/s]27054 examples [00:31, 866.97 examples/s]27144 examples [00:31, 875.56 examples/s]27233 examples [00:31, 877.67 examples/s]27322 examples [00:31, 830.59 examples/s]27416 examples [00:31, 858.18 examples/s]27506 examples [00:31, 869.75 examples/s]27597 examples [00:31, 880.11 examples/s]27689 examples [00:31, 890.63 examples/s]27780 examples [00:31, 893.99 examples/s]27870 examples [00:32, 879.90 examples/s]27959 examples [00:32, 876.93 examples/s]28047 examples [00:32, 875.28 examples/s]28140 examples [00:32, 888.60 examples/s]28235 examples [00:32, 904.09 examples/s]28329 examples [00:32, 913.71 examples/s]28421 examples [00:32, 909.88 examples/s]28516 examples [00:32, 918.59 examples/s]28608 examples [00:32, 917.29 examples/s]28700 examples [00:32, 913.16 examples/s]28792 examples [00:33, 889.92 examples/s]28882 examples [00:33, 874.44 examples/s]28973 examples [00:33, 884.02 examples/s]29062 examples [00:33, 879.45 examples/s]29151 examples [00:33, 882.24 examples/s]29240 examples [00:33, 877.98 examples/s]29329 examples [00:33, 879.54 examples/s]29417 examples [00:33, 872.91 examples/s]29505 examples [00:33, 868.73 examples/s]29592 examples [00:34, 864.37 examples/s]29679 examples [00:34, 857.08 examples/s]29765 examples [00:34, 837.07 examples/s]29856 examples [00:34, 856.36 examples/s]29942 examples [00:34, 849.36 examples/s]30028 examples [00:34, 818.30 examples/s]30114 examples [00:34, 829.39 examples/s]30199 examples [00:34, 835.27 examples/s]30283 examples [00:34, 828.53 examples/s]30377 examples [00:34, 857.04 examples/s]30468 examples [00:35, 870.42 examples/s]30559 examples [00:35, 879.71 examples/s]30648 examples [00:35, 859.92 examples/s]30735 examples [00:35, 851.63 examples/s]30825 examples [00:35, 864.96 examples/s]30913 examples [00:35, 867.60 examples/s]31000 examples [00:35, 850.97 examples/s]31087 examples [00:35, 855.54 examples/s]31173 examples [00:35, 834.49 examples/s]31257 examples [00:35, 825.39 examples/s]31340 examples [00:36, 812.66 examples/s]31422 examples [00:36, 790.13 examples/s]31511 examples [00:36, 817.24 examples/s]31601 examples [00:36, 838.68 examples/s]31692 examples [00:36, 857.81 examples/s]31779 examples [00:36, 856.79 examples/s]31866 examples [00:36, 860.33 examples/s]31953 examples [00:36, 857.96 examples/s]32040 examples [00:36, 859.27 examples/s]32127 examples [00:37, 851.74 examples/s]32221 examples [00:37, 874.23 examples/s]32309 examples [00:37, 844.17 examples/s]32404 examples [00:37, 872.09 examples/s]32492 examples [00:37, 819.44 examples/s]32580 examples [00:37, 835.41 examples/s]32670 examples [00:37, 852.97 examples/s]32763 examples [00:37, 874.33 examples/s]32859 examples [00:37, 896.30 examples/s]32950 examples [00:37, 878.51 examples/s]33039 examples [00:38, 878.61 examples/s]33128 examples [00:38, 868.88 examples/s]33218 examples [00:38, 876.78 examples/s]33311 examples [00:38, 890.15 examples/s]33403 examples [00:38, 897.11 examples/s]33493 examples [00:38, 884.92 examples/s]33586 examples [00:38, 897.41 examples/s]33678 examples [00:38, 902.12 examples/s]33771 examples [00:38, 908.87 examples/s]33863 examples [00:38, 911.45 examples/s]33957 examples [00:39, 918.00 examples/s]34049 examples [00:39, 881.13 examples/s]34138 examples [00:39, 865.14 examples/s]34225 examples [00:39, 862.66 examples/s]34314 examples [00:39, 870.34 examples/s]34406 examples [00:39, 883.39 examples/s]34499 examples [00:39, 894.63 examples/s]34592 examples [00:39, 903.18 examples/s]34684 examples [00:39, 906.69 examples/s]34775 examples [00:40, 904.36 examples/s]34868 examples [00:40, 909.62 examples/s]34960 examples [00:40, 900.09 examples/s]35051 examples [00:40, 877.22 examples/s]35140 examples [00:40, 867.54 examples/s]35227 examples [00:40, 832.30 examples/s]35313 examples [00:40, 839.30 examples/s]35398 examples [00:40, 822.82 examples/s]35485 examples [00:40, 836.25 examples/s]35569 examples [00:40, 819.96 examples/s]35658 examples [00:41, 837.88 examples/s]35750 examples [00:41, 858.25 examples/s]35838 examples [00:41, 862.06 examples/s]35928 examples [00:41, 870.94 examples/s]36016 examples [00:41, 855.15 examples/s]36102 examples [00:41, 840.09 examples/s]36192 examples [00:41, 854.81 examples/s]36284 examples [00:41, 871.66 examples/s]36372 examples [00:41, 868.20 examples/s]36459 examples [00:41, 838.14 examples/s]36544 examples [00:42, 826.23 examples/s]36633 examples [00:42, 843.69 examples/s]36727 examples [00:42, 868.28 examples/s]36815 examples [00:42, 863.07 examples/s]36904 examples [00:42, 868.95 examples/s]36992 examples [00:42, 870.57 examples/s]37086 examples [00:42, 889.55 examples/s]37177 examples [00:42, 893.27 examples/s]37270 examples [00:42, 901.92 examples/s]37363 examples [00:43, 907.55 examples/s]37454 examples [00:43, 903.80 examples/s]37545 examples [00:43, 887.85 examples/s]37634 examples [00:43, 872.33 examples/s]37722 examples [00:43, 860.08 examples/s]37809 examples [00:43, 852.61 examples/s]37895 examples [00:43, 817.25 examples/s]37983 examples [00:43, 834.01 examples/s]38068 examples [00:43, 838.67 examples/s]38154 examples [00:43, 842.40 examples/s]38239 examples [00:44, 840.91 examples/s]38327 examples [00:44, 851.69 examples/s]38418 examples [00:44, 867.21 examples/s]38508 examples [00:44, 875.51 examples/s]38601 examples [00:44, 888.71 examples/s]38692 examples [00:44, 894.67 examples/s]38784 examples [00:44, 900.93 examples/s]38877 examples [00:44, 907.03 examples/s]38971 examples [00:44, 914.04 examples/s]39063 examples [00:44, 900.40 examples/s]39154 examples [00:45, 902.25 examples/s]39245 examples [00:45, 895.32 examples/s]39335 examples [00:45, 884.08 examples/s]39431 examples [00:45, 903.22 examples/s]39522 examples [00:45, 900.52 examples/s]39613 examples [00:45, 873.20 examples/s]39704 examples [00:45, 881.46 examples/s]39797 examples [00:45, 894.24 examples/s]39890 examples [00:45, 902.37 examples/s]39985 examples [00:45, 915.67 examples/s]40077 examples [00:46, 871.14 examples/s]40165 examples [00:46, 872.43 examples/s]40258 examples [00:46, 887.69 examples/s]40350 examples [00:46, 895.39 examples/s]40440 examples [00:46, 892.96 examples/s]40530 examples [00:46, 886.66 examples/s]40619 examples [00:46, 876.81 examples/s]40710 examples [00:46, 884.56 examples/s]40799 examples [00:46, 882.79 examples/s]40892 examples [00:47, 895.79 examples/s]40982 examples [00:47, 886.09 examples/s]41075 examples [00:47, 898.63 examples/s]41165 examples [00:47, 888.82 examples/s]41258 examples [00:47, 899.04 examples/s]41349 examples [00:47, 878.57 examples/s]41438 examples [00:47, 878.98 examples/s]41527 examples [00:47, 849.28 examples/s]41613 examples [00:47, 849.76 examples/s]41699 examples [00:47, 840.91 examples/s]41784 examples [00:48, 840.12 examples/s]41872 examples [00:48, 850.29 examples/s]41958 examples [00:48, 837.71 examples/s]42042 examples [00:48, 830.06 examples/s]42126 examples [00:48, 828.86 examples/s]42209 examples [00:48, 829.05 examples/s]42296 examples [00:48, 838.46 examples/s]42380 examples [00:48, 825.45 examples/s]42463 examples [00:48, 824.36 examples/s]42546 examples [00:48, 812.26 examples/s]42633 examples [00:49, 828.49 examples/s]42716 examples [00:49, 817.01 examples/s]42799 examples [00:49, 819.18 examples/s]42883 examples [00:49, 823.91 examples/s]42966 examples [00:49, 817.55 examples/s]43048 examples [00:49, 756.44 examples/s]43133 examples [00:49, 781.04 examples/s]43215 examples [00:49, 790.19 examples/s]43297 examples [00:49, 798.21 examples/s]43389 examples [00:50, 830.60 examples/s]43478 examples [00:50, 846.06 examples/s]43569 examples [00:50, 862.80 examples/s]43657 examples [00:50, 865.34 examples/s]43744 examples [00:50, 858.03 examples/s]43831 examples [00:50, 855.77 examples/s]43917 examples [00:50, 837.95 examples/s]44001 examples [00:50, 833.33 examples/s]44090 examples [00:50, 848.81 examples/s]44180 examples [00:50, 862.54 examples/s]44267 examples [00:51, 860.73 examples/s]44354 examples [00:51, 851.70 examples/s]44443 examples [00:51, 861.34 examples/s]44530 examples [00:51, 855.84 examples/s]44619 examples [00:51, 865.80 examples/s]44709 examples [00:51, 873.29 examples/s]44797 examples [00:51, 871.69 examples/s]44887 examples [00:51, 876.79 examples/s]44981 examples [00:51, 893.39 examples/s]45072 examples [00:51, 895.75 examples/s]45164 examples [00:52, 900.31 examples/s]45258 examples [00:52, 911.12 examples/s]45350 examples [00:52, 907.91 examples/s]45441 examples [00:52, 906.21 examples/s]45532 examples [00:52, 904.29 examples/s]45624 examples [00:52, 907.84 examples/s]45715 examples [00:52, 907.18 examples/s]45806 examples [00:52, 900.21 examples/s]45897 examples [00:52, 889.63 examples/s]45987 examples [00:52, 876.83 examples/s]46077 examples [00:53, 883.44 examples/s]46170 examples [00:53, 894.61 examples/s]46261 examples [00:53, 897.95 examples/s]46355 examples [00:53, 907.77 examples/s]46446 examples [00:53, 907.22 examples/s]46537 examples [00:53, 903.13 examples/s]46628 examples [00:53, 902.59 examples/s]46719 examples [00:53, 899.22 examples/s]46810 examples [00:53, 902.20 examples/s]46901 examples [00:53, 902.13 examples/s]46992 examples [00:54, 893.62 examples/s]47082 examples [00:54, 889.22 examples/s]47176 examples [00:54, 903.02 examples/s]47267 examples [00:54, 876.91 examples/s]47357 examples [00:54, 879.18 examples/s]47446 examples [00:54, 873.42 examples/s]47537 examples [00:54, 884.03 examples/s]47628 examples [00:54, 889.99 examples/s]47723 examples [00:54, 906.73 examples/s]47816 examples [00:55, 913.18 examples/s]47908 examples [00:55, 906.85 examples/s]47999 examples [00:55, 889.40 examples/s]48089 examples [00:55, 879.50 examples/s]48178 examples [00:55, 875.04 examples/s]48266 examples [00:55, 873.65 examples/s]48354 examples [00:55, 870.66 examples/s]48442 examples [00:55, 835.93 examples/s]48528 examples [00:55, 842.40 examples/s]48613 examples [00:55, 841.80 examples/s]48701 examples [00:56, 851.86 examples/s]48792 examples [00:56, 865.91 examples/s]48882 examples [00:56, 875.18 examples/s]48972 examples [00:56, 881.37 examples/s]49061 examples [00:56, 881.60 examples/s]49150 examples [00:56, 859.46 examples/s]49237 examples [00:56, 847.50 examples/s]49327 examples [00:56, 860.19 examples/s]49419 examples [00:56, 875.76 examples/s]49507 examples [00:56, 869.51 examples/s]49598 examples [00:57, 879.98 examples/s]49690 examples [00:57, 890.15 examples/s]49780 examples [00:57, 886.22 examples/s]49869 examples [00:57, 880.70 examples/s]49961 examples [00:57, 890.63 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5482/50000 [00:00<00:00, 54819.01 examples/s] 33%|███▎      | 16377/50000 [00:00<00:00, 64421.09 examples/s] 52%|█████▏    | 25915/50000 [00:00<00:00, 71370.26 examples/s] 75%|███████▌  | 37610/50000 [00:00<00:00, 80819.34 examples/s] 99%|█████████▊| 49313/50000 [00:00<00:00, 89087.74 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 765.63 examples/s]166 examples [00:00, 795.84 examples/s]258 examples [00:00, 829.14 examples/s]349 examples [00:00, 850.86 examples/s]442 examples [00:00, 871.39 examples/s]535 examples [00:00, 886.26 examples/s]628 examples [00:00, 898.21 examples/s]719 examples [00:00, 899.10 examples/s]811 examples [00:00, 903.20 examples/s]902 examples [00:01, 904.58 examples/s]997 examples [00:01, 916.58 examples/s]1091 examples [00:01, 921.59 examples/s]1183 examples [00:01, 916.08 examples/s]1278 examples [00:01, 925.71 examples/s]1375 examples [00:01, 935.90 examples/s]1470 examples [00:01, 939.78 examples/s]1564 examples [00:01, 870.42 examples/s]1655 examples [00:01, 881.62 examples/s]1744 examples [00:01, 882.55 examples/s]1837 examples [00:02, 893.88 examples/s]1930 examples [00:02, 902.07 examples/s]2026 examples [00:02, 906.12 examples/s]2120 examples [00:02, 913.53 examples/s]2213 examples [00:02, 917.21 examples/s]2305 examples [00:02, 896.33 examples/s]2399 examples [00:02, 906.15 examples/s]2490 examples [00:02, 874.92 examples/s]2581 examples [00:02, 884.09 examples/s]2674 examples [00:02, 896.95 examples/s]2768 examples [00:03, 908.59 examples/s]2860 examples [00:03, 902.05 examples/s]2951 examples [00:03, 903.89 examples/s]3043 examples [00:03, 907.21 examples/s]3134 examples [00:03, 902.89 examples/s]3225 examples [00:03, 888.96 examples/s]3318 examples [00:03, 900.74 examples/s]3413 examples [00:03, 914.28 examples/s]3505 examples [00:03, 915.45 examples/s]3597 examples [00:03, 897.99 examples/s]3691 examples [00:04, 908.86 examples/s]3784 examples [00:04, 913.45 examples/s]3876 examples [00:04, 908.08 examples/s]3970 examples [00:04, 915.32 examples/s]4064 examples [00:04, 920.46 examples/s]4157 examples [00:04, 917.97 examples/s]4252 examples [00:04, 926.35 examples/s]4348 examples [00:04, 935.22 examples/s]4442 examples [00:04, 917.74 examples/s]4534 examples [00:05, 911.38 examples/s]4626 examples [00:05, 890.78 examples/s]4716 examples [00:05, 864.84 examples/s]4803 examples [00:05, 852.30 examples/s]4889 examples [00:05, 852.36 examples/s]4979 examples [00:05, 865.88 examples/s]5074 examples [00:05, 889.11 examples/s]5167 examples [00:05, 898.91 examples/s]5258 examples [00:05, 891.55 examples/s]5348 examples [00:05, 859.37 examples/s]5435 examples [00:06, 859.85 examples/s]5522 examples [00:06, 841.98 examples/s]5607 examples [00:06, 826.37 examples/s]5690 examples [00:06, 822.61 examples/s]5773 examples [00:06, 821.19 examples/s]5856 examples [00:06, 823.77 examples/s]5947 examples [00:06, 846.37 examples/s]6037 examples [00:06, 859.00 examples/s]6132 examples [00:06, 882.54 examples/s]6226 examples [00:06, 897.60 examples/s]6317 examples [00:07, 880.20 examples/s]6410 examples [00:07, 892.70 examples/s]6500 examples [00:07, 885.26 examples/s]6589 examples [00:07, 883.21 examples/s]6678 examples [00:07, 876.04 examples/s]6766 examples [00:07, 869.99 examples/s]6854 examples [00:07, 870.71 examples/s]6942 examples [00:07, 871.13 examples/s]7030 examples [00:07, 858.54 examples/s]7119 examples [00:08, 865.83 examples/s]7213 examples [00:08, 885.71 examples/s]7305 examples [00:08, 893.69 examples/s]7400 examples [00:08, 909.22 examples/s]7492 examples [00:08, 911.92 examples/s]7584 examples [00:08, 910.57 examples/s]7676 examples [00:08, 910.05 examples/s]7769 examples [00:08, 915.35 examples/s]7866 examples [00:08, 930.78 examples/s]7960 examples [00:08, 926.99 examples/s]8055 examples [00:09, 931.06 examples/s]8149 examples [00:09, 893.98 examples/s]8239 examples [00:09, 878.93 examples/s]8334 examples [00:09, 897.53 examples/s]8425 examples [00:09, 896.96 examples/s]8518 examples [00:09, 903.36 examples/s]8609 examples [00:09, 900.53 examples/s]8700 examples [00:09, 901.32 examples/s]8791 examples [00:09, 819.28 examples/s]8882 examples [00:09, 832.53 examples/s]8968 examples [00:10, 837.42 examples/s]9058 examples [00:10, 852.85 examples/s]9147 examples [00:10, 861.12 examples/s]9238 examples [00:10, 875.07 examples/s]9329 examples [00:10, 882.64 examples/s]9422 examples [00:10, 895.58 examples/s]9514 examples [00:10, 900.66 examples/s]9605 examples [00:10, 903.02 examples/s]9696 examples [00:10, 895.56 examples/s]9786 examples [00:10, 893.32 examples/s]9881 examples [00:11, 908.02 examples/s]9972 examples [00:11, 893.07 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSW0NN7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSW0NN7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7faa87dba950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7faa87dba950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7faa87dba950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa11256c50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa1125a278>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7faa87dba950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7faa87dba950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7faa87dba950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa87da6ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7faa853fb668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fa9ff8ad6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fa9ff8ad6a8> 

  function with postional parmater data_info <function split_train_valid at 0x7fa9ff8ad6a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fa9ff8ad7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fa9ff8ad7b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fa9ff8ad7b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=77284794f06eb4fac4e49fda830872bf285a108d19bb7dabd3add5280c2664dc
  Stored in directory: /tmp/pip-ephem-wheel-cache-qdgyev8g/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:59:47, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:55:58, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:30:28, 22.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:21:49, 32.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:01<5:08:30, 46.4kB/s].vector_cache/glove.6B.zip:   1%|          | 6.39M/862M [00:01<3:35:22, 66.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.96M/862M [00:01<2:30:28, 94.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<1:44:58, 135kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:13:10, 192kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<51:06, 274kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<35:46, 390kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:01<25:04, 555kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:02<17:33, 788kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:02<12:21, 1.12MB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.8M/862M [00:02<08:44, 1.57MB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.1M/862M [00:02<06:11, 2.21MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<04:24, 3.08MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<03:09, 4.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<04:34, 2.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:06, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<07:03, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<05:46, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<04:13, 3.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<08:19, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<07:27, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<05:38, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:07<04:10, 3.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:08<08:47, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:59, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<06:01, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<04:26, 2.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<08:57, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<07:54, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<05:53, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:48, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:12<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:13<04:50, 2.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:03, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:52, 2.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<04:30, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:15<03:20, 3.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<10:34, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<08:46, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<06:42, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<04:53, 2.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<08:57, 1.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<07:53, 1.64MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:19<05:49, 2.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:19<04:17, 3.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<11:03, 1.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<09:22, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:21<06:57, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<07:27, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<06:48, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:22<05:15, 2.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<03:52, 3.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<08:06, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<07:16, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<05:25, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:23, 1.98MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:47, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:34, 2.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:21, 3.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:54, 1.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:29, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<06:15, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:34, 2.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:50, 1.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<09:49, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<07:11, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:10, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:11:17, 175kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<51:22, 242kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<36:14, 343kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<25:26, 487kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<4:23:50, 47.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<3:06:10, 66.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<2:10:28, 94.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:33:32, 132kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:07:21, 183kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<47:50, 257kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<33:39, 365kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<26:50, 456kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<20:18, 603kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<14:34, 839kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:39, 963kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:21, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:36, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<05:30, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:36, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:35, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:00, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:12, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:19, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:53, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:14, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:32, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:56, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:53, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:38, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:45, 2.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:39, 3.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<02:45, 4.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:10, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:35, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:03, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<05:09, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:19, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:26, 1.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:37, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<04:04, 2.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<27:19, 428kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<20:36, 568kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<14:46, 790kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<10:26, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<50:15, 231kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<36:38, 317kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<25:56, 447kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<18:16, 633kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<22:03, 524kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<16:51, 685kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<12:06, 953kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<08:34, 1.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<2:57:41, 64.7kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<2:05:45, 91.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<1:28:12, 130kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<1:03:51, 179kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<46:03, 248kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<32:31, 350kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<25:00, 454kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<18:54, 600kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<13:30, 838kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<09:35, 1.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<16:48, 671kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<12:55, 872kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:34, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<06:49, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<11:28, 978kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:22, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<06:53, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:05, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:35, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:10, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:50, 2.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:44, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:18, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:01, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:09, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:53, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:40, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<02:44, 3.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:50, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:17, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:47, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:40, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:43, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:02, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:33, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:23, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:17, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:32, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<03:29, 3.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<02:36, 4.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<09:22, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:07, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<06:01, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:26, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:34, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:32, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:41, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:35, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:36, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<07:31, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:30, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:18, 1.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:08, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:01, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:54, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:33, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:51, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:39, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:33, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<06:21, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:28, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:15, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:18, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<06:02, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<06:17, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:57, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:14, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:00, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<05:54, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<06:05, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<06:03, 1.74MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:56, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:57, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<06:00, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<05:51, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:55, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:51, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:58, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:48, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<06:02, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:42, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:50, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:35, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:42, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:29, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:41, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:17, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<05:40, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<05:14, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<05:44, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:18, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:39, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:13, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:45, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:28, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:06, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:10, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<04:57, 2.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<05:03, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:52, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:56, 2.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:43, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:39, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<04:33, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<04:43, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:28, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:40, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:30, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:38, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:26, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:34, 2.28MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:18, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:32, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:14, 2.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:30, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:16, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:37, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:50, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:40, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:23, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:22, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<04:07, 2.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<04:07, 2.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:54, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:54, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:44, 2.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:45, 2.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:05, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<03:36, 2.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<03:58, 2.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<03:43, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<03:21, 3.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<03:28, 2.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:09, 3.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:17, 3.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:06, 3.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:02, 3.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<02:58, 3.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<02:50, 3.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<02:46, 3.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<09:14, 1.11MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:54, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:00, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:19, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:52, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:40, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:27, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:42, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:39, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:55, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:14, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:40, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:18, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<02:52, 3.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:17, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:48, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:23, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:15, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:27, 2.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<02:50, 3.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<02:29, 4.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:16, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:34, 1.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:43, 1.76MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:27, 2.92MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<02:42, 3.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<7:44:31, 21.7kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<5:28:51, 30.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<3:50:56, 43.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<2:41:42, 62.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<1:53:14, 88.4kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<1:22:58, 120kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<1:01:09, 163kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<43:32, 229kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<30:38, 325kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<21:48, 456kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<15:28, 641kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<40:44, 244kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<34:38, 286kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<25:43, 385kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<18:22, 539kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<13:08, 752kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<09:26, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:24, 735kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<11:35, 849kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:36, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<06:23, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<04:39, 2.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:52, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<09:37, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:45, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:38, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:50, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<08:13, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:39, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<03:33, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<2:55:43, 54.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<2:06:11, 76.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<1:28:56, 108kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<1:02:28, 154kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<43:42, 219kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<40:15, 238kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<30:58, 309kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<22:15, 430kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<15:52, 601kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<12:52, 738kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:37, 817kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:46, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<06:17, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:59, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:57, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:05, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:26, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:32, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:06, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:49, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:30, 2.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:57, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:01, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:08, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:32, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:09, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:30, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:09, 2.91MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:01, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:21, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:02, 2.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:41, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:06, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:30, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:03, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<02:58, 3.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<22:14, 405kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<17:48, 506kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<12:57, 695kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<09:15, 971kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<08:36, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:26, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:31, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:00, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:24, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:06, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:36, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:26, 2.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:40, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<03:59, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:01, 2.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:00, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:41, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:39, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:43, 3.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:25, 1.96MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:58, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:54, 2.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:52, 3.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:53, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:32, 1.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<02:48, 3.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:27, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:15, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:16, 2.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:25, 3.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:45, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:33, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:12, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:07, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:27, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:13, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<04:33, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:32, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<02:53, 2.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:39, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:21, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:27, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:14, 2.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:26, 3.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<14:23, 574kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<12:44, 649kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:33, 863kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:49, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<07:18, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<07:31, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:54, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:35, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<06:26, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:01, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:45, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:11, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:13, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:06, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:11, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:20, 3.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:14, 970kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:58, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<06:04, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:24, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:54, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:37, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:27, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:14, 2.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:43, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:36, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:28, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:15, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:31, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:35, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:52, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<02:39, 2.90MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:24, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:54, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:34, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:29, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:31, 3.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<1:03:09, 121kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<46:06, 166kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<32:43, 233kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<23:09, 329kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<16:26, 462kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<11:43, 646kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<14:26, 524kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<12:30, 605kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<09:13, 819kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:46, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:02, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:44, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:55, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:30, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:07, 1.46MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:45, 1.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:53, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:16, 3.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<08:45, 849kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<10:11, 730kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<08:14, 902kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<05:59, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:24, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:20, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<05:37, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:57, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:41, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:27, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:38, 2.77MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:00, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:38, 954kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:11, 1.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:51, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:35, 2.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:42, 2.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:06, 3.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<2:54:49, 41.3kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<2:04:12, 58.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<1:27:19, 82.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<1:01:10, 118kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<42:54, 167kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<32:44, 219kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<24:53, 287kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<17:55, 399kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<12:46, 558kB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:04<09:05, 782kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<06:32, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<13:38, 519kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<13:31, 524kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<10:28, 675kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<07:35, 931kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<05:26, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:00, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<11:20, 619kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:53, 709kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<07:25, 945kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<05:26, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<03:59, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:57, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<10:56, 635kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<11:27, 606kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:48, 788kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:33, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:44, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:31, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:31, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:40, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:26, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:30, 2.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<01:57, 3.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<17:16, 394kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<17:01, 400kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<13:15, 513kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<09:35, 709kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<06:54, 982kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<05:05, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:46, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<09:52, 683kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<09:17, 726kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<07:07, 946kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:10, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:54, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:42, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:22, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:56, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:48, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:52, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:13, 2.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<01:45, 3.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<18:10, 363kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<15:03, 438kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<11:05, 594kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<07:55, 829kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<05:47, 1.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<04:12, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<10:26, 625kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<09:19, 700kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<07:04, 923kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:08, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:45, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:51, 2.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<09:56, 650kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<11:31, 561kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:59, 718kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:35, 978kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:50, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:35, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:41, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<09:13, 694kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 479M/862M [03:26<08:34, 746kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:26, 992kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:46, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:31, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:41, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:37, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:19, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:09, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:06, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:22, 2.65MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<01:50, 3.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:22, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:41, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:29, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:20, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:28, 2.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:00, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<10:30, 589kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<09:25, 657kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<07:01, 879kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:12, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:47, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:53, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:12, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<2:30:45, 40.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<1:47:35, 56.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<1:15:37, 80.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<53:09, 115kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<37:13, 163kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<26:15, 231kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<21:01, 288kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<16:46, 361kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<12:12, 495kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<08:53, 678kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<06:25, 937kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<04:45, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<03:31, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<05:44, 1.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<05:59, 997kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:36, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:23, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:34, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:00, 2.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:40, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<07:26, 795kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:07, 965kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:31, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:19, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<19:01, 307kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<15:07, 386kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<11:02, 528kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<07:52, 738kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<05:42, 1.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<04:08, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<08:46, 658kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<09:43, 594kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:41, 750kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:39, 1.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:08, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<03:02, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:17, 2.50MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<26:17, 217kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<20:04, 284kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<14:27, 394kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<10:14, 554kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<07:20, 771kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<05:15, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<1:39:24, 56.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<1:12:39, 77.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<51:40, 109kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<36:17, 155kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<25:26, 220kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<17:55, 311kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<17:04, 326kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<13:28, 413kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<09:47, 567kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:00, 791kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<05:03, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<03:41, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<07:55, 694kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<08:34, 641kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:37, 830kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:56, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:34, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:39, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:24, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:52, 925kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:04, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:56, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:00, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:18, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:44, 3.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:57, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:02, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:06, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:17, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:43, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:24, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<05:30, 960kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:34, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:28, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:32, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:52, 2.79MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<07:48, 670kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<07:40, 681kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:54, 884kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:16, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:03, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:58, 862kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<06:07, 840kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:47, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:28, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:32, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:24, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:52, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:46, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:51, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:04, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:35, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:10, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:16, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:29, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:48, 2.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:14, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:22, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:27, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:58, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:34, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:16, 1.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:33, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<01:55, 2.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<01:24, 3.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:24, 650kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<06:26, 746kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:49, 993kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:26, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:12, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:11, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:11, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:19, 2.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:46, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:14, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:34, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:51, 2.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:48, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:26, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:42, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:05, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:30, 3.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:25, 610kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:12, 729kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:33, 991kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:26, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:19, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:31, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:48, 2.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:36, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:23, 1.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:34, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:50, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:31, 662kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:24, 798kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:56, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:50, 1.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:08, 1.35MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:10, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:27, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:46, 2.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:44, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:52, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:14, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:36, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:42, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:49, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:12, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:34, 2.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:49, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:43, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:48, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<02:01, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:35, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:49, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:11, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<01:12, 3.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<13:03, 299kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<10:05, 387kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<07:14, 537kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<05:04, 760kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<05:24, 710kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:43, 812kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:29, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:32, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:33, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:41, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:06, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:31, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:12, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:25, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:52, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:24, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:44, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:01, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:37, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:10, 3.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:23, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:28, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:55, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:26, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:03, 3.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:01, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:58, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:16, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:40, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:53, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:05, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:37, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:11, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:45, 1.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:43, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:19, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:31, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:02, 3.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:28, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:40, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:21, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:01, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:27, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:38, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:18, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:24, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:37, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:16, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:22, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:33, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:12, 2.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:53, 3.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:36, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:43, 1.70MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:20, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<00:57, 2.99MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:12, 893kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:49, 1.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:06, 1.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<01:29, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:08, 889kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:45, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:03, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:26, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<04:48, 569kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:49, 712kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:49, 964kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:01, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:01, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:56, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:29, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:04, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:40, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:41, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:18, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:18, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:24, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:06, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:47, 3.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:35, 683kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:00, 814kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:13, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<01:33, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<05:25, 440kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<04:16, 559kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<03:05, 767kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<02:08, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<04:57, 469kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:55, 591kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:50, 811kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:58, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<05:08, 438kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:02, 557kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:55, 765kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<02:03, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:17, 953kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:01, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:31, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:03, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:32, 596kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:53, 728kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:06, 993kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:29, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:45, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:38, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:14, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:14, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:15, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:57, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:42, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:02, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:05, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<00:36, 3.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:28, 529kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:47, 657kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:02, 895kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:25, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:48, 980kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:41, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:27, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:06, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:48, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:58, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:08, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:39, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:57, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:07, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:52, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:38, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:46, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:58, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:47, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:33, 2.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:01, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:04, 1.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:37, 2.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:42, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:40, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:28, 2.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:00, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:02, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:34, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:04, 1.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:04, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:49, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:35, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:47, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:27, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:34, 1.97MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:39, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:22, 2.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:02, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:58, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:43, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:34, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:20, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:33, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:36, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:20, 2.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:22, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:16, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:26, 1.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:20, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:14, 2.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:28, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:22, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:35, 873kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:31, 982kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:23, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:15, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:29, 929kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:26, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:19, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:12, 1.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:16, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:16, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:12, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:13, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:05, 2.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 3.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 819kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 940kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.26MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<79:14:05,  1.40it/s]  0%|          | 728/400000 [00:00<55:22:05,  2.00it/s]  0%|          | 1460/400000 [00:00<38:41:28,  2.86it/s]  1%|          | 2207/400000 [00:01<27:02:15,  4.09it/s]  1%|          | 2905/400000 [00:01<18:53:52,  5.84it/s]  1%|          | 3605/400000 [00:01<13:12:35,  8.34it/s]  1%|          | 4292/400000 [00:01<9:14:08, 11.90it/s]   1%|▏         | 5034/400000 [00:01<6:27:26, 16.99it/s]  1%|▏         | 5741/400000 [00:01<4:30:59, 24.25it/s]  2%|▏         | 6461/400000 [00:01<3:09:37, 34.59it/s]  2%|▏         | 7148/400000 [00:01<2:12:48, 49.30it/s]  2%|▏         | 7874/400000 [00:01<1:33:03, 70.22it/s]  2%|▏         | 8608/400000 [00:01<1:05:17, 99.91it/s]  2%|▏         | 9348/400000 [00:02<45:52, 141.91it/s]   3%|▎         | 10099/400000 [00:02<32:18, 201.10it/s]  3%|▎         | 10834/400000 [00:02<22:50, 283.95it/s]  3%|▎         | 11565/400000 [00:02<16:13, 399.00it/s]  3%|▎         | 12294/400000 [00:02<11:36, 556.72it/s]  3%|▎         | 13030/400000 [00:02<08:22, 770.33it/s]  3%|▎         | 13759/400000 [00:02<06:07, 1052.09it/s]  4%|▎         | 14485/400000 [00:02<04:32, 1413.44it/s]  4%|▍         | 15207/400000 [00:02<03:26, 1859.64it/s]  4%|▍         | 15953/400000 [00:02<02:40, 2399.67it/s]  4%|▍         | 16689/400000 [00:03<02:07, 3007.43it/s]  4%|▍         | 17426/400000 [00:03<01:44, 3656.15it/s]  5%|▍         | 18157/400000 [00:03<01:28, 4300.61it/s]  5%|▍         | 18888/400000 [00:03<01:19, 4823.23it/s]  5%|▍         | 19625/400000 [00:03<01:10, 5379.93it/s]  5%|▌         | 20346/400000 [00:03<01:06, 5749.41it/s]  5%|▌         | 21067/400000 [00:03<01:01, 6120.10it/s]  5%|▌         | 21788/400000 [00:03<00:59, 6395.10it/s]  6%|▌         | 22513/400000 [00:03<00:56, 6627.70it/s]  6%|▌         | 23238/400000 [00:03<00:55, 6802.46it/s]  6%|▌         | 23958/400000 [00:04<00:54, 6886.78it/s]  6%|▌         | 24706/400000 [00:04<00:53, 7054.34it/s]  6%|▋         | 25448/400000 [00:04<00:52, 7157.96it/s]  7%|▋         | 26186/400000 [00:04<00:51, 7220.85it/s]  7%|▋         | 26938/400000 [00:04<00:51, 7305.59it/s]  7%|▋         | 27680/400000 [00:04<00:50, 7337.26it/s]  7%|▋         | 28419/400000 [00:04<00:50, 7301.97it/s]  7%|▋         | 29153/400000 [00:04<00:50, 7280.14it/s]  7%|▋         | 29884/400000 [00:04<00:50, 7265.54it/s]  8%|▊         | 30623/400000 [00:04<00:50, 7301.19it/s]  8%|▊         | 31358/400000 [00:05<00:50, 7313.35it/s]  8%|▊         | 32091/400000 [00:05<00:51, 7076.56it/s]  8%|▊         | 32801/400000 [00:05<00:51, 7061.91it/s]  8%|▊         | 33509/400000 [00:05<00:52, 6929.81it/s]  9%|▊         | 34252/400000 [00:05<00:51, 7071.34it/s]  9%|▊         | 34962/400000 [00:05<00:52, 6937.43it/s]  9%|▉         | 35658/400000 [00:05<00:53, 6856.26it/s]  9%|▉         | 36346/400000 [00:05<00:53, 6742.02it/s]  9%|▉         | 37043/400000 [00:05<00:53, 6807.01it/s]  9%|▉         | 37790/400000 [00:05<00:51, 6991.88it/s] 10%|▉         | 38537/400000 [00:06<00:50, 7126.99it/s] 10%|▉         | 39296/400000 [00:06<00:49, 7258.39it/s] 10%|█         | 40042/400000 [00:06<00:49, 7316.20it/s] 10%|█         | 40776/400000 [00:06<00:49, 7197.08it/s] 10%|█         | 41504/400000 [00:06<00:49, 7219.61it/s] 11%|█         | 42244/400000 [00:06<00:49, 7272.74it/s] 11%|█         | 42992/400000 [00:06<00:48, 7332.55it/s] 11%|█         | 43726/400000 [00:06<00:48, 7318.44it/s] 11%|█         | 44459/400000 [00:06<00:48, 7304.82it/s] 11%|█▏        | 45196/400000 [00:06<00:48, 7322.72it/s] 11%|█▏        | 45943/400000 [00:07<00:48, 7364.80it/s] 12%|█▏        | 46692/400000 [00:07<00:47, 7400.19it/s] 12%|█▏        | 47435/400000 [00:07<00:47, 7408.86it/s] 12%|█▏        | 48177/400000 [00:07<00:47, 7363.92it/s] 12%|█▏        | 48914/400000 [00:07<00:47, 7362.39it/s] 12%|█▏        | 49651/400000 [00:07<00:47, 7359.09it/s] 13%|█▎        | 50396/400000 [00:07<00:47, 7384.26it/s] 13%|█▎        | 51148/400000 [00:07<00:46, 7423.34it/s] 13%|█▎        | 51891/400000 [00:07<00:47, 7345.10it/s] 13%|█▎        | 52646/400000 [00:08<00:46, 7402.94it/s] 13%|█▎        | 53405/400000 [00:08<00:46, 7457.53it/s] 14%|█▎        | 54163/400000 [00:08<00:46, 7491.51it/s] 14%|█▎        | 54926/400000 [00:08<00:45, 7532.19it/s] 14%|█▍        | 55680/400000 [00:08<00:46, 7475.42it/s] 14%|█▍        | 56439/400000 [00:08<00:45, 7509.26it/s] 14%|█▍        | 57191/400000 [00:08<00:46, 7405.33it/s] 14%|█▍        | 57933/400000 [00:08<00:47, 7160.03it/s] 15%|█▍        | 58652/400000 [00:08<00:48, 7016.99it/s] 15%|█▍        | 59356/400000 [00:08<00:48, 6969.63it/s] 15%|█▌        | 60100/400000 [00:09<00:47, 7103.70it/s] 15%|█▌        | 60852/400000 [00:09<00:46, 7221.65it/s] 15%|█▌        | 61577/400000 [00:09<00:46, 7229.64it/s] 16%|█▌        | 62302/400000 [00:09<00:47, 7081.26it/s] 16%|█▌        | 63055/400000 [00:09<00:46, 7208.51it/s] 16%|█▌        | 63804/400000 [00:09<00:46, 7290.56it/s] 16%|█▌        | 64535/400000 [00:09<00:46, 7162.70it/s] 16%|█▋        | 65273/400000 [00:09<00:46, 7222.95it/s] 17%|█▋        | 66020/400000 [00:09<00:45, 7293.85it/s] 17%|█▋        | 66751/400000 [00:09<00:46, 7229.97it/s] 17%|█▋        | 67494/400000 [00:10<00:45, 7286.34it/s] 17%|█▋        | 68234/400000 [00:10<00:45, 7318.13it/s] 17%|█▋        | 68969/400000 [00:10<00:45, 7326.62it/s] 17%|█▋        | 69721/400000 [00:10<00:44, 7382.02it/s] 18%|█▊        | 70460/400000 [00:10<00:44, 7326.57it/s] 18%|█▊        | 71193/400000 [00:10<00:45, 7248.08it/s] 18%|█▊        | 71919/400000 [00:10<00:45, 7160.12it/s] 18%|█▊        | 72669/400000 [00:10<00:45, 7257.55it/s] 18%|█▊        | 73403/400000 [00:10<00:44, 7279.86it/s] 19%|█▊        | 74132/400000 [00:10<00:46, 7079.72it/s] 19%|█▊        | 74891/400000 [00:11<00:45, 7223.15it/s] 19%|█▉        | 75644/400000 [00:11<00:44, 7311.77it/s] 19%|█▉        | 76377/400000 [00:11<00:44, 7309.22it/s] 19%|█▉        | 77129/400000 [00:11<00:43, 7370.38it/s] 19%|█▉        | 77867/400000 [00:11<00:44, 7282.17it/s] 20%|█▉        | 78597/400000 [00:11<00:45, 7107.16it/s] 20%|█▉        | 79310/400000 [00:11<00:47, 6768.73it/s] 20%|██        | 80055/400000 [00:11<00:45, 6958.59it/s] 20%|██        | 80781/400000 [00:11<00:45, 7023.41it/s] 20%|██        | 81487/400000 [00:12<00:46, 6831.01it/s] 21%|██        | 82174/400000 [00:12<00:47, 6735.86it/s] 21%|██        | 82851/400000 [00:12<00:47, 6669.93it/s] 21%|██        | 83556/400000 [00:12<00:46, 6778.45it/s] 21%|██        | 84301/400000 [00:12<00:45, 6965.50it/s] 21%|██▏       | 85049/400000 [00:12<00:44, 7111.93it/s] 21%|██▏       | 85782/400000 [00:12<00:43, 7174.00it/s] 22%|██▏       | 86538/400000 [00:12<00:43, 7285.41it/s] 22%|██▏       | 87301/400000 [00:12<00:42, 7383.75it/s] 22%|██▏       | 88049/400000 [00:12<00:42, 7410.37it/s] 22%|██▏       | 88792/400000 [00:13<00:44, 7064.15it/s] 22%|██▏       | 89503/400000 [00:13<00:44, 6939.45it/s] 23%|██▎       | 90236/400000 [00:13<00:43, 7044.25it/s] 23%|██▎       | 90952/400000 [00:13<00:43, 7076.15it/s] 23%|██▎       | 91677/400000 [00:13<00:43, 7126.23it/s] 23%|██▎       | 92392/400000 [00:13<00:43, 7009.05it/s] 23%|██▎       | 93095/400000 [00:13<00:43, 6980.66it/s] 23%|██▎       | 93795/400000 [00:13<00:44, 6906.86it/s] 24%|██▎       | 94545/400000 [00:13<00:43, 7072.86it/s] 24%|██▍       | 95283/400000 [00:13<00:42, 7161.48it/s] 24%|██▍       | 96001/400000 [00:14<00:42, 7096.99it/s] 24%|██▍       | 96712/400000 [00:14<00:43, 7036.98it/s] 24%|██▍       | 97417/400000 [00:14<00:43, 7019.67it/s] 25%|██▍       | 98189/400000 [00:14<00:41, 7214.71it/s] 25%|██▍       | 98929/400000 [00:14<00:41, 7269.10it/s] 25%|██▍       | 99680/400000 [00:14<00:40, 7339.51it/s] 25%|██▌       | 100416/400000 [00:14<00:41, 7139.56it/s] 25%|██▌       | 101132/400000 [00:14<00:42, 6989.03it/s] 25%|██▌       | 101875/400000 [00:14<00:41, 7113.97it/s] 26%|██▌       | 102611/400000 [00:14<00:41, 7185.52it/s] 26%|██▌       | 103367/400000 [00:15<00:40, 7291.57it/s] 26%|██▌       | 104109/400000 [00:15<00:40, 7329.21it/s] 26%|██▌       | 104865/400000 [00:15<00:39, 7395.33it/s] 26%|██▋       | 105618/400000 [00:15<00:39, 7434.53it/s] 27%|██▋       | 106363/400000 [00:15<00:39, 7390.95it/s] 27%|██▋       | 107103/400000 [00:15<00:39, 7338.95it/s] 27%|██▋       | 107838/400000 [00:15<00:40, 7257.32it/s] 27%|██▋       | 108577/400000 [00:15<00:39, 7295.13it/s] 27%|██▋       | 109310/400000 [00:15<00:39, 7304.87it/s] 28%|██▊       | 110050/400000 [00:15<00:39, 7332.91it/s] 28%|██▊       | 110801/400000 [00:16<00:39, 7382.98it/s] 28%|██▊       | 111558/400000 [00:16<00:38, 7435.94it/s] 28%|██▊       | 112319/400000 [00:16<00:38, 7484.80it/s] 28%|██▊       | 113068/400000 [00:16<00:39, 7306.93it/s] 28%|██▊       | 113810/400000 [00:16<00:38, 7339.68it/s] 29%|██▊       | 114545/400000 [00:16<00:39, 7174.63it/s] 29%|██▉       | 115283/400000 [00:16<00:39, 7234.91it/s] 29%|██▉       | 116035/400000 [00:16<00:38, 7317.59it/s] 29%|██▉       | 116768/400000 [00:16<00:38, 7301.56it/s] 29%|██▉       | 117499/400000 [00:17<00:38, 7261.29it/s] 30%|██▉       | 118254/400000 [00:17<00:38, 7344.65it/s] 30%|██▉       | 119008/400000 [00:17<00:37, 7400.95it/s] 30%|██▉       | 119775/400000 [00:17<00:37, 7477.54it/s] 30%|███       | 120534/400000 [00:17<00:37, 7509.52it/s] 30%|███       | 121286/400000 [00:17<00:37, 7412.78it/s] 31%|███       | 122028/400000 [00:17<00:39, 7110.65it/s] 31%|███       | 122759/400000 [00:17<00:38, 7167.93it/s] 31%|███       | 123492/400000 [00:17<00:38, 7215.32it/s] 31%|███       | 124216/400000 [00:17<00:38, 7185.34it/s] 31%|███       | 124943/400000 [00:18<00:38, 7208.02it/s] 31%|███▏      | 125689/400000 [00:18<00:37, 7280.18it/s] 32%|███▏      | 126443/400000 [00:18<00:37, 7351.49it/s] 32%|███▏      | 127206/400000 [00:18<00:36, 7432.78it/s] 32%|███▏      | 127950/400000 [00:18<00:36, 7419.83it/s] 32%|███▏      | 128693/400000 [00:18<00:36, 7352.49it/s] 32%|███▏      | 129429/400000 [00:18<00:37, 7276.43it/s] 33%|███▎      | 130158/400000 [00:18<00:37, 7114.38it/s] 33%|███▎      | 130871/400000 [00:18<00:38, 7057.26it/s] 33%|███▎      | 131607/400000 [00:18<00:37, 7145.42it/s] 33%|███▎      | 132335/400000 [00:19<00:37, 7184.02it/s] 33%|███▎      | 133079/400000 [00:19<00:36, 7257.25it/s] 33%|███▎      | 133806/400000 [00:19<00:37, 7174.24it/s] 34%|███▎      | 134539/400000 [00:19<00:36, 7219.33it/s] 34%|███▍      | 135277/400000 [00:19<00:36, 7265.72it/s] 34%|███▍      | 136025/400000 [00:19<00:36, 7328.00it/s] 34%|███▍      | 136759/400000 [00:19<00:36, 7305.39it/s] 34%|███▍      | 137492/400000 [00:19<00:35, 7311.74it/s] 35%|███▍      | 138224/400000 [00:19<00:36, 7076.93it/s] 35%|███▍      | 138975/400000 [00:19<00:36, 7199.44it/s] 35%|███▍      | 139697/400000 [00:20<00:36, 7103.40it/s] 35%|███▌      | 140451/400000 [00:20<00:35, 7227.11it/s] 35%|███▌      | 141213/400000 [00:20<00:35, 7339.46it/s] 35%|███▌      | 141953/400000 [00:20<00:35, 7354.74it/s] 36%|███▌      | 142698/400000 [00:20<00:34, 7382.08it/s] 36%|███▌      | 143437/400000 [00:20<00:34, 7364.06it/s] 36%|███▌      | 144174/400000 [00:20<00:34, 7355.98it/s] 36%|███▌      | 144915/400000 [00:20<00:34, 7371.35it/s] 36%|███▋      | 145653/400000 [00:20<00:34, 7368.70it/s] 37%|███▋      | 146391/400000 [00:20<00:34, 7282.19it/s] 37%|███▋      | 147129/400000 [00:21<00:34, 7309.73it/s] 37%|███▋      | 147861/400000 [00:21<00:34, 7282.70it/s] 37%|███▋      | 148590/400000 [00:21<00:34, 7241.90it/s] 37%|███▋      | 149315/400000 [00:21<00:36, 6805.89it/s] 38%|███▊      | 150002/400000 [00:21<00:36, 6782.74it/s] 38%|███▊      | 150685/400000 [00:21<00:39, 6342.93it/s] 38%|███▊      | 151328/400000 [00:21<00:39, 6282.69it/s] 38%|███▊      | 151963/400000 [00:21<00:41, 6016.25it/s] 38%|███▊      | 152711/400000 [00:21<00:38, 6389.72it/s] 38%|███▊      | 153441/400000 [00:22<00:37, 6637.23it/s] 39%|███▊      | 154183/400000 [00:22<00:35, 6853.00it/s] 39%|███▊      | 154922/400000 [00:22<00:34, 7005.34it/s] 39%|███▉      | 155666/400000 [00:22<00:34, 7128.20it/s] 39%|███▉      | 156417/400000 [00:22<00:33, 7235.88it/s] 39%|███▉      | 157176/400000 [00:22<00:33, 7336.91it/s] 39%|███▉      | 157913/400000 [00:22<00:33, 7311.59it/s] 40%|███▉      | 158649/400000 [00:22<00:32, 7325.27it/s] 40%|███▉      | 159393/400000 [00:22<00:32, 7356.95it/s] 40%|████      | 160130/400000 [00:22<00:33, 7256.33it/s] 40%|████      | 160877/400000 [00:23<00:32, 7318.26it/s] 40%|████      | 161610/400000 [00:23<00:32, 7312.20it/s] 41%|████      | 162366/400000 [00:23<00:32, 7383.78it/s] 41%|████      | 163105/400000 [00:23<00:32, 7281.05it/s] 41%|████      | 163844/400000 [00:23<00:32, 7311.44it/s] 41%|████      | 164584/400000 [00:23<00:32, 7335.10it/s] 41%|████▏     | 165322/400000 [00:23<00:31, 7348.20it/s] 42%|████▏     | 166072/400000 [00:23<00:31, 7391.19it/s] 42%|████▏     | 166812/400000 [00:23<00:31, 7289.51it/s] 42%|████▏     | 167542/400000 [00:23<00:32, 7254.15it/s] 42%|████▏     | 168294/400000 [00:24<00:31, 7330.21it/s] 42%|████▏     | 169053/400000 [00:24<00:31, 7403.84it/s] 42%|████▏     | 169794/400000 [00:24<00:31, 7276.61it/s] 43%|████▎     | 170532/400000 [00:24<00:31, 7306.78it/s] 43%|████▎     | 171286/400000 [00:24<00:31, 7374.06it/s] 43%|████▎     | 172024/400000 [00:24<00:31, 7195.02it/s] 43%|████▎     | 172745/400000 [00:24<00:32, 7100.35it/s] 43%|████▎     | 173457/400000 [00:24<00:32, 7018.32it/s] 44%|████▎     | 174185/400000 [00:24<00:31, 7093.99it/s] 44%|████▎     | 174925/400000 [00:24<00:31, 7181.49it/s] 44%|████▍     | 175661/400000 [00:25<00:31, 7234.08it/s] 44%|████▍     | 176394/400000 [00:25<00:30, 7261.20it/s] 44%|████▍     | 177130/400000 [00:25<00:30, 7289.07it/s] 44%|████▍     | 177880/400000 [00:25<00:30, 7349.94it/s] 45%|████▍     | 178622/400000 [00:25<00:30, 7368.63it/s] 45%|████▍     | 179366/400000 [00:25<00:29, 7388.90it/s] 45%|████▌     | 180106/400000 [00:25<00:30, 7176.07it/s] 45%|████▌     | 180826/400000 [00:25<00:31, 7022.28it/s] 45%|████▌     | 181531/400000 [00:25<00:31, 6859.08it/s] 46%|████▌     | 182219/400000 [00:26<00:32, 6650.49it/s] 46%|████▌     | 182887/400000 [00:26<00:32, 6643.74it/s] 46%|████▌     | 183563/400000 [00:26<00:32, 6677.67it/s] 46%|████▌     | 184316/400000 [00:26<00:31, 6911.60it/s] 46%|████▋     | 185071/400000 [00:26<00:30, 7090.18it/s] 46%|████▋     | 185784/400000 [00:26<00:30, 7014.10it/s] 47%|████▋     | 186496/400000 [00:26<00:30, 7044.38it/s] 47%|████▋     | 187203/400000 [00:26<00:30, 7049.05it/s] 47%|████▋     | 187951/400000 [00:26<00:29, 7171.09it/s] 47%|████▋     | 188697/400000 [00:26<00:29, 7252.95it/s] 47%|████▋     | 189429/400000 [00:27<00:28, 7272.38it/s] 48%|████▊     | 190167/400000 [00:27<00:28, 7301.83it/s] 48%|████▊     | 190898/400000 [00:27<00:28, 7276.64it/s] 48%|████▊     | 191638/400000 [00:27<00:28, 7309.79it/s] 48%|████▊     | 192370/400000 [00:27<00:28, 7225.90it/s] 48%|████▊     | 193125/400000 [00:27<00:28, 7319.16it/s] 48%|████▊     | 193863/400000 [00:27<00:28, 7336.32it/s] 49%|████▊     | 194598/400000 [00:27<00:28, 7191.40it/s] 49%|████▉     | 195332/400000 [00:27<00:28, 7233.86it/s] 49%|████▉     | 196061/400000 [00:27<00:28, 7250.57it/s] 49%|████▉     | 196808/400000 [00:28<00:27, 7314.16it/s] 49%|████▉     | 197556/400000 [00:28<00:27, 7361.25it/s] 50%|████▉     | 198293/400000 [00:28<00:27, 7214.30it/s] 50%|████▉     | 199041/400000 [00:28<00:27, 7289.97it/s] 50%|████▉     | 199771/400000 [00:28<00:28, 7111.59it/s] 50%|█████     | 200484/400000 [00:28<00:28, 6956.44it/s] 50%|█████     | 201182/400000 [00:28<00:28, 6903.06it/s] 50%|█████     | 201874/400000 [00:28<00:28, 6880.37it/s] 51%|█████     | 202598/400000 [00:28<00:28, 6983.60it/s] 51%|█████     | 203352/400000 [00:28<00:27, 7140.50it/s] 51%|█████     | 204105/400000 [00:29<00:27, 7251.90it/s] 51%|█████     | 204849/400000 [00:29<00:26, 7306.41it/s] 51%|█████▏    | 205581/400000 [00:29<00:26, 7256.20it/s] 52%|█████▏    | 206329/400000 [00:29<00:26, 7320.43it/s] 52%|█████▏    | 207062/400000 [00:29<00:26, 7252.62it/s] 52%|█████▏    | 207788/400000 [00:29<00:27, 6927.07it/s] 52%|█████▏    | 208540/400000 [00:29<00:26, 7092.30it/s] 52%|█████▏    | 209253/400000 [00:29<00:27, 7061.29it/s] 53%|█████▎    | 210008/400000 [00:29<00:26, 7199.37it/s] 53%|█████▎    | 210756/400000 [00:29<00:25, 7279.60it/s] 53%|█████▎    | 211502/400000 [00:30<00:25, 7332.46it/s] 53%|█████▎    | 212251/400000 [00:30<00:25, 7359.40it/s] 53%|█████▎    | 212988/400000 [00:30<00:26, 7102.05it/s] 53%|█████▎    | 213701/400000 [00:30<00:26, 6973.72it/s] 54%|█████▎    | 214401/400000 [00:30<00:27, 6833.94it/s] 54%|█████▍    | 215099/400000 [00:30<00:26, 6876.46it/s] 54%|█████▍    | 215855/400000 [00:30<00:26, 7067.39it/s] 54%|█████▍    | 216597/400000 [00:30<00:25, 7168.72it/s] 54%|█████▍    | 217316/400000 [00:30<00:25, 7088.98it/s] 55%|█████▍    | 218067/400000 [00:31<00:25, 7209.61it/s] 55%|█████▍    | 218808/400000 [00:31<00:24, 7267.80it/s] 55%|█████▍    | 219544/400000 [00:31<00:24, 7246.37it/s] 55%|█████▌    | 220270/400000 [00:31<00:24, 7242.12it/s] 55%|█████▌    | 221016/400000 [00:31<00:24, 7303.87it/s] 55%|█████▌    | 221747/400000 [00:31<00:24, 7272.01it/s] 56%|█████▌    | 222475/400000 [00:31<00:24, 7220.15it/s] 56%|█████▌    | 223206/400000 [00:31<00:24, 7245.34it/s] 56%|█████▌    | 223931/400000 [00:31<00:24, 7090.81it/s] 56%|█████▌    | 224642/400000 [00:31<00:24, 7085.44it/s] 56%|█████▋    | 225382/400000 [00:32<00:24, 7174.06it/s] 57%|█████▋    | 226101/400000 [00:32<00:24, 7155.60it/s] 57%|█████▋    | 226818/400000 [00:32<00:24, 7073.74it/s] 57%|█████▋    | 227526/400000 [00:32<00:24, 6987.19it/s] 57%|█████▋    | 228226/400000 [00:32<00:25, 6838.66it/s] 57%|█████▋    | 228912/400000 [00:32<00:25, 6784.20it/s] 57%|█████▋    | 229650/400000 [00:32<00:24, 6950.92it/s] 58%|█████▊    | 230360/400000 [00:32<00:24, 6994.24it/s] 58%|█████▊    | 231089/400000 [00:32<00:23, 7079.05it/s] 58%|█████▊    | 231839/400000 [00:32<00:23, 7198.08it/s] 58%|█████▊    | 232591/400000 [00:33<00:22, 7290.76it/s] 58%|█████▊    | 233348/400000 [00:33<00:22, 7370.84it/s] 59%|█████▊    | 234097/400000 [00:33<00:22, 7404.05it/s] 59%|█████▊    | 234839/400000 [00:33<00:22, 7400.66it/s] 59%|█████▉    | 235580/400000 [00:33<00:22, 7365.29it/s] 59%|█████▉    | 236340/400000 [00:33<00:22, 7434.02it/s] 59%|█████▉    | 237084/400000 [00:33<00:22, 7363.07it/s] 59%|█████▉    | 237827/400000 [00:33<00:21, 7382.66it/s] 60%|█████▉    | 238566/400000 [00:33<00:22, 7311.74it/s] 60%|█████▉    | 239298/400000 [00:33<00:22, 7291.99it/s] 60%|██████    | 240042/400000 [00:34<00:21, 7333.38it/s] 60%|██████    | 240798/400000 [00:34<00:21, 7397.45it/s] 60%|██████    | 241540/400000 [00:34<00:21, 7401.48it/s] 61%|██████    | 242281/400000 [00:34<00:21, 7346.27it/s] 61%|██████    | 243025/400000 [00:34<00:21, 7373.05it/s] 61%|██████    | 243763/400000 [00:34<00:21, 7366.91it/s] 61%|██████    | 244500/400000 [00:34<00:21, 7323.70it/s] 61%|██████▏   | 245254/400000 [00:34<00:20, 7386.13it/s] 61%|██████▏   | 245993/400000 [00:34<00:20, 7334.09it/s] 62%|██████▏   | 246728/400000 [00:34<00:20, 7336.32it/s] 62%|██████▏   | 247488/400000 [00:35<00:20, 7412.52it/s] 62%|██████▏   | 248230/400000 [00:35<00:20, 7407.27it/s] 62%|██████▏   | 248982/400000 [00:35<00:20, 7439.31it/s] 62%|██████▏   | 249727/400000 [00:35<00:20, 7438.24it/s] 63%|██████▎   | 250471/400000 [00:35<00:20, 7333.76it/s] 63%|██████▎   | 251205/400000 [00:35<00:20, 7210.07it/s] 63%|██████▎   | 251930/400000 [00:35<00:20, 7221.10it/s] 63%|██████▎   | 252664/400000 [00:35<00:20, 7254.05it/s] 63%|██████▎   | 253390/400000 [00:35<00:20, 7067.34it/s] 64%|██████▎   | 254102/400000 [00:36<00:20, 7081.35it/s] 64%|██████▎   | 254836/400000 [00:36<00:20, 7155.24it/s] 64%|██████▍   | 255586/400000 [00:36<00:19, 7254.80it/s] 64%|██████▍   | 256329/400000 [00:36<00:19, 7303.91it/s] 64%|██████▍   | 257063/400000 [00:36<00:19, 7314.14it/s] 64%|██████▍   | 257795/400000 [00:36<00:19, 7203.95it/s] 65%|██████▍   | 258517/400000 [00:36<00:20, 7027.15it/s] 65%|██████▍   | 259226/400000 [00:36<00:19, 7045.80it/s] 65%|██████▍   | 259981/400000 [00:36<00:19, 7188.12it/s] 65%|██████▌   | 260711/400000 [00:36<00:19, 7216.47it/s] 65%|██████▌   | 261448/400000 [00:37<00:19, 7261.40it/s] 66%|██████▌   | 262175/400000 [00:37<00:19, 7230.21it/s] 66%|██████▌   | 262924/400000 [00:37<00:18, 7304.86it/s] 66%|██████▌   | 263656/400000 [00:37<00:18, 7293.71it/s] 66%|██████▌   | 264386/400000 [00:37<00:18, 7276.47it/s] 66%|██████▋   | 265126/400000 [00:37<00:18, 7309.56it/s] 66%|██████▋   | 265858/400000 [00:37<00:18, 7309.86it/s] 67%|██████▋   | 266600/400000 [00:37<00:18, 7341.62it/s] 67%|██████▋   | 267347/400000 [00:37<00:17, 7377.93it/s] 67%|██████▋   | 268094/400000 [00:37<00:17, 7405.10it/s] 67%|██████▋   | 268835/400000 [00:38<00:18, 7073.93it/s] 67%|██████▋   | 269546/400000 [00:38<00:18, 7054.88it/s] 68%|██████▊   | 270287/400000 [00:38<00:18, 7157.49it/s] 68%|██████▊   | 271015/400000 [00:38<00:17, 7193.39it/s] 68%|██████▊   | 271736/400000 [00:38<00:17, 7156.54it/s] 68%|██████▊   | 272480/400000 [00:38<00:17, 7237.25it/s] 68%|██████▊   | 273208/400000 [00:38<00:17, 7248.78it/s] 68%|██████▊   | 273953/400000 [00:38<00:17, 7306.84it/s] 69%|██████▊   | 274716/400000 [00:38<00:16, 7398.70it/s] 69%|██████▉   | 275473/400000 [00:38<00:16, 7448.43it/s] 69%|██████▉   | 276222/400000 [00:39<00:16, 7459.04it/s] 69%|██████▉   | 276969/400000 [00:39<00:17, 7234.04it/s] 69%|██████▉   | 277698/400000 [00:39<00:16, 7248.15it/s] 70%|██████▉   | 278431/400000 [00:39<00:16, 7271.97it/s] 70%|██████▉   | 279160/400000 [00:39<00:16, 7269.77it/s] 70%|██████▉   | 279888/400000 [00:39<00:16, 7247.83it/s] 70%|███████   | 280614/400000 [00:39<00:17, 6966.10it/s] 70%|███████   | 281314/400000 [00:39<00:17, 6912.90it/s] 71%|███████   | 282014/400000 [00:39<00:17, 6936.80it/s] 71%|███████   | 282710/400000 [00:39<00:16, 6920.36it/s] 71%|███████   | 283404/400000 [00:40<00:16, 6894.94it/s] 71%|███████   | 284123/400000 [00:40<00:16, 6978.67it/s] 71%|███████   | 284876/400000 [00:40<00:16, 7133.01it/s] 71%|███████▏  | 285617/400000 [00:40<00:15, 7211.53it/s] 72%|███████▏  | 286356/400000 [00:40<00:15, 7262.18it/s] 72%|███████▏  | 287121/400000 [00:40<00:15, 7371.69it/s] 72%|███████▏  | 287867/400000 [00:40<00:15, 7396.97it/s] 72%|███████▏  | 288623/400000 [00:40<00:14, 7444.46it/s] 72%|███████▏  | 289375/400000 [00:40<00:14, 7465.81it/s] 73%|███████▎  | 290123/400000 [00:40<00:15, 7209.09it/s] 73%|███████▎  | 290860/400000 [00:41<00:15, 7253.47it/s] 73%|███████▎  | 291607/400000 [00:41<00:14, 7316.74it/s] 73%|███████▎  | 292370/400000 [00:41<00:14, 7407.89it/s] 73%|███████▎  | 293135/400000 [00:41<00:14, 7478.28it/s] 73%|███████▎  | 293884/400000 [00:41<00:14, 7457.45it/s] 74%|███████▎  | 294631/400000 [00:41<00:14, 7292.40it/s] 74%|███████▍  | 295373/400000 [00:41<00:14, 7329.41it/s] 74%|███████▍  | 296107/400000 [00:41<00:14, 7272.68it/s] 74%|███████▍  | 296868/400000 [00:41<00:13, 7370.52it/s] 74%|███████▍  | 297620/400000 [00:41<00:13, 7413.06it/s] 75%|███████▍  | 298373/400000 [00:42<00:13, 7446.75it/s] 75%|███████▍  | 299137/400000 [00:42<00:13, 7502.77it/s] 75%|███████▍  | 299899/400000 [00:42<00:13, 7536.88it/s] 75%|███████▌  | 300668/400000 [00:42<00:13, 7579.58it/s] 75%|███████▌  | 301428/400000 [00:42<00:12, 7585.16it/s] 76%|███████▌  | 302190/400000 [00:42<00:12, 7594.59it/s] 76%|███████▌  | 302950/400000 [00:42<00:13, 7460.38it/s] 76%|███████▌  | 303697/400000 [00:42<00:13, 7396.58it/s] 76%|███████▌  | 304438/400000 [00:42<00:13, 7332.46it/s] 76%|███████▋  | 305172/400000 [00:43<00:12, 7308.60it/s] 76%|███████▋  | 305923/400000 [00:43<00:12, 7365.66it/s] 77%|███████▋  | 306660/400000 [00:43<00:12, 7300.95it/s] 77%|███████▋  | 307391/400000 [00:43<00:12, 7210.22it/s] 77%|███████▋  | 308113/400000 [00:43<00:12, 7111.94it/s] 77%|███████▋  | 308848/400000 [00:43<00:12, 7180.99it/s] 77%|███████▋  | 309567/400000 [00:43<00:12, 7145.65it/s] 78%|███████▊  | 310283/400000 [00:43<00:13, 6768.67it/s] 78%|███████▊  | 310994/400000 [00:43<00:12, 6865.61it/s] 78%|███████▊  | 311726/400000 [00:43<00:12, 6995.78it/s] 78%|███████▊  | 312429/400000 [00:44<00:12, 6917.44it/s] 78%|███████▊  | 313124/400000 [00:44<00:13, 6682.19it/s] 78%|███████▊  | 313819/400000 [00:44<00:12, 6758.17it/s] 79%|███████▊  | 314524/400000 [00:44<00:12, 6841.45it/s] 79%|███████▉  | 315263/400000 [00:44<00:12, 6996.97it/s] 79%|███████▉  | 315965/400000 [00:44<00:12, 6951.95it/s] 79%|███████▉  | 316708/400000 [00:44<00:11, 7088.65it/s] 79%|███████▉  | 317455/400000 [00:44<00:11, 7198.63it/s] 80%|███████▉  | 318179/400000 [00:44<00:11, 7210.01it/s] 80%|███████▉  | 318934/400000 [00:44<00:11, 7307.46it/s] 80%|███████▉  | 319667/400000 [00:45<00:11, 7273.00it/s] 80%|████████  | 320412/400000 [00:45<00:10, 7322.71it/s] 80%|████████  | 321145/400000 [00:45<00:11, 7121.60it/s] 80%|████████  | 321895/400000 [00:45<00:10, 7229.95it/s] 81%|████████  | 322640/400000 [00:45<00:10, 7291.97it/s] 81%|████████  | 323371/400000 [00:45<00:10, 7206.59it/s] 81%|████████  | 324127/400000 [00:45<00:10, 7307.87it/s] 81%|████████  | 324884/400000 [00:45<00:10, 7382.91it/s] 81%|████████▏ | 325624/400000 [00:45<00:10, 7224.93it/s] 82%|████████▏ | 326348/400000 [00:45<00:10, 7221.67it/s] 82%|████████▏ | 327074/400000 [00:46<00:10, 7232.81it/s] 82%|████████▏ | 327830/400000 [00:46<00:09, 7325.41it/s] 82%|████████▏ | 328571/400000 [00:46<00:09, 7347.76it/s] 82%|████████▏ | 329307/400000 [00:46<00:09, 7334.66it/s] 83%|████████▎ | 330041/400000 [00:46<00:09, 7298.79it/s] 83%|████████▎ | 330772/400000 [00:46<00:09, 7175.84it/s] 83%|████████▎ | 331518/400000 [00:46<00:09, 7258.55it/s] 83%|████████▎ | 332267/400000 [00:46<00:09, 7325.35it/s] 83%|████████▎ | 333001/400000 [00:46<00:09, 7176.39it/s] 83%|████████▎ | 333720/400000 [00:47<00:09, 7041.33it/s] 84%|████████▎ | 334432/400000 [00:47<00:09, 7063.70it/s] 84%|████████▍ | 335157/400000 [00:47<00:09, 7108.90it/s] 84%|████████▍ | 335897/400000 [00:47<00:08, 7191.58it/s] 84%|████████▍ | 336636/400000 [00:47<00:08, 7249.88it/s] 84%|████████▍ | 337362/400000 [00:47<00:08, 7070.86it/s] 85%|████████▍ | 338084/400000 [00:47<00:08, 7114.04it/s] 85%|████████▍ | 338826/400000 [00:47<00:08, 7201.97it/s] 85%|████████▍ | 339571/400000 [00:47<00:08, 7272.71it/s] 85%|████████▌ | 340312/400000 [00:47<00:08, 7311.75it/s] 85%|████████▌ | 341055/400000 [00:48<00:08, 7344.79it/s] 85%|████████▌ | 341790/400000 [00:48<00:07, 7316.57it/s] 86%|████████▌ | 342535/400000 [00:48<00:07, 7354.36it/s] 86%|████████▌ | 343287/400000 [00:48<00:07, 7400.64it/s] 86%|████████▌ | 344029/400000 [00:48<00:07, 7404.78it/s] 86%|████████▌ | 344770/400000 [00:48<00:07, 7340.69it/s] 86%|████████▋ | 345505/400000 [00:48<00:07, 7218.58it/s] 87%|████████▋ | 346245/400000 [00:48<00:07, 7269.67it/s] 87%|████████▋ | 346994/400000 [00:48<00:07, 7333.15it/s] 87%|████████▋ | 347738/400000 [00:48<00:07, 7364.81it/s] 87%|████████▋ | 348482/400000 [00:49<00:06, 7386.76it/s] 87%|████████▋ | 349221/400000 [00:49<00:07, 6998.40it/s] 87%|████████▋ | 349926/400000 [00:49<00:07, 6779.11it/s] 88%|████████▊ | 350666/400000 [00:49<00:07, 6952.68it/s] 88%|████████▊ | 351397/400000 [00:49<00:06, 7054.53it/s] 88%|████████▊ | 352119/400000 [00:49<00:06, 7101.57it/s] 88%|████████▊ | 352832/400000 [00:49<00:06, 6956.11it/s] 88%|████████▊ | 353530/400000 [00:49<00:06, 6910.10it/s] 89%|████████▊ | 354280/400000 [00:49<00:06, 7075.31it/s] 89%|████████▉ | 355016/400000 [00:49<00:06, 7157.66it/s] 89%|████████▉ | 355734/400000 [00:50<00:06, 7134.15it/s] 89%|████████▉ | 356449/400000 [00:50<00:06, 6950.17it/s] 89%|████████▉ | 357146/400000 [00:50<00:06, 6797.73it/s] 89%|████████▉ | 357828/400000 [00:50<00:06, 6731.43it/s] 90%|████████▉ | 358552/400000 [00:50<00:06, 6875.81it/s] 90%|████████▉ | 359242/400000 [00:50<00:06, 6740.68it/s] 90%|████████▉ | 359918/400000 [00:50<00:06, 6647.40it/s] 90%|█████████ | 360612/400000 [00:50<00:05, 6730.00it/s] 90%|█████████ | 361351/400000 [00:50<00:05, 6914.21it/s] 91%|█████████ | 362090/400000 [00:51<00:05, 7050.30it/s] 91%|█████████ | 362803/400000 [00:51<00:05, 7071.24it/s] 91%|█████████ | 363512/400000 [00:51<00:05, 6963.26it/s] 91%|█████████ | 364263/400000 [00:51<00:05, 7117.12it/s] 91%|█████████▏| 365012/400000 [00:51<00:04, 7224.63it/s] 91%|█████████▏| 365758/400000 [00:51<00:04, 7292.91it/s] 92%|█████████▏| 366489/400000 [00:51<00:04, 7246.07it/s] 92%|█████████▏| 367216/400000 [00:51<00:04, 7250.52it/s] 92%|█████████▏| 367942/400000 [00:51<00:04, 7192.72it/s] 92%|█████████▏| 368662/400000 [00:51<00:04, 7172.94it/s] 92%|█████████▏| 369380/400000 [00:52<00:04, 7086.81it/s] 93%|█████████▎| 370095/400000 [00:52<00:04, 7103.43it/s] 93%|█████████▎| 370830/400000 [00:52<00:04, 7174.62it/s] 93%|█████████▎| 371548/400000 [00:52<00:03, 7124.07it/s] 93%|█████████▎| 372278/400000 [00:52<00:03, 7174.20it/s] 93%|█████████▎| 373007/400000 [00:52<00:03, 7208.20it/s] 93%|█████████▎| 373734/400000 [00:52<00:03, 7224.22it/s] 94%|█████████▎| 374457/400000 [00:52<00:03, 7166.55it/s] 94%|█████████▍| 375194/400000 [00:52<00:03, 7225.20it/s] 94%|█████████▍| 375919/400000 [00:52<00:03, 7232.47it/s] 94%|█████████▍| 376648/400000 [00:53<00:03, 7247.45it/s] 94%|█████████▍| 377373/400000 [00:53<00:03, 7191.76it/s] 95%|█████████▍| 378093/400000 [00:53<00:03, 6942.79it/s] 95%|█████████▍| 378817/400000 [00:53<00:03, 7028.16it/s] 95%|█████████▍| 379522/400000 [00:53<00:03, 6821.01it/s] 95%|█████████▌| 380279/400000 [00:53<00:02, 7028.44it/s] 95%|█████████▌| 380986/400000 [00:53<00:02, 7013.51it/s] 95%|█████████▌| 381724/400000 [00:53<00:02, 7119.20it/s] 96%|█████████▌| 382474/400000 [00:53<00:02, 7227.00it/s] 96%|█████████▌| 383226/400000 [00:53<00:02, 7310.69it/s] 96%|█████████▌| 383960/400000 [00:54<00:02, 7316.87it/s] 96%|█████████▌| 384693/400000 [00:54<00:02, 7171.70it/s] 96%|█████████▋| 385437/400000 [00:54<00:02, 7250.05it/s] 97%|█████████▋| 386180/400000 [00:54<00:01, 7302.62it/s] 97%|█████████▋| 386927/400000 [00:54<00:01, 7350.33it/s] 97%|█████████▋| 387663/400000 [00:54<00:01, 7269.97it/s] 97%|█████████▋| 388391/400000 [00:54<00:01, 7264.78it/s] 97%|█████████▋| 389135/400000 [00:54<00:01, 7314.83it/s] 97%|█████████▋| 389891/400000 [00:54<00:01, 7384.10it/s] 98%|█████████▊| 390630/400000 [00:54<00:01, 7331.66it/s] 98%|█████████▊| 391364/400000 [00:55<00:01, 7320.95it/s] 98%|█████████▊| 392097/400000 [00:55<00:01, 7280.82it/s] 98%|█████████▊| 392834/400000 [00:55<00:00, 7306.23it/s] 98%|█████████▊| 393565/400000 [00:55<00:00, 7250.08it/s] 99%|█████████▊| 394305/400000 [00:55<00:00, 7291.85it/s] 99%|█████████▉| 395035/400000 [00:55<00:00, 7250.30it/s] 99%|█████████▉| 395761/400000 [00:55<00:00, 7249.08it/s] 99%|█████████▉| 396487/400000 [00:55<00:00, 7181.35it/s] 99%|█████████▉| 397206/400000 [00:55<00:00, 7171.20it/s] 99%|█████████▉| 397924/400000 [00:55<00:00, 7118.70it/s]100%|█████████▉| 398653/400000 [00:56<00:00, 7167.84it/s]100%|█████████▉| 399376/400000 [00:56<00:00, 7183.70it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7109.13it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fa9ff959c18>, <torchtext.data.dataset.TabularDataset object at 0x7fa9ff959c50>, <torchtext.vocab.Vocab object at 0x7fa9ff95c198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.01 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.01 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.01 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]2020-06-16 18:19:24.992205: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-16 18:19:24.996505: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-16 18:19:24.996762: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557d58dfc590 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 18:19:24.996821: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3645440/9912422 [00:00<00:00, 36450610.94it/s]9920512it [00:00, 35988119.03it/s]                             
0it [00:00, ?it/s]32768it [00:00, 649461.08it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 985168.98it/s]1654784it [00:00, 12421753.43it/s]                          
0it [00:00, ?it/s]8192it [00:00, 222360.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
