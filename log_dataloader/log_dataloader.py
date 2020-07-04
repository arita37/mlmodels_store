
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1f6ff44048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1f6ff44048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1fdbaee1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1fdbaee1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1ff9e34ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1ff9e34ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f88e1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f88e1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f88e1a620> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 475115.76it/s] 90%|████████▉ | 8871936/9912422 [00:00<00:01, 677162.04it/s]9920512it [00:00, 46909528.62it/s]                           
0it [00:00, ?it/s]32768it [00:00, 677192.63it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 990644.38it/s]1654784it [00:00, 12235856.34it/s]                          
0it [00:00, ?it/s]8192it [00:00, 217772.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f8620d940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f6fe50be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1f88e1a268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1f88e1a268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1f88e1a268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:12,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:12,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:11,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:11,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:48,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:48,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:31,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:30,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:30,  4.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:20,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:19,  6.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:00<00:09, 11.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 11.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 15.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 20.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 27.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 27.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 42.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 42.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 60.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 69.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 83.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.14s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.14s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.14s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.07 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.14s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.07 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.57 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ url]
0 examples [00:00, ? examples/s]2020-07-04 18:11:02.204438: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-04 18:11:02.217320: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-07-04 18:11:02.217599: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f1e9b47650 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 18:11:02.217625: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.18 examples/s]92 examples [00:00,  4.54 examples/s]199 examples [00:00,  6.48 examples/s]302 examples [00:00,  9.23 examples/s]405 examples [00:00, 13.13 examples/s]511 examples [00:00, 18.66 examples/s]615 examples [00:00, 26.46 examples/s]721 examples [00:01, 37.40 examples/s]827 examples [00:01, 52.62 examples/s]933 examples [00:01, 73.61 examples/s]1035 examples [00:01, 102.00 examples/s]1142 examples [00:01, 139.98 examples/s]1247 examples [00:01, 189.10 examples/s]1354 examples [00:01, 251.10 examples/s]1463 examples [00:01, 326.30 examples/s]1571 examples [00:01, 412.27 examples/s]1680 examples [00:01, 506.52 examples/s]1787 examples [00:02, 596.50 examples/s]1893 examples [00:02, 685.11 examples/s]1999 examples [00:02, 753.23 examples/s]2103 examples [00:02, 819.15 examples/s]2207 examples [00:02, 872.25 examples/s]2310 examples [00:02, 911.41 examples/s]2419 examples [00:02, 956.23 examples/s]2524 examples [00:02, 981.17 examples/s]2633 examples [00:02, 1008.05 examples/s]2739 examples [00:02, 1020.66 examples/s]2845 examples [00:03, 1018.65 examples/s]2950 examples [00:03, 1027.63 examples/s]3057 examples [00:03, 1039.48 examples/s]3163 examples [00:03, 1045.22 examples/s]3269 examples [00:03, 1035.72 examples/s]3374 examples [00:03, 1024.78 examples/s]3483 examples [00:03, 1042.25 examples/s]3588 examples [00:03, 1044.33 examples/s]3695 examples [00:03, 1050.92 examples/s]3801 examples [00:03, 1043.82 examples/s]3907 examples [00:04, 1045.92 examples/s]4012 examples [00:04, 1033.34 examples/s]4119 examples [00:04, 1043.16 examples/s]4224 examples [00:04, 1042.83 examples/s]4330 examples [00:04, 1046.09 examples/s]4435 examples [00:04, 1041.18 examples/s]4542 examples [00:04, 1047.81 examples/s]4649 examples [00:04, 1053.15 examples/s]4756 examples [00:04, 1056.98 examples/s]4862 examples [00:04, 1052.92 examples/s]4968 examples [00:05, 1044.03 examples/s]5073 examples [00:05, 1041.60 examples/s]5178 examples [00:05, 1031.98 examples/s]5283 examples [00:05, 1036.49 examples/s]5387 examples [00:05, 1022.47 examples/s]5492 examples [00:05, 1027.89 examples/s]5601 examples [00:05, 1043.05 examples/s]5709 examples [00:05, 1052.28 examples/s]5815 examples [00:05, 1023.51 examples/s]5919 examples [00:05, 1026.73 examples/s]6022 examples [00:06, 1026.66 examples/s]6125 examples [00:06, 1021.13 examples/s]6228 examples [00:06, 1022.59 examples/s]6332 examples [00:06, 1025.87 examples/s]6435 examples [00:06, 1012.12 examples/s]6538 examples [00:06, 1017.10 examples/s]6646 examples [00:06, 1034.99 examples/s]6753 examples [00:06, 1045.12 examples/s]6861 examples [00:06, 1055.03 examples/s]6968 examples [00:07, 1059.20 examples/s]7075 examples [00:07, 1060.85 examples/s]7186 examples [00:07, 1072.60 examples/s]7295 examples [00:07, 1075.39 examples/s]7403 examples [00:07, 1073.19 examples/s]7511 examples [00:07, 1073.45 examples/s]7619 examples [00:07, 1067.77 examples/s]7728 examples [00:07, 1072.71 examples/s]7838 examples [00:07, 1079.68 examples/s]7946 examples [00:07, 1069.55 examples/s]8053 examples [00:08, 1053.25 examples/s]8159 examples [00:08, 1045.80 examples/s]8266 examples [00:08, 1050.08 examples/s]8375 examples [00:08, 1059.06 examples/s]8484 examples [00:08, 1066.79 examples/s]8593 examples [00:08, 1073.26 examples/s]8701 examples [00:08, 1068.79 examples/s]8808 examples [00:08, 1063.56 examples/s]8918 examples [00:08, 1072.25 examples/s]9027 examples [00:08, 1076.55 examples/s]9135 examples [00:09, 1071.24 examples/s]9243 examples [00:09, 1069.60 examples/s]9353 examples [00:09, 1076.50 examples/s]9463 examples [00:09, 1080.69 examples/s]9574 examples [00:09, 1088.24 examples/s]9683 examples [00:09, 1081.37 examples/s]9792 examples [00:09, 1041.81 examples/s]9898 examples [00:09, 1046.54 examples/s]10003 examples [00:09, 1001.14 examples/s]10111 examples [00:09, 1022.88 examples/s]10218 examples [00:10, 1036.08 examples/s]10326 examples [00:10, 1048.54 examples/s]10435 examples [00:10, 1058.45 examples/s]10545 examples [00:10, 1068.93 examples/s]10655 examples [00:10, 1077.01 examples/s]10765 examples [00:10, 1082.98 examples/s]10874 examples [00:10, 1079.78 examples/s]10987 examples [00:10, 1092.19 examples/s]11097 examples [00:10, 1073.71 examples/s]11205 examples [00:10, 1072.17 examples/s]11313 examples [00:11, 1066.35 examples/s]11420 examples [00:11, 1059.02 examples/s]11530 examples [00:11, 1068.14 examples/s]11638 examples [00:11, 1069.75 examples/s]11747 examples [00:11, 1074.35 examples/s]11856 examples [00:11, 1078.40 examples/s]11964 examples [00:11, 1076.25 examples/s]12074 examples [00:11, 1081.14 examples/s]12184 examples [00:11, 1084.17 examples/s]12293 examples [00:11, 1072.81 examples/s]12402 examples [00:12, 1075.03 examples/s]12510 examples [00:12, 1074.23 examples/s]12622 examples [00:12, 1084.98 examples/s]12733 examples [00:12, 1091.06 examples/s]12843 examples [00:12, 1091.83 examples/s]12954 examples [00:12, 1094.53 examples/s]13064 examples [00:12, 1092.42 examples/s]13174 examples [00:12, 1057.40 examples/s]13283 examples [00:12, 1066.69 examples/s]13390 examples [00:13, 1062.39 examples/s]13497 examples [00:13, 1062.06 examples/s]13605 examples [00:13, 1066.90 examples/s]13716 examples [00:13, 1079.44 examples/s]13825 examples [00:13, 1078.44 examples/s]13935 examples [00:13, 1083.53 examples/s]14045 examples [00:13, 1086.34 examples/s]14154 examples [00:13, 1084.49 examples/s]14263 examples [00:13, 1075.16 examples/s]14372 examples [00:13, 1078.96 examples/s]14480 examples [00:14, 1077.10 examples/s]14590 examples [00:14, 1081.12 examples/s]14699 examples [00:14, 1075.87 examples/s]14808 examples [00:14, 1079.11 examples/s]14917 examples [00:14, 1080.75 examples/s]15027 examples [00:14, 1086.44 examples/s]15136 examples [00:14, 1081.95 examples/s]15245 examples [00:14, 1082.56 examples/s]15356 examples [00:14, 1090.11 examples/s]15468 examples [00:14, 1097.42 examples/s]15578 examples [00:15, 1085.48 examples/s]15687 examples [00:15, 1084.01 examples/s]15796 examples [00:15, 1044.69 examples/s]15905 examples [00:15, 1056.44 examples/s]16015 examples [00:15, 1067.03 examples/s]16125 examples [00:15, 1075.47 examples/s]16234 examples [00:15, 1079.02 examples/s]16343 examples [00:15, 1080.16 examples/s]16452 examples [00:15, 1035.33 examples/s]16562 examples [00:15, 1052.15 examples/s]16672 examples [00:16, 1064.50 examples/s]16781 examples [00:16, 1071.82 examples/s]16889 examples [00:16, 1062.62 examples/s]16997 examples [00:16, 1066.61 examples/s]17106 examples [00:16, 1071.75 examples/s]17216 examples [00:16, 1079.10 examples/s]17324 examples [00:16, 1070.36 examples/s]17432 examples [00:16, 1071.23 examples/s]17541 examples [00:16, 1075.99 examples/s]17649 examples [00:16, 1076.31 examples/s]17758 examples [00:17, 1077.80 examples/s]17866 examples [00:17, 1074.84 examples/s]17974 examples [00:17, 1075.32 examples/s]18082 examples [00:17, 1075.05 examples/s]18193 examples [00:17, 1082.54 examples/s]18302 examples [00:17, 1083.49 examples/s]18411 examples [00:17, 1084.88 examples/s]18520 examples [00:17, 1074.27 examples/s]18631 examples [00:17, 1082.73 examples/s]18740 examples [00:17, 1082.09 examples/s]18849 examples [00:18, 1080.32 examples/s]18959 examples [00:18, 1084.97 examples/s]19068 examples [00:18, 1080.45 examples/s]19178 examples [00:18, 1083.67 examples/s]19288 examples [00:18, 1086.58 examples/s]19398 examples [00:18, 1088.85 examples/s]19508 examples [00:18, 1090.68 examples/s]19618 examples [00:18, 1078.15 examples/s]19726 examples [00:18, 1050.36 examples/s]19833 examples [00:19, 1055.67 examples/s]19941 examples [00:19, 1062.07 examples/s]20048 examples [00:19, 1005.35 examples/s]20158 examples [00:19, 1029.66 examples/s]20268 examples [00:19, 1048.72 examples/s]20379 examples [00:19, 1064.89 examples/s]20486 examples [00:19, 1062.86 examples/s]20593 examples [00:19, 1064.23 examples/s]20702 examples [00:19, 1069.67 examples/s]20810 examples [00:19, 1071.44 examples/s]20919 examples [00:20, 1076.05 examples/s]21027 examples [00:20, 1075.48 examples/s]21135 examples [00:20, 1049.80 examples/s]21242 examples [00:20, 1055.29 examples/s]21349 examples [00:20, 1059.21 examples/s]21457 examples [00:20, 1065.25 examples/s]21566 examples [00:20, 1072.52 examples/s]21675 examples [00:20, 1075.45 examples/s]21783 examples [00:20, 1069.68 examples/s]21894 examples [00:20, 1080.67 examples/s]22007 examples [00:21, 1092.45 examples/s]22117 examples [00:21, 1090.93 examples/s]22229 examples [00:21, 1097.62 examples/s]22339 examples [00:21, 1083.35 examples/s]22448 examples [00:21, 1083.51 examples/s]22558 examples [00:21, 1086.29 examples/s]22667 examples [00:21, 1086.90 examples/s]22776 examples [00:21, 1085.66 examples/s]22885 examples [00:21, 1080.99 examples/s]22994 examples [00:21, 1053.64 examples/s]23100 examples [00:22, 1053.94 examples/s]23209 examples [00:22, 1062.90 examples/s]23319 examples [00:22, 1071.48 examples/s]23428 examples [00:22, 1076.29 examples/s]23537 examples [00:22, 1078.29 examples/s]23646 examples [00:22, 1079.06 examples/s]23755 examples [00:22, 1079.38 examples/s]23865 examples [00:22, 1084.68 examples/s]23974 examples [00:22, 1085.23 examples/s]24086 examples [00:22, 1094.43 examples/s]24196 examples [00:23, 1085.74 examples/s]24305 examples [00:23, 1080.21 examples/s]24414 examples [00:23, 1079.40 examples/s]24522 examples [00:23, 1071.93 examples/s]24632 examples [00:23, 1078.59 examples/s]24741 examples [00:23, 1081.71 examples/s]24850 examples [00:23, 1081.82 examples/s]24959 examples [00:23, 1067.47 examples/s]25066 examples [00:23, 1068.02 examples/s]25176 examples [00:23, 1076.40 examples/s]25284 examples [00:24, 1075.02 examples/s]25394 examples [00:24, 1080.28 examples/s]25503 examples [00:24, 1079.50 examples/s]25611 examples [00:24, 1054.08 examples/s]25723 examples [00:24, 1071.42 examples/s]25833 examples [00:24, 1079.78 examples/s]25942 examples [00:24, 1081.69 examples/s]26052 examples [00:24, 1084.80 examples/s]26161 examples [00:24, 1077.69 examples/s]26269 examples [00:25, 1072.40 examples/s]26377 examples [00:25, 1052.75 examples/s]26486 examples [00:25, 1061.80 examples/s]26595 examples [00:25, 1067.32 examples/s]26704 examples [00:25, 1071.22 examples/s]26814 examples [00:25, 1077.73 examples/s]26924 examples [00:25, 1082.29 examples/s]27033 examples [00:25, 1081.97 examples/s]27142 examples [00:25, 1081.31 examples/s]27251 examples [00:25, 1063.91 examples/s]27359 examples [00:26, 1066.98 examples/s]27466 examples [00:26, 1039.72 examples/s]27575 examples [00:26, 1053.17 examples/s]27683 examples [00:26, 1059.89 examples/s]27791 examples [00:26, 1064.58 examples/s]27898 examples [00:26, 1053.21 examples/s]28006 examples [00:26, 1059.06 examples/s]28116 examples [00:26, 1069.45 examples/s]28225 examples [00:26, 1072.24 examples/s]28334 examples [00:26, 1076.56 examples/s]28444 examples [00:27, 1079.11 examples/s]28553 examples [00:27, 1080.34 examples/s]28662 examples [00:27, 1066.20 examples/s]28771 examples [00:27, 1071.12 examples/s]28880 examples [00:27, 1074.50 examples/s]28988 examples [00:27, 1067.48 examples/s]29096 examples [00:27, 1070.24 examples/s]29206 examples [00:27, 1077.88 examples/s]29315 examples [00:27, 1081.36 examples/s]29424 examples [00:27, 1074.05 examples/s]29532 examples [00:28, 1063.77 examples/s]29639 examples [00:28, 1048.66 examples/s]29747 examples [00:28, 1055.76 examples/s]29853 examples [00:28, 1031.48 examples/s]29963 examples [00:28, 1048.91 examples/s]30069 examples [00:28, 1003.90 examples/s]30178 examples [00:28, 1026.45 examples/s]30285 examples [00:28, 1038.61 examples/s]30393 examples [00:28, 1050.05 examples/s]30503 examples [00:28, 1063.47 examples/s]30612 examples [00:29, 1068.81 examples/s]30721 examples [00:29, 1074.00 examples/s]30830 examples [00:29, 1078.56 examples/s]30939 examples [00:29, 1081.77 examples/s]31049 examples [00:29, 1084.96 examples/s]31158 examples [00:29, 1083.37 examples/s]31267 examples [00:29, 1084.02 examples/s]31378 examples [00:29, 1088.83 examples/s]31487 examples [00:29, 1084.71 examples/s]31596 examples [00:29, 1078.14 examples/s]31704 examples [00:30, 1077.22 examples/s]31812 examples [00:30, 1076.79 examples/s]31921 examples [00:30, 1080.34 examples/s]32030 examples [00:30, 1079.33 examples/s]32138 examples [00:30, 1074.78 examples/s]32247 examples [00:30, 1077.99 examples/s]32357 examples [00:30, 1083.41 examples/s]32468 examples [00:30, 1090.02 examples/s]32578 examples [00:30, 1058.36 examples/s]32686 examples [00:31, 1062.96 examples/s]32793 examples [00:31, 1047.19 examples/s]32898 examples [00:31, 1035.72 examples/s]33007 examples [00:31, 1050.26 examples/s]33114 examples [00:31, 1054.21 examples/s]33223 examples [00:31, 1064.12 examples/s]33330 examples [00:31, 1065.46 examples/s]33437 examples [00:31, 1057.91 examples/s]33545 examples [00:31, 1064.38 examples/s]33653 examples [00:31, 1068.26 examples/s]33762 examples [00:32, 1073.83 examples/s]33870 examples [00:32, 1052.71 examples/s]33980 examples [00:32, 1065.77 examples/s]34090 examples [00:32, 1075.35 examples/s]34198 examples [00:32, 1075.20 examples/s]34307 examples [00:32, 1077.67 examples/s]34416 examples [00:32, 1079.33 examples/s]34525 examples [00:32, 1080.26 examples/s]34635 examples [00:32, 1084.40 examples/s]34744 examples [00:32, 1084.74 examples/s]34853 examples [00:33, 1081.18 examples/s]34962 examples [00:33, 1079.20 examples/s]35072 examples [00:33, 1083.85 examples/s]35182 examples [00:33, 1087.72 examples/s]35291 examples [00:33, 1084.88 examples/s]35400 examples [00:33, 1078.63 examples/s]35508 examples [00:33, 1073.79 examples/s]35619 examples [00:33, 1082.01 examples/s]35728 examples [00:33, 1079.29 examples/s]35839 examples [00:33, 1086.19 examples/s]35948 examples [00:34, 1083.27 examples/s]36058 examples [00:34, 1086.33 examples/s]36167 examples [00:34, 1068.73 examples/s]36277 examples [00:34, 1076.99 examples/s]36386 examples [00:34, 1078.45 examples/s]36494 examples [00:34, 1075.54 examples/s]36603 examples [00:34, 1078.73 examples/s]36714 examples [00:34, 1086.69 examples/s]36825 examples [00:34, 1091.43 examples/s]36935 examples [00:34, 1092.68 examples/s]37045 examples [00:35, 1080.86 examples/s]37154 examples [00:35, 1068.08 examples/s]37262 examples [00:35, 1069.03 examples/s]37372 examples [00:35, 1076.06 examples/s]37483 examples [00:35, 1083.67 examples/s]37592 examples [00:35, 1074.46 examples/s]37701 examples [00:35, 1076.35 examples/s]37809 examples [00:35, 1040.16 examples/s]37914 examples [00:35, 1026.49 examples/s]38023 examples [00:35, 1042.44 examples/s]38130 examples [00:36, 1048.91 examples/s]38239 examples [00:36, 1060.04 examples/s]38347 examples [00:36, 1063.83 examples/s]38457 examples [00:36, 1073.00 examples/s]38568 examples [00:36, 1081.42 examples/s]38677 examples [00:36, 1078.10 examples/s]38787 examples [00:36, 1083.58 examples/s]38896 examples [00:36, 1058.89 examples/s]39003 examples [00:36, 1028.66 examples/s]39112 examples [00:37, 1044.65 examples/s]39218 examples [00:37, 1048.67 examples/s]39327 examples [00:37, 1058.33 examples/s]39433 examples [00:37, 1046.84 examples/s]39544 examples [00:37, 1062.90 examples/s]39651 examples [00:37, 1060.67 examples/s]39758 examples [00:37, 1061.18 examples/s]39869 examples [00:37, 1073.40 examples/s]39979 examples [00:37, 1079.67 examples/s]40088 examples [00:37, 1019.68 examples/s]40195 examples [00:38, 1006.17 examples/s]40303 examples [00:38, 1025.78 examples/s]40413 examples [00:38, 1046.20 examples/s]40521 examples [00:38, 1056.11 examples/s]40631 examples [00:38, 1067.10 examples/s]40738 examples [00:38, 1063.79 examples/s]40845 examples [00:38, 1061.55 examples/s]40956 examples [00:38, 1073.54 examples/s]41064 examples [00:38, 1073.50 examples/s]41173 examples [00:38, 1077.69 examples/s]41281 examples [00:39, 1073.27 examples/s]41389 examples [00:39, 1074.04 examples/s]41497 examples [00:39, 1074.20 examples/s]41605 examples [00:39, 1075.17 examples/s]41715 examples [00:39, 1080.94 examples/s]41825 examples [00:39, 1084.22 examples/s]41934 examples [00:39, 1083.37 examples/s]42043 examples [00:39, 1078.52 examples/s]42152 examples [00:39, 1080.24 examples/s]42261 examples [00:39, 1079.17 examples/s]42369 examples [00:40, 1072.87 examples/s]42477 examples [00:40, 1072.48 examples/s]42585 examples [00:40, 1069.06 examples/s]42692 examples [00:40, 1033.42 examples/s]42801 examples [00:40, 1049.47 examples/s]42908 examples [00:40, 1053.85 examples/s]43017 examples [00:40, 1061.25 examples/s]43128 examples [00:40, 1074.87 examples/s]43236 examples [00:40, 1037.67 examples/s]43349 examples [00:41, 1061.68 examples/s]43457 examples [00:41, 1065.59 examples/s]43564 examples [00:41, 1059.58 examples/s]43674 examples [00:41, 1071.12 examples/s]43785 examples [00:41, 1082.01 examples/s]43895 examples [00:41, 1085.24 examples/s]44004 examples [00:41, 1073.53 examples/s]44114 examples [00:41, 1079.53 examples/s]44223 examples [00:41, 1080.76 examples/s]44332 examples [00:41, 1077.71 examples/s]44440 examples [00:42, 1077.96 examples/s]44548 examples [00:42, 1070.51 examples/s]44656 examples [00:42, 1056.16 examples/s]44765 examples [00:42, 1064.83 examples/s]44875 examples [00:42, 1073.95 examples/s]44983 examples [00:42, 1071.94 examples/s]45091 examples [00:42, 1071.64 examples/s]45200 examples [00:42, 1075.64 examples/s]45308 examples [00:42, 1073.76 examples/s]45416 examples [00:42, 1074.29 examples/s]45524 examples [00:43, 1067.53 examples/s]45631 examples [00:43, 1067.79 examples/s]45738 examples [00:43, 1065.77 examples/s]45846 examples [00:43, 1068.40 examples/s]45953 examples [00:43, 1031.60 examples/s]46058 examples [00:43, 1035.02 examples/s]46164 examples [00:43, 1040.93 examples/s]46273 examples [00:43, 1054.25 examples/s]46379 examples [00:43, 1052.03 examples/s]46488 examples [00:43, 1061.68 examples/s]46597 examples [00:44, 1068.59 examples/s]46705 examples [00:44, 1068.76 examples/s]46814 examples [00:44, 1073.08 examples/s]46923 examples [00:44, 1077.05 examples/s]47032 examples [00:44, 1079.39 examples/s]47141 examples [00:44, 1079.65 examples/s]47249 examples [00:44, 1051.59 examples/s]47357 examples [00:44, 1058.18 examples/s]47464 examples [00:44, 1059.88 examples/s]47574 examples [00:44, 1071.49 examples/s]47684 examples [00:45, 1077.86 examples/s]47792 examples [00:45, 1077.16 examples/s]47900 examples [00:45, 1069.88 examples/s]48008 examples [00:45, 1058.37 examples/s]48116 examples [00:45, 1062.57 examples/s]48226 examples [00:45, 1071.07 examples/s]48334 examples [00:45, 1072.04 examples/s]48443 examples [00:45, 1074.49 examples/s]48551 examples [00:45, 1043.17 examples/s]48658 examples [00:45, 1050.09 examples/s]48767 examples [00:46, 1061.49 examples/s]48875 examples [00:46, 1066.14 examples/s]48985 examples [00:46, 1073.16 examples/s]49096 examples [00:46, 1082.82 examples/s]49205 examples [00:46, 1044.49 examples/s]49313 examples [00:46, 1054.72 examples/s]49419 examples [00:46, 1053.40 examples/s]49526 examples [00:46, 1055.70 examples/s]49636 examples [00:46, 1066.57 examples/s]49745 examples [00:47, 1072.83 examples/s]49853 examples [00:47, 1074.10 examples/s]49961 examples [00:47, 1058.05 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7426/50000 [00:00<00:00, 74255.29 examples/s] 42%|████▏     | 20794/50000 [00:00<00:00, 85681.09 examples/s] 69%|██████▊   | 34357/50000 [00:00<00:00, 96321.75 examples/s] 96%|█████████▋| 48130/50000 [00:00<00:00, 105869.38 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 867.71 examples/s]198 examples [00:00, 928.32 examples/s]308 examples [00:00, 973.24 examples/s]417 examples [00:00, 1004.21 examples/s]527 examples [00:00, 1031.06 examples/s]635 examples [00:00, 1044.88 examples/s]746 examples [00:00, 1061.78 examples/s]859 examples [00:00, 1078.98 examples/s]972 examples [00:00, 1091.18 examples/s]1082 examples [00:01, 1091.58 examples/s]1189 examples [00:01, 1070.55 examples/s]1295 examples [00:01, 1056.34 examples/s]1406 examples [00:01, 1068.97 examples/s]1518 examples [00:01, 1081.84 examples/s]1630 examples [00:01, 1091.33 examples/s]1739 examples [00:01, 1089.09 examples/s]1852 examples [00:01, 1098.79 examples/s]1962 examples [00:01, 1084.49 examples/s]2071 examples [00:01, 1075.91 examples/s]2181 examples [00:02, 1082.33 examples/s]2290 examples [00:02, 1084.06 examples/s]2400 examples [00:02, 1087.98 examples/s]2512 examples [00:02, 1096.88 examples/s]2622 examples [00:02, 1091.43 examples/s]2732 examples [00:02, 1093.42 examples/s]2842 examples [00:02, 1081.10 examples/s]2952 examples [00:02, 1085.47 examples/s]3061 examples [00:02, 1086.81 examples/s]3170 examples [00:02, 1075.65 examples/s]3281 examples [00:03, 1084.21 examples/s]3390 examples [00:03, 1031.40 examples/s]3494 examples [00:03, 1013.66 examples/s]3605 examples [00:03, 1040.66 examples/s]3713 examples [00:03, 1051.45 examples/s]3822 examples [00:03, 1062.53 examples/s]3929 examples [00:03, 1045.58 examples/s]4037 examples [00:03, 1053.27 examples/s]4143 examples [00:03, 1050.45 examples/s]4252 examples [00:03, 1060.78 examples/s]4363 examples [00:04, 1074.86 examples/s]4472 examples [00:04, 1077.12 examples/s]4581 examples [00:04, 1080.02 examples/s]4693 examples [00:04, 1088.89 examples/s]4802 examples [00:04, 1085.34 examples/s]4911 examples [00:04, 1082.54 examples/s]5020 examples [00:04, 1084.15 examples/s]5130 examples [00:04, 1087.34 examples/s]5239 examples [00:04, 1079.34 examples/s]5347 examples [00:04, 1050.38 examples/s]5458 examples [00:05, 1066.65 examples/s]5567 examples [00:05, 1072.99 examples/s]5678 examples [00:05, 1082.88 examples/s]5788 examples [00:05, 1085.94 examples/s]5897 examples [00:05, 1085.48 examples/s]6006 examples [00:05, 1083.47 examples/s]6115 examples [00:05, 1085.37 examples/s]6226 examples [00:05, 1092.22 examples/s]6336 examples [00:05, 1090.27 examples/s]6446 examples [00:05, 1089.33 examples/s]6555 examples [00:06, 1063.24 examples/s]6662 examples [00:06, 1035.10 examples/s]6772 examples [00:06, 1052.50 examples/s]6883 examples [00:06, 1066.89 examples/s]6994 examples [00:06, 1076.87 examples/s]7104 examples [00:06, 1081.01 examples/s]7213 examples [00:06, 1079.14 examples/s]7323 examples [00:06, 1083.78 examples/s]7433 examples [00:06, 1087.74 examples/s]7544 examples [00:07, 1092.68 examples/s]7655 examples [00:07, 1097.46 examples/s]7765 examples [00:07, 1096.27 examples/s]7876 examples [00:07, 1098.97 examples/s]7987 examples [00:07, 1099.77 examples/s]8097 examples [00:07, 1093.06 examples/s]8207 examples [00:07, 1094.05 examples/s]8317 examples [00:07, 1085.23 examples/s]8426 examples [00:07, 1073.00 examples/s]8534 examples [00:07, 1044.74 examples/s]8644 examples [00:08, 1060.03 examples/s]8753 examples [00:08, 1067.76 examples/s]8860 examples [00:08, 1056.32 examples/s]8966 examples [00:08, 1055.83 examples/s]9078 examples [00:08, 1071.87 examples/s]9188 examples [00:08, 1078.23 examples/s]9300 examples [00:08, 1089.36 examples/s]9410 examples [00:08, 1091.97 examples/s]9522 examples [00:08, 1098.82 examples/s]9632 examples [00:08, 1096.63 examples/s]9742 examples [00:09, 1082.10 examples/s]9851 examples [00:09, 1069.37 examples/s]9959 examples [00:09, 1057.46 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQL6BHL/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQL6BHL/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f88e1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f88e1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f88e1a620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f0e2605c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f0e29b5c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f88e1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f88e1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f88e1a620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f6fe508d0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f6fe50470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1f009471e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1f009471e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f1f009471e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1f009472f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1f009472f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1f009472f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=0174065eb1f1362aa840ef1f0aa265a970353c30ae6bd165d3a35a2093e4a98f
  Stored in directory: /tmp/pip-ephem-wheel-cache-bg1h5d2s/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:18:58, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:03:23, 18.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:11:38, 26.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:26:41, 37.1kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:30:01, 53.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.77M/862M [00:01<3:07:58, 75.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:11:06, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<1:31:25, 154kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:03:45, 220kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<44:27, 313kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<31:06, 446kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.2M/862M [00:01<21:42, 635kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:01<15:16, 899kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<10:42, 1.27MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<07:35, 1.79MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<05:21, 2.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:02<06:44, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<06:36, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:29, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:14, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<05:57, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<05:20, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<04:10, 3.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:07<03:04, 4.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<19:33, 679kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<16:31, 803kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<12:16, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:09<08:41, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<31:44, 416kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<23:35, 560kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<16:45, 787kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<14:47, 889kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<11:41, 1.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:13<08:29, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<09:02, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<07:40, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<05:38, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:02, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<07:36, 1.71MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<05:54, 2.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:17<04:16, 3.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<15:41, 825kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<12:17, 1.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:18<08:52, 1.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<09:13, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<07:46, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:20<05:45, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<07:02, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<06:14, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:22<04:38, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:16, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:00, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<05:29, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<03:58, 3.19MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<19:20, 655kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<14:36, 867kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:34, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<07:31, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<56:02, 225kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<40:30, 311kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<28:37, 439kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<22:56, 546kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<18:38, 672kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:34, 922kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<09:39, 1.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:10, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:48, 1.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:32<07:10, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:55, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:48, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:04, 2.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:28, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:56, 1.77MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:58, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:31, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:32, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:52, 2.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:16, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:37, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:14, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:49, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:18, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:01, 3.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:39, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:11, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:55, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:07, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:18, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:00, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:24, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:02, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:46, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:10, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:56, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:43, 3.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:19, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:55, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:44, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:21, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:00, 1.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:43, 2.45MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:27, 3.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:54, 1.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:07, 2.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:13, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:31, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:09, 2.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:35, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:04, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:50, 2.95MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:21, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:53, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:42, 3.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:48, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<03:38, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:35, 3.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:06, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:49, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:33, 2.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:20, 3.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:38, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:29, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:46, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:54, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:21, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:47, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:37, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:04, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:45, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:33, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:18, 3.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<43:29, 247kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<31:31, 341kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<22:15, 482kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<18:02, 592kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<14:48, 721kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:49, 986kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<07:47, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:19, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:53, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<05:05, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:23, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:00, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<03:36, 2.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<10:06:41, 17.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<7:05:31, 24.6kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<4:57:23, 35.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<3:29:53, 49.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<2:29:00, 69.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:44:42, 99.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:14:36, 139kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<53:15, 194kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<37:27, 275kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<28:31, 360kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<21:00, 488kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<14:55, 685kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<12:49, 795kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<11:04, 921kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:10, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<05:50, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<09:15, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:30, 1.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:27, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<06:11, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:21, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:57, 2.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:08, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:37, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:28, 2.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:46, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:21, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:15, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:04, 3.20MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<1:13:17, 134kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<52:16, 188kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<36:45, 267kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<27:55, 350kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<20:31, 476kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<14:34, 670kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<12:25, 782kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<09:41, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:00, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:10, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:00, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:23, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:21, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:37, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:21, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:08, 3.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<09:30, 1.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:38, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:33, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<06:05, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<06:11, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:48, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:52, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:22, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<03:18, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:28, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:00, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:58, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:55, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:57, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:52, 2.36MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:54, 3.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:12, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:47, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:46, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<02:43, 3.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<24:04, 375kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<17:46, 508kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<12:36, 714kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<10:53, 823kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<08:33, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<06:10, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:22, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:17, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:47, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<03:26, 2.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<11:14, 786kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:45, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<06:17, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:27, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:18, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:51, 1.80MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:47, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:12, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:01, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:54, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<10:06, 852kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<07:58, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:44, 1.49MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:01, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:07, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:40, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<03:21, 2.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:47, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:41, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:09, 2.03MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:47, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:16, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:10, 2.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:04, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:44, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:48, 2.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:48, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:30, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:31, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:35, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:49, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:14, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:09, 2.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:02, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:34, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:38, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<02:38, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<14:31, 555kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<11:00, 732kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<07:54, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:17, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:48, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:11, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<03:43, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<15:04, 526kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<11:24, 694kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<08:08, 969kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:24, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:51, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:11, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<03:43, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<12:06, 643kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<09:18, 836kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<06:42, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:23, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:06, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:37, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:19, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:17, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:13, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:49, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:22, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:44, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:39, 2.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<10:53, 690kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<08:25, 890kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<06:03, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:53, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:46, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:23, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<03:10, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:58, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:10, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:50, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:18, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:21, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:26, 2.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:02, 1.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:17, 1.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:09, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:48, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:18, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:23, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:34, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:21, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:55, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:18, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:34, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:31, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:32, 2.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:19, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:28, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:18, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:50, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:25, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:33, 2.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:03, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:18, 2.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:06, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:37, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:53, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<02:06, 3.18MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<11:41, 571kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:54, 749kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<06:23, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:55, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:36, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:13, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:04, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:19, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:29, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<01:49, 3.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<08:29, 762kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:37, 975kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:46, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:45, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:57, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:28, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:50, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:59, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:09, 2.90MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:33, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:51, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:50, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:21, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:40, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:04, 2.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:14, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:31, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:37, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:10, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:30, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<01:59, 3.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<15:11, 394kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<11:16, 530kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<08:01, 742kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:53, 857kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<06:04, 972kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:32, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:14, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<09:24, 621kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<07:12, 809kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<05:11, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:53, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:39, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:31, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:30, 2.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<09:45, 585kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<07:26, 765kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:19, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:57, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<04:03, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:58, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:21, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:32, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:47, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:00, 2.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<10:01, 548kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<07:36, 721kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<05:26, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:00, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:39, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:29, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:31, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:38, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:08, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:19, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:07, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:26, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:46, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:30, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:01, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:14, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:41, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:01, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:23, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:43, 2.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<09:12, 552kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<06:57, 729kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<04:58, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:36, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:44, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:02, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:10, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:28, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:46, 2.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<07:17, 669kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:37, 865kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<04:02, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:52, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:13, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:22, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:42, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:51, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:13, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<01:35, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:59, 936kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:58, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<02:53, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:02, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:08, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:26, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:45, 2.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<08:20, 543kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:18, 716kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<04:31, 995kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:07, 1.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:52, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:57, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<02:06, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<08:02, 546kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<06:04, 721kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<04:19, 1.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:00, 1.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:42, 1.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:49, 1.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<02:00, 2.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<07:55, 537kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<06:00, 707kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<04:17, 982kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:54, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:39, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:46, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<07:51, 524kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<05:54, 694kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<04:13, 966kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:51, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:33, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:40, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:54, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:35, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:56, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:09, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:22, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:27, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:53, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:21, 2.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:30, 1.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:09, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:35, 2.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:56, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:12, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:42, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:15, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:00, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:47, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:20, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:44, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:58, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:31, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:06, 3.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:25, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:01, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:34, 2.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:08, 3.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:37, 753kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:37, 962kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:36, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:34, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:29, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:53, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:20, 2.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:21, 998kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:44, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:06, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:12, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:42, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:13, 2.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<05:26, 590kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:08, 775kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:56, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:45, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:16, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:39, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:50, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:56, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:29, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:04, 2.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:19, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:56, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:39, 1.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:49, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:25, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:01, 2.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:12, 549kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:56, 724kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:48, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:34, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:25, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:50, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:17, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<04:19, 631kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:18, 822kB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:11<02:22, 1.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<02:13, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:50, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:20, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:37, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:16, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<00:54, 2.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<03:47, 665kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:54, 864kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:04, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:59, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:55, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:26, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:01, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:44, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:27, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:04, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:16, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:07, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:50, 2.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:05, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:59, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:44, 2.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:10, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:54, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:39, 3.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:50, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:30, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:05, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:13, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:17, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:59, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:42, 2.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:30, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:15, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:55, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:03, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:08, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:52, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:37, 2.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:15, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:04, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:47, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:55, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:00, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:47, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:33, 3.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<09:15, 183kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<06:37, 254kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<04:37, 359kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<03:31, 460kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:48, 576kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<02:01, 793kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<01:24, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:45, 879kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:23, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:59, 1.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:01, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:52, 1.69MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:38, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:49, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:27, 2.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:22, 565kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<01:48, 737kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<01:16, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:09, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:56, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:40, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:44, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:38, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:28, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:34, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:23, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:29, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:26, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<00:18, 3.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<07:20, 136kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<05:12, 190kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<03:35, 270kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<02:37, 355kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<02:02, 455kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:27, 628kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:58, 888kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:18, 656kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:00, 852kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:42, 1.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:38, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:37, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:27, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:19, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:28, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:24, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:17, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:20, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:22, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:17, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:11, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:42, 825kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:33, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:18, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:13, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:08, 3.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:40, 660kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:33, 785kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:24, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:15, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:29, 756kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:22, 966kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:15, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:05, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:58, 242kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:41, 334kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:26, 471kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:17, 585kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:13, 709kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:09, 968kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:04, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 780kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 994kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.79MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 812/400000 [00:00<00:49, 8110.94it/s]  0%|          | 1618/400000 [00:00<00:49, 8094.16it/s]  1%|          | 2461/400000 [00:00<00:48, 8192.00it/s]  1%|          | 3315/400000 [00:00<00:47, 8292.70it/s]  1%|          | 4170/400000 [00:00<00:47, 8366.02it/s]  1%|▏         | 5014/400000 [00:00<00:47, 8386.71it/s]  1%|▏         | 5831/400000 [00:00<00:47, 8320.20it/s]  2%|▏         | 6684/400000 [00:00<00:46, 8379.54it/s]  2%|▏         | 7546/400000 [00:00<00:46, 8448.74it/s]  2%|▏         | 8403/400000 [00:01<00:46, 8483.60it/s]  2%|▏         | 9231/400000 [00:01<00:46, 8420.08it/s]  3%|▎         | 10080/400000 [00:01<00:46, 8439.17it/s]  3%|▎         | 10913/400000 [00:01<00:46, 8360.85it/s]  3%|▎         | 11742/400000 [00:01<00:46, 8296.74it/s]  3%|▎         | 12592/400000 [00:01<00:46, 8355.74it/s]  3%|▎         | 13427/400000 [00:01<00:46, 8352.30it/s]  4%|▎         | 14284/400000 [00:01<00:45, 8414.66it/s]  4%|▍         | 15124/400000 [00:01<00:45, 8402.61it/s]  4%|▍         | 15964/400000 [00:01<00:46, 8276.17it/s]  4%|▍         | 16815/400000 [00:02<00:45, 8342.23it/s]  4%|▍         | 17666/400000 [00:02<00:45, 8387.89it/s]  5%|▍         | 18531/400000 [00:02<00:45, 8462.49it/s]  5%|▍         | 19378/400000 [00:02<00:45, 8431.89it/s]  5%|▌         | 20222/400000 [00:02<00:45, 8418.92it/s]  5%|▌         | 21064/400000 [00:02<00:45, 8396.08it/s]  5%|▌         | 21909/400000 [00:02<00:44, 8411.36it/s]  6%|▌         | 22751/400000 [00:02<00:45, 8358.78it/s]  6%|▌         | 23600/400000 [00:02<00:44, 8395.71it/s]  6%|▌         | 24441/400000 [00:02<00:44, 8397.15it/s]  6%|▋         | 25282/400000 [00:03<00:44, 8400.20it/s]  7%|▋         | 26123/400000 [00:03<00:44, 8396.45it/s]  7%|▋         | 26963/400000 [00:03<00:44, 8382.83it/s]  7%|▋         | 27802/400000 [00:03<00:44, 8272.27it/s]  7%|▋         | 28651/400000 [00:03<00:44, 8334.37it/s]  7%|▋         | 29516/400000 [00:03<00:43, 8424.61it/s]  8%|▊         | 30359/400000 [00:03<00:43, 8408.52it/s]  8%|▊         | 31201/400000 [00:03<00:43, 8394.52it/s]  8%|▊         | 32041/400000 [00:03<00:44, 8316.47it/s]  8%|▊         | 32895/400000 [00:03<00:43, 8379.59it/s]  8%|▊         | 33749/400000 [00:04<00:43, 8425.12it/s]  9%|▊         | 34592/400000 [00:04<00:43, 8397.79it/s]  9%|▉         | 35450/400000 [00:04<00:43, 8451.45it/s]  9%|▉         | 36300/400000 [00:04<00:42, 8462.49it/s]  9%|▉         | 37149/400000 [00:04<00:42, 8469.71it/s] 10%|▉         | 38007/400000 [00:04<00:42, 8501.57it/s] 10%|▉         | 38858/400000 [00:04<00:43, 8392.08it/s] 10%|▉         | 39706/400000 [00:04<00:42, 8416.03it/s] 10%|█         | 40548/400000 [00:04<00:42, 8397.70it/s] 10%|█         | 41388/400000 [00:04<00:43, 8337.82it/s] 11%|█         | 42243/400000 [00:05<00:42, 8399.12it/s] 11%|█         | 43099/400000 [00:05<00:42, 8443.98it/s] 11%|█         | 43953/400000 [00:05<00:42, 8472.04it/s] 11%|█         | 44805/400000 [00:05<00:41, 8486.30it/s] 11%|█▏        | 45654/400000 [00:05<00:41, 8457.45it/s] 12%|█▏        | 46512/400000 [00:05<00:41, 8491.24it/s] 12%|█▏        | 47362/400000 [00:05<00:41, 8469.09it/s] 12%|█▏        | 48221/400000 [00:05<00:41, 8502.51it/s] 12%|█▏        | 49088/400000 [00:05<00:41, 8549.77it/s] 12%|█▏        | 49951/400000 [00:05<00:40, 8571.05it/s] 13%|█▎        | 50812/400000 [00:06<00:40, 8581.55it/s] 13%|█▎        | 51671/400000 [00:06<00:40, 8562.08it/s] 13%|█▎        | 52528/400000 [00:06<00:40, 8503.65it/s] 13%|█▎        | 53379/400000 [00:06<00:40, 8495.93it/s] 14%|█▎        | 54229/400000 [00:06<00:40, 8475.92it/s] 14%|█▍        | 55094/400000 [00:06<00:40, 8524.25it/s] 14%|█▍        | 55948/400000 [00:06<00:40, 8528.12it/s] 14%|█▍        | 56801/400000 [00:06<00:40, 8486.84it/s] 14%|█▍        | 57658/400000 [00:06<00:40, 8511.38it/s] 15%|█▍        | 58510/400000 [00:06<00:40, 8484.77it/s] 15%|█▍        | 59367/400000 [00:07<00:40, 8508.92it/s] 15%|█▌        | 60222/400000 [00:07<00:39, 8520.09it/s] 15%|█▌        | 61075/400000 [00:07<00:39, 8508.83it/s] 15%|█▌        | 61932/400000 [00:07<00:39, 8525.11it/s] 16%|█▌        | 62785/400000 [00:07<00:39, 8480.08it/s] 16%|█▌        | 63645/400000 [00:07<00:39, 8514.40it/s] 16%|█▌        | 64503/400000 [00:07<00:39, 8532.99it/s] 16%|█▋        | 65357/400000 [00:07<00:39, 8497.91it/s] 17%|█▋        | 66214/400000 [00:07<00:39, 8517.83it/s] 17%|█▋        | 67066/400000 [00:07<00:39, 8500.30it/s] 17%|█▋        | 67918/400000 [00:08<00:39, 8505.59it/s] 17%|█▋        | 68780/400000 [00:08<00:38, 8539.44it/s] 17%|█▋        | 69635/400000 [00:08<00:38, 8491.63it/s] 18%|█▊        | 70485/400000 [00:08<00:39, 8426.74it/s] 18%|█▊        | 71328/400000 [00:08<00:39, 8398.33it/s] 18%|█▊        | 72172/400000 [00:08<00:38, 8410.11it/s] 18%|█▊        | 73026/400000 [00:08<00:38, 8448.25it/s] 18%|█▊        | 73871/400000 [00:08<00:38, 8436.05it/s] 19%|█▊        | 74726/400000 [00:08<00:38, 8468.74it/s] 19%|█▉        | 75573/400000 [00:08<00:38, 8424.91it/s] 19%|█▉        | 76429/400000 [00:09<00:38, 8462.80it/s] 19%|█▉        | 77284/400000 [00:09<00:38, 8485.73it/s] 20%|█▉        | 78133/400000 [00:09<00:38, 8384.45it/s] 20%|█▉        | 78972/400000 [00:09<00:38, 8384.99it/s] 20%|█▉        | 79811/400000 [00:09<00:38, 8341.65it/s] 20%|██        | 80675/400000 [00:09<00:37, 8428.68it/s] 20%|██        | 81519/400000 [00:09<00:38, 8266.46it/s] 21%|██        | 82354/400000 [00:09<00:38, 8290.06it/s] 21%|██        | 83206/400000 [00:09<00:37, 8356.78it/s] 21%|██        | 84043/400000 [00:09<00:37, 8357.81it/s] 21%|██        | 84880/400000 [00:10<00:37, 8319.76it/s] 21%|██▏       | 85713/400000 [00:10<00:37, 8316.95it/s] 22%|██▏       | 86549/400000 [00:10<00:37, 8327.76it/s] 22%|██▏       | 87396/400000 [00:10<00:37, 8369.47it/s] 22%|██▏       | 88237/400000 [00:10<00:37, 8379.71it/s] 22%|██▏       | 89076/400000 [00:10<00:37, 8379.58it/s] 22%|██▏       | 89936/400000 [00:10<00:36, 8442.33it/s] 23%|██▎       | 90781/400000 [00:10<00:36, 8420.96it/s] 23%|██▎       | 91636/400000 [00:10<00:36, 8458.81it/s] 23%|██▎       | 92494/400000 [00:10<00:36, 8494.47it/s] 23%|██▎       | 93344/400000 [00:11<00:36, 8448.69it/s] 24%|██▎       | 94200/400000 [00:11<00:36, 8480.87it/s] 24%|██▍       | 95049/400000 [00:11<00:36, 8446.06it/s] 24%|██▍       | 95899/400000 [00:11<00:35, 8460.58it/s] 24%|██▍       | 96753/400000 [00:11<00:35, 8483.82it/s] 24%|██▍       | 97615/400000 [00:11<00:35, 8523.24it/s] 25%|██▍       | 98473/400000 [00:11<00:35, 8539.07it/s] 25%|██▍       | 99327/400000 [00:11<00:35, 8503.96it/s] 25%|██▌       | 100178/400000 [00:11<00:35, 8487.12it/s] 25%|██▌       | 101034/400000 [00:11<00:35, 8507.07it/s] 25%|██▌       | 101885/400000 [00:12<00:35, 8422.30it/s] 26%|██▌       | 102728/400000 [00:12<00:36, 8249.77it/s] 26%|██▌       | 103556/400000 [00:12<00:35, 8257.93it/s] 26%|██▌       | 104428/400000 [00:12<00:35, 8389.56it/s] 26%|██▋       | 105297/400000 [00:12<00:34, 8474.82it/s] 27%|██▋       | 106146/400000 [00:12<00:34, 8468.29it/s] 27%|██▋       | 107012/400000 [00:12<00:34, 8523.57it/s] 27%|██▋       | 107865/400000 [00:12<00:34, 8522.54it/s] 27%|██▋       | 108720/400000 [00:12<00:34, 8528.75it/s] 27%|██▋       | 109575/400000 [00:12<00:34, 8531.48it/s] 28%|██▊       | 110429/400000 [00:13<00:34, 8467.39it/s] 28%|██▊       | 111302/400000 [00:13<00:33, 8543.03it/s] 28%|██▊       | 112157/400000 [00:13<00:33, 8509.24it/s] 28%|██▊       | 113035/400000 [00:13<00:33, 8588.00it/s] 28%|██▊       | 113911/400000 [00:13<00:33, 8638.17it/s] 29%|██▊       | 114776/400000 [00:13<00:33, 8576.03it/s] 29%|██▉       | 115642/400000 [00:13<00:33, 8598.11it/s] 29%|██▉       | 116503/400000 [00:13<00:33, 8430.43it/s] 29%|██▉       | 117347/400000 [00:13<00:33, 8428.90it/s] 30%|██▉       | 118211/400000 [00:14<00:33, 8490.32it/s] 30%|██▉       | 119065/400000 [00:14<00:33, 8504.54it/s] 30%|██▉       | 119929/400000 [00:14<00:32, 8542.52it/s] 30%|███       | 120789/400000 [00:14<00:32, 8559.05it/s] 30%|███       | 121650/400000 [00:14<00:32, 8572.12it/s] 31%|███       | 122510/400000 [00:14<00:32, 8578.95it/s] 31%|███       | 123369/400000 [00:14<00:32, 8483.07it/s] 31%|███       | 124218/400000 [00:14<00:32, 8381.33it/s] 31%|███▏      | 125057/400000 [00:14<00:32, 8339.03it/s] 31%|███▏      | 125921/400000 [00:14<00:32, 8425.56it/s] 32%|███▏      | 126770/400000 [00:15<00:32, 8443.79it/s] 32%|███▏      | 127621/400000 [00:15<00:32, 8461.03it/s] 32%|███▏      | 128479/400000 [00:15<00:31, 8494.00it/s] 32%|███▏      | 129334/400000 [00:15<00:31, 8508.88it/s] 33%|███▎      | 130205/400000 [00:15<00:31, 8566.25it/s] 33%|███▎      | 131071/400000 [00:15<00:31, 8593.20it/s] 33%|███▎      | 131931/400000 [00:15<00:31, 8548.38it/s] 33%|███▎      | 132801/400000 [00:15<00:31, 8593.02it/s] 33%|███▎      | 133665/400000 [00:15<00:30, 8605.94it/s] 34%|███▎      | 134531/400000 [00:15<00:30, 8621.99it/s] 34%|███▍      | 135396/400000 [00:16<00:30, 8628.16it/s] 34%|███▍      | 136259/400000 [00:16<00:30, 8531.21it/s] 34%|███▍      | 137125/400000 [00:16<00:30, 8567.33it/s] 34%|███▍      | 137993/400000 [00:16<00:30, 8598.23it/s] 35%|███▍      | 138854/400000 [00:16<00:30, 8476.88it/s] 35%|███▍      | 139703/400000 [00:16<00:30, 8459.15it/s] 35%|███▌      | 140550/400000 [00:16<00:30, 8370.08it/s] 35%|███▌      | 141412/400000 [00:16<00:30, 8441.50it/s] 36%|███▌      | 142268/400000 [00:16<00:30, 8474.78it/s] 36%|███▌      | 143121/400000 [00:16<00:30, 8489.67it/s] 36%|███▌      | 143981/400000 [00:17<00:30, 8520.11it/s] 36%|███▌      | 144834/400000 [00:17<00:30, 8462.06it/s] 36%|███▋      | 145695/400000 [00:17<00:29, 8504.30it/s] 37%|███▋      | 146550/400000 [00:17<00:29, 8516.11it/s] 37%|███▋      | 147402/400000 [00:17<00:30, 8369.19it/s] 37%|███▋      | 148244/400000 [00:17<00:30, 8382.77it/s] 37%|███▋      | 149083/400000 [00:17<00:30, 8356.31it/s] 37%|███▋      | 149922/400000 [00:17<00:29, 8366.30it/s] 38%|███▊      | 150759/400000 [00:17<00:29, 8316.72it/s] 38%|███▊      | 151595/400000 [00:17<00:29, 8326.78it/s] 38%|███▊      | 152449/400000 [00:18<00:29, 8388.18it/s] 38%|███▊      | 153289/400000 [00:18<00:29, 8354.82it/s] 39%|███▊      | 154133/400000 [00:18<00:29, 8377.41it/s] 39%|███▊      | 154998/400000 [00:18<00:28, 8454.52it/s] 39%|███▉      | 155856/400000 [00:18<00:28, 8490.32it/s] 39%|███▉      | 156713/400000 [00:18<00:28, 8511.69it/s] 39%|███▉      | 157565/400000 [00:18<00:28, 8482.42it/s] 40%|███▉      | 158434/400000 [00:18<00:28, 8541.87it/s] 40%|███▉      | 159293/400000 [00:18<00:28, 8556.05it/s] 40%|████      | 160149/400000 [00:18<00:28, 8510.75it/s] 40%|████      | 161007/400000 [00:19<00:28, 8530.05it/s] 40%|████      | 161866/400000 [00:19<00:27, 8547.68it/s] 41%|████      | 162732/400000 [00:19<00:27, 8578.61it/s] 41%|████      | 163599/400000 [00:19<00:27, 8604.70it/s] 41%|████      | 164460/400000 [00:19<00:27, 8583.04it/s] 41%|████▏     | 165325/400000 [00:19<00:27, 8600.88it/s] 42%|████▏     | 166186/400000 [00:19<00:27, 8572.06it/s] 42%|████▏     | 167044/400000 [00:19<00:27, 8530.98it/s] 42%|████▏     | 167902/400000 [00:19<00:27, 8545.00it/s] 42%|████▏     | 168757/400000 [00:19<00:27, 8460.91it/s] 42%|████▏     | 169605/400000 [00:20<00:27, 8464.75it/s] 43%|████▎     | 170452/400000 [00:20<00:27, 8431.23it/s] 43%|████▎     | 171314/400000 [00:20<00:26, 8486.61it/s] 43%|████▎     | 172180/400000 [00:20<00:26, 8535.00it/s] 43%|████▎     | 173034/400000 [00:20<00:26, 8507.69it/s] 43%|████▎     | 173899/400000 [00:20<00:26, 8547.24it/s] 44%|████▎     | 174754/400000 [00:20<00:26, 8474.35it/s] 44%|████▍     | 175602/400000 [00:20<00:26, 8473.09it/s] 44%|████▍     | 176459/400000 [00:20<00:26, 8499.59it/s] 44%|████▍     | 177310/400000 [00:20<00:26, 8489.73it/s] 45%|████▍     | 178160/400000 [00:21<00:26, 8370.16it/s] 45%|████▍     | 179007/400000 [00:21<00:26, 8397.33it/s] 45%|████▍     | 179861/400000 [00:21<00:26, 8438.74it/s] 45%|████▌     | 180716/400000 [00:21<00:25, 8468.89it/s] 45%|████▌     | 181564/400000 [00:21<00:25, 8420.12it/s] 46%|████▌     | 182408/400000 [00:21<00:25, 8425.73it/s] 46%|████▌     | 183251/400000 [00:21<00:25, 8391.49it/s] 46%|████▌     | 184101/400000 [00:21<00:25, 8422.14it/s] 46%|████▌     | 184946/400000 [00:21<00:25, 8428.85it/s] 46%|████▋     | 185789/400000 [00:21<00:25, 8406.15it/s] 47%|████▋     | 186630/400000 [00:22<00:25, 8364.96it/s] 47%|████▋     | 187476/400000 [00:22<00:25, 8392.00it/s] 47%|████▋     | 188327/400000 [00:22<00:25, 8425.81it/s] 47%|████▋     | 189185/400000 [00:22<00:24, 8470.67it/s] 48%|████▊     | 190033/400000 [00:22<00:25, 8385.52it/s] 48%|████▊     | 190877/400000 [00:22<00:24, 8398.39it/s] 48%|████▊     | 191718/400000 [00:22<00:24, 8369.77it/s] 48%|████▊     | 192583/400000 [00:22<00:24, 8449.55it/s] 48%|████▊     | 193440/400000 [00:22<00:24, 8482.86it/s] 49%|████▊     | 194294/400000 [00:22<00:24, 8499.49it/s] 49%|████▉     | 195145/400000 [00:23<00:24, 8483.64it/s] 49%|████▉     | 195994/400000 [00:23<00:24, 8438.68it/s] 49%|████▉     | 196848/400000 [00:23<00:23, 8466.38it/s] 49%|████▉     | 197695/400000 [00:23<00:24, 8343.31it/s] 50%|████▉     | 198530/400000 [00:23<00:24, 8344.01it/s] 50%|████▉     | 199380/400000 [00:23<00:23, 8388.65it/s] 50%|█████     | 200226/400000 [00:23<00:23, 8409.04it/s] 50%|█████     | 201083/400000 [00:23<00:23, 8454.03it/s] 50%|█████     | 201939/400000 [00:23<00:23, 8483.71it/s] 51%|█████     | 202789/400000 [00:23<00:23, 8487.45it/s] 51%|█████     | 203646/400000 [00:24<00:23, 8509.19it/s] 51%|█████     | 204498/400000 [00:24<00:23, 8494.40it/s] 51%|█████▏    | 205360/400000 [00:24<00:22, 8529.53it/s] 52%|█████▏    | 206214/400000 [00:24<00:22, 8532.61it/s] 52%|█████▏    | 207068/400000 [00:24<00:22, 8534.71it/s] 52%|█████▏    | 207922/400000 [00:24<00:22, 8504.19it/s] 52%|█████▏    | 208773/400000 [00:24<00:22, 8348.38it/s] 52%|█████▏    | 209633/400000 [00:24<00:22, 8419.45it/s] 53%|█████▎    | 210476/400000 [00:24<00:22, 8422.45it/s] 53%|█████▎    | 211319/400000 [00:24<00:22, 8397.04it/s] 53%|█████▎    | 212172/400000 [00:25<00:22, 8436.15it/s] 53%|█████▎    | 213020/400000 [00:25<00:22, 8449.18it/s] 53%|█████▎    | 213884/400000 [00:25<00:21, 8504.59it/s] 54%|█████▎    | 214745/400000 [00:25<00:21, 8534.85it/s] 54%|█████▍    | 215599/400000 [00:25<00:21, 8511.27it/s] 54%|█████▍    | 216459/400000 [00:25<00:21, 8537.63it/s] 54%|█████▍    | 217313/400000 [00:25<00:21, 8503.05it/s] 55%|█████▍    | 218164/400000 [00:25<00:21, 8501.85it/s] 55%|█████▍    | 219015/400000 [00:25<00:21, 8498.08it/s] 55%|█████▍    | 219865/400000 [00:26<00:21, 8430.74it/s] 55%|█████▌    | 220720/400000 [00:26<00:21, 8465.54it/s] 55%|█████▌    | 221567/400000 [00:26<00:21, 8462.13it/s] 56%|█████▌    | 222444/400000 [00:26<00:20, 8552.10it/s] 56%|█████▌    | 223312/400000 [00:26<00:20, 8589.58it/s] 56%|█████▌    | 224179/400000 [00:26<00:20, 8612.83it/s] 56%|█████▋    | 225041/400000 [00:26<00:20, 8557.75it/s] 56%|█████▋    | 225897/400000 [00:26<00:20, 8495.00it/s] 57%|█████▋    | 226747/400000 [00:26<00:20, 8486.75it/s] 57%|█████▋    | 227620/400000 [00:26<00:20, 8555.86it/s] 57%|█████▋    | 228476/400000 [00:27<00:20, 8552.30it/s] 57%|█████▋    | 229355/400000 [00:27<00:19, 8620.06it/s] 58%|█████▊    | 230218/400000 [00:27<00:19, 8610.79it/s] 58%|█████▊    | 231096/400000 [00:27<00:19, 8658.25it/s] 58%|█████▊    | 231963/400000 [00:27<00:19, 8634.96it/s] 58%|█████▊    | 232827/400000 [00:27<00:19, 8603.32it/s] 58%|█████▊    | 233689/400000 [00:27<00:19, 8605.89it/s] 59%|█████▊    | 234550/400000 [00:27<00:19, 8590.92it/s] 59%|█████▉    | 235411/400000 [00:27<00:19, 8596.08it/s] 59%|█████▉    | 236280/400000 [00:27<00:18, 8621.28it/s] 59%|█████▉    | 237143/400000 [00:28<00:18, 8621.91it/s] 60%|█████▉    | 238011/400000 [00:28<00:18, 8636.69it/s] 60%|█████▉    | 238875/400000 [00:28<00:18, 8635.35it/s] 60%|█████▉    | 239739/400000 [00:28<00:18, 8452.22it/s] 60%|██████    | 240603/400000 [00:28<00:18, 8507.47it/s] 60%|██████    | 241474/400000 [00:28<00:18, 8566.10it/s] 61%|██████    | 242332/400000 [00:28<00:18, 8566.39it/s] 61%|██████    | 243190/400000 [00:28<00:18, 8447.65it/s] 61%|██████    | 244057/400000 [00:28<00:18, 8512.74it/s] 61%|██████    | 244909/400000 [00:28<00:18, 8513.54it/s] 61%|██████▏   | 245768/400000 [00:29<00:18, 8533.14it/s] 62%|██████▏   | 246622/400000 [00:29<00:18, 8431.20it/s] 62%|██████▏   | 247469/400000 [00:29<00:18, 8440.49it/s] 62%|██████▏   | 248314/400000 [00:29<00:17, 8438.80it/s] 62%|██████▏   | 249184/400000 [00:29<00:17, 8513.72it/s] 63%|██████▎   | 250047/400000 [00:29<00:17, 8545.80it/s] 63%|██████▎   | 250902/400000 [00:29<00:17, 8481.65it/s] 63%|██████▎   | 251751/400000 [00:29<00:17, 8452.56it/s] 63%|██████▎   | 252632/400000 [00:29<00:17, 8556.11it/s] 63%|██████▎   | 253504/400000 [00:29<00:17, 8603.62it/s] 64%|██████▎   | 254365/400000 [00:30<00:17, 8544.62it/s] 64%|██████▍   | 255220/400000 [00:30<00:17, 8492.96it/s] 64%|██████▍   | 256075/400000 [00:30<00:16, 8507.28it/s] 64%|██████▍   | 256950/400000 [00:30<00:16, 8577.22it/s] 64%|██████▍   | 257815/400000 [00:30<00:16, 8598.33it/s] 65%|██████▍   | 258676/400000 [00:30<00:16, 8567.06it/s] 65%|██████▍   | 259533/400000 [00:30<00:16, 8530.77it/s] 65%|██████▌   | 260387/400000 [00:30<00:16, 8523.59it/s] 65%|██████▌   | 261240/400000 [00:30<00:16, 8482.48it/s] 66%|██████▌   | 262105/400000 [00:30<00:16, 8531.74it/s] 66%|██████▌   | 262959/400000 [00:31<00:16, 8485.84it/s] 66%|██████▌   | 263808/400000 [00:31<00:16, 8445.06it/s] 66%|██████▌   | 264653/400000 [00:31<00:16, 8377.91it/s] 66%|██████▋   | 265515/400000 [00:31<00:15, 8448.10it/s] 67%|██████▋   | 266361/400000 [00:31<00:15, 8449.60it/s] 67%|██████▋   | 267217/400000 [00:31<00:15, 8480.49it/s] 67%|██████▋   | 268066/400000 [00:31<00:15, 8420.07it/s] 67%|██████▋   | 268909/400000 [00:31<00:15, 8420.03it/s] 67%|██████▋   | 269761/400000 [00:31<00:15, 8448.24it/s] 68%|██████▊   | 270608/400000 [00:31<00:15, 8453.65it/s] 68%|██████▊   | 271461/400000 [00:32<00:15, 8475.71it/s] 68%|██████▊   | 272309/400000 [00:32<00:15, 8450.94it/s] 68%|██████▊   | 273155/400000 [00:32<00:15, 8385.96it/s] 68%|██████▊   | 273994/400000 [00:32<00:15, 8382.63it/s] 69%|██████▊   | 274833/400000 [00:32<00:14, 8379.77it/s] 69%|██████▉   | 275695/400000 [00:32<00:14, 8450.22it/s] 69%|██████▉   | 276541/400000 [00:32<00:14, 8426.45it/s] 69%|██████▉   | 277384/400000 [00:32<00:14, 8405.03it/s] 70%|██████▉   | 278225/400000 [00:32<00:14, 8353.43it/s] 70%|██████▉   | 279076/400000 [00:32<00:14, 8398.87it/s] 70%|██████▉   | 279938/400000 [00:33<00:14, 8463.53it/s] 70%|███████   | 280785/400000 [00:33<00:14, 8436.98it/s] 70%|███████   | 281632/400000 [00:33<00:14, 8444.33it/s] 71%|███████   | 282478/400000 [00:33<00:13, 8447.03it/s] 71%|███████   | 283328/400000 [00:33<00:13, 8460.69it/s] 71%|███████   | 284182/400000 [00:33<00:13, 8481.95it/s] 71%|███████▏  | 285032/400000 [00:33<00:13, 8485.53it/s] 71%|███████▏  | 285886/400000 [00:33<00:13, 8501.43it/s] 72%|███████▏  | 286753/400000 [00:33<00:13, 8549.70it/s] 72%|███████▏  | 287609/400000 [00:33<00:13, 8551.38it/s] 72%|███████▏  | 288470/400000 [00:34<00:13, 8566.99it/s] 72%|███████▏  | 289328/400000 [00:34<00:12, 8569.53it/s] 73%|███████▎  | 290185/400000 [00:34<00:12, 8533.22it/s] 73%|███████▎  | 291044/400000 [00:34<00:12, 8548.25it/s] 73%|███████▎  | 291902/400000 [00:34<00:12, 8554.97it/s] 73%|███████▎  | 292758/400000 [00:34<00:12, 8480.64it/s] 73%|███████▎  | 293609/400000 [00:34<00:12, 8487.48it/s] 74%|███████▎  | 294458/400000 [00:34<00:12, 8422.73it/s] 74%|███████▍  | 295319/400000 [00:34<00:12, 8477.72it/s] 74%|███████▍  | 296175/400000 [00:34<00:12, 8499.52it/s] 74%|███████▍  | 297026/400000 [00:35<00:12, 8410.23it/s] 74%|███████▍  | 297868/400000 [00:35<00:12, 8404.36it/s] 75%|███████▍  | 298709/400000 [00:35<00:12, 8391.78it/s] 75%|███████▍  | 299569/400000 [00:35<00:11, 8451.97it/s] 75%|███████▌  | 300426/400000 [00:35<00:11, 8486.04it/s] 75%|███████▌  | 301281/400000 [00:35<00:11, 8505.01it/s] 76%|███████▌  | 302132/400000 [00:35<00:11, 8500.02it/s] 76%|███████▌  | 302983/400000 [00:35<00:11, 8460.54it/s] 76%|███████▌  | 303849/400000 [00:35<00:11, 8517.35it/s] 76%|███████▌  | 304706/400000 [00:35<00:11, 8532.01it/s] 76%|███████▋  | 305572/400000 [00:36<00:11, 8567.51it/s] 77%|███████▋  | 306429/400000 [00:36<00:11, 8503.06it/s] 77%|███████▋  | 307281/400000 [00:36<00:10, 8506.83it/s] 77%|███████▋  | 308141/400000 [00:36<00:10, 8532.85it/s] 77%|███████▋  | 309006/400000 [00:36<00:10, 8566.55it/s] 77%|███████▋  | 309867/400000 [00:36<00:10, 8576.81it/s] 78%|███████▊  | 310725/400000 [00:36<00:10, 8527.52it/s] 78%|███████▊  | 311578/400000 [00:36<00:10, 8454.89it/s] 78%|███████▊  | 312424/400000 [00:36<00:11, 7932.92it/s] 78%|███████▊  | 313281/400000 [00:37<00:10, 8112.63it/s] 79%|███████▊  | 314119/400000 [00:37<00:10, 8190.79it/s] 79%|███████▊  | 314943/400000 [00:37<00:10, 8189.51it/s] 79%|███████▉  | 315788/400000 [00:37<00:10, 8263.95it/s] 79%|███████▉  | 316647/400000 [00:37<00:09, 8357.72it/s] 79%|███████▉  | 317502/400000 [00:37<00:09, 8412.62it/s] 80%|███████▉  | 318361/400000 [00:37<00:09, 8463.20it/s] 80%|███████▉  | 319209/400000 [00:37<00:09, 8444.60it/s] 80%|████████  | 320055/400000 [00:37<00:09, 8428.89it/s] 80%|████████  | 320908/400000 [00:37<00:09, 8457.53it/s] 80%|████████  | 321768/400000 [00:38<00:09, 8497.11it/s] 81%|████████  | 322629/400000 [00:38<00:09, 8528.30it/s] 81%|████████  | 323483/400000 [00:38<00:08, 8517.86it/s] 81%|████████  | 324335/400000 [00:38<00:08, 8507.90it/s] 81%|████████▏ | 325188/400000 [00:38<00:08, 8512.75it/s] 82%|████████▏ | 326042/400000 [00:38<00:08, 8519.41it/s] 82%|████████▏ | 326903/400000 [00:38<00:08, 8545.88it/s] 82%|████████▏ | 327758/400000 [00:38<00:08, 8541.54it/s] 82%|████████▏ | 328613/400000 [00:38<00:08, 8468.32it/s] 82%|████████▏ | 329470/400000 [00:38<00:08, 8498.19it/s] 83%|████████▎ | 330322/400000 [00:39<00:08, 8504.00it/s] 83%|████████▎ | 331191/400000 [00:39<00:08, 8558.15it/s] 83%|████████▎ | 332060/400000 [00:39<00:07, 8597.02it/s] 83%|████████▎ | 332920/400000 [00:39<00:07, 8581.11it/s] 83%|████████▎ | 333779/400000 [00:39<00:07, 8541.00it/s] 84%|████████▎ | 334645/400000 [00:39<00:07, 8575.00it/s] 84%|████████▍ | 335512/400000 [00:39<00:07, 8602.61it/s] 84%|████████▍ | 336373/400000 [00:39<00:07, 8532.13it/s] 84%|████████▍ | 337227/400000 [00:39<00:07, 8480.83it/s] 85%|████████▍ | 338099/400000 [00:39<00:07, 8550.66it/s] 85%|████████▍ | 338969/400000 [00:40<00:07, 8592.20it/s] 85%|████████▍ | 339829/400000 [00:40<00:07, 8504.52it/s] 85%|████████▌ | 340680/400000 [00:40<00:06, 8481.90it/s] 85%|████████▌ | 341529/400000 [00:40<00:06, 8445.60it/s] 86%|████████▌ | 342374/400000 [00:40<00:06, 8423.13it/s] 86%|████████▌ | 343255/400000 [00:40<00:06, 8533.12it/s] 86%|████████▌ | 344130/400000 [00:40<00:06, 8596.17it/s] 86%|████████▌ | 344996/400000 [00:40<00:06, 8613.55it/s] 86%|████████▋ | 345858/400000 [00:40<00:06, 8513.47it/s] 87%|████████▋ | 346730/400000 [00:40<00:06, 8574.15it/s] 87%|████████▋ | 347600/400000 [00:41<00:06, 8609.67it/s] 87%|████████▋ | 348478/400000 [00:41<00:05, 8657.87it/s] 87%|████████▋ | 349345/400000 [00:41<00:05, 8622.35it/s] 88%|████████▊ | 350208/400000 [00:41<00:05, 8573.99it/s] 88%|████████▊ | 351073/400000 [00:41<00:05, 8596.05it/s] 88%|████████▊ | 351947/400000 [00:41<00:05, 8638.44it/s] 88%|████████▊ | 352817/400000 [00:41<00:05, 8655.38it/s] 88%|████████▊ | 353683/400000 [00:41<00:05, 8589.18it/s] 89%|████████▊ | 354543/400000 [00:41<00:05, 8522.36it/s] 89%|████████▉ | 355396/400000 [00:41<00:05, 8524.10it/s] 89%|████████▉ | 356268/400000 [00:42<00:05, 8579.65it/s] 89%|████████▉ | 357135/400000 [00:42<00:04, 8605.66it/s] 89%|████████▉ | 357996/400000 [00:42<00:04, 8605.27it/s] 90%|████████▉ | 358857/400000 [00:42<00:04, 8559.49it/s] 90%|████████▉ | 359723/400000 [00:42<00:04, 8589.06it/s] 90%|█████████ | 360597/400000 [00:42<00:04, 8631.34it/s] 90%|█████████ | 361462/400000 [00:42<00:04, 8635.55it/s] 91%|█████████ | 362326/400000 [00:42<00:04, 8373.85it/s] 91%|█████████ | 363182/400000 [00:42<00:04, 8428.65it/s] 91%|█████████ | 364035/400000 [00:42<00:04, 8456.95it/s] 91%|█████████ | 364893/400000 [00:43<00:04, 8493.48it/s] 91%|█████████▏| 365744/400000 [00:43<00:04, 8446.02it/s] 92%|█████████▏| 366594/400000 [00:43<00:03, 8461.25it/s] 92%|█████████▏| 367460/400000 [00:43<00:03, 8518.98it/s] 92%|█████████▏| 368317/400000 [00:43<00:03, 8533.55it/s] 92%|█████████▏| 369171/400000 [00:43<00:03, 8529.86it/s] 93%|█████████▎| 370025/400000 [00:43<00:03, 8493.27it/s] 93%|█████████▎| 370881/400000 [00:43<00:03, 8513.01it/s] 93%|█████████▎| 371733/400000 [00:43<00:03, 8468.82it/s] 93%|█████████▎| 372597/400000 [00:43<00:03, 8517.21it/s] 93%|█████████▎| 373449/400000 [00:44<00:03, 8517.14it/s] 94%|█████████▎| 374312/400000 [00:44<00:03, 8550.27it/s] 94%|█████████▍| 375168/400000 [00:44<00:02, 8537.94it/s] 94%|█████████▍| 376022/400000 [00:44<00:02, 8507.99it/s] 94%|█████████▍| 376873/400000 [00:44<00:02, 8491.84it/s] 94%|█████████▍| 377734/400000 [00:44<00:02, 8526.64it/s] 95%|█████████▍| 378587/400000 [00:44<00:02, 8506.81it/s] 95%|█████████▍| 379438/400000 [00:44<00:02, 8459.73it/s] 95%|█████████▌| 380285/400000 [00:44<00:02, 8412.45it/s] 95%|█████████▌| 381147/400000 [00:44<00:02, 8472.76it/s] 96%|█████████▌| 382004/400000 [00:45<00:02, 8501.51it/s] 96%|█████████▌| 382856/400000 [00:45<00:02, 8505.72it/s] 96%|█████████▌| 383707/400000 [00:45<00:01, 8450.93it/s] 96%|█████████▌| 384553/400000 [00:45<00:01, 8287.61it/s] 96%|█████████▋| 385411/400000 [00:45<00:01, 8370.66it/s] 97%|█████████▋| 386280/400000 [00:45<00:01, 8463.02it/s] 97%|█████████▋| 387128/400000 [00:45<00:01, 8464.99it/s] 97%|█████████▋| 387976/400000 [00:45<00:01, 8452.84it/s] 97%|█████████▋| 388822/400000 [00:45<00:01, 8396.75it/s] 97%|█████████▋| 389690/400000 [00:45<00:01, 8477.24it/s] 98%|█████████▊| 390551/400000 [00:46<00:01, 8515.87it/s] 98%|█████████▊| 391413/400000 [00:46<00:01, 8546.52it/s] 98%|█████████▊| 392268/400000 [00:46<00:00, 8500.72it/s] 98%|█████████▊| 393119/400000 [00:46<00:00, 8476.26it/s] 98%|█████████▊| 393980/400000 [00:46<00:00, 8513.65it/s] 99%|█████████▊| 394835/400000 [00:46<00:00, 8521.79it/s] 99%|█████████▉| 395703/400000 [00:46<00:00, 8566.90it/s] 99%|█████████▉| 396560/400000 [00:46<00:00, 8540.32it/s] 99%|█████████▉| 397415/400000 [00:46<00:00, 8339.14it/s]100%|█████████▉| 398266/400000 [00:46<00:00, 8388.70it/s]100%|█████████▉| 399121/400000 [00:47<00:00, 8434.26it/s]100%|█████████▉| 399985/400000 [00:47<00:00, 8494.23it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8475.55it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1f00962ba8>, <torchtext.data.dataset.TabularDataset object at 0x7f1f00962cf8>, <torchtext.vocab.Vocab object at 0x7f1f00962c18>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.68 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.68 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.68 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.22 file/s]2020-07-04 18:20:09.730414: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-04 18:20:09.735110: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-07-04 18:20:09.735326: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55977c9d2860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 18:20:09.735342: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 464437.94it/s] 88%|████████▊ | 8708096/9912422 [00:00<00:01, 661961.01it/s]9920512it [00:00, 45330704.72it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1436549.01it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 965724.08it/s]1654784it [00:00, 12535950.97it/s]                         
0it [00:00, ?it/s]8192it [00:00, 224629.24it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
