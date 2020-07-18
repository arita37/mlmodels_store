
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c6fa6f43d1647e24f9360f90bb6efd98b22c7435', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'c6fa6f43d1647e24f9360f90bb6efd98b22c7435', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c6fa6f43d1647e24f9360f90bb6efd98b22c7435

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c6fa6f43d1647e24f9360f90bb6efd98b22c7435

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/c6fa6f43d1647e24f9360f90bb6efd98b22c7435

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ffa005759d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ffa005759d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ffa6b1e0510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ffa6b1e0510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ffa8a42fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ffa8a42fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ffa18514950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ffa18514950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ffa18514950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3596288/9912422 [00:00<00:00, 35553074.48it/s]9920512it [00:00, 34733605.04it/s]                             
0it [00:00, ?it/s]32768it [00:00, 505686.27it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477852.41it/s]1654784it [00:00, 11896192.81it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193568.36it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ffa16a5f630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ffa16a848d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ffa18514598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ffa18514598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ffa18514598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.11 MiB/s][A

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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 46.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.58 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.20s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 73.58 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.20s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.20s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.12 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.21s/ url]
0 examples [00:00, ? examples/s]2020-07-18 18:49:16.401913: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-18 18:49:16.417129: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-18 18:49:16.417395: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559a75eeb530 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 18:49:16.417420: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
39 examples [00:00, 385.97 examples/s]129 examples [00:00, 465.44 examples/s]217 examples [00:00, 541.17 examples/s]305 examples [00:00, 610.83 examples/s]387 examples [00:00, 661.17 examples/s]480 examples [00:00, 722.60 examples/s]575 examples [00:00, 777.55 examples/s]663 examples [00:00, 803.42 examples/s]749 examples [00:00, 818.70 examples/s]840 examples [00:01, 843.69 examples/s]926 examples [00:01, 811.76 examples/s]1013 examples [00:01, 827.97 examples/s]1098 examples [00:01, 831.77 examples/s]1185 examples [00:01, 840.73 examples/s]1271 examples [00:01, 846.08 examples/s]1356 examples [00:01, 841.30 examples/s]1441 examples [00:01, 790.92 examples/s]1525 examples [00:01, 802.86 examples/s]1606 examples [00:01, 786.76 examples/s]1686 examples [00:02, 776.80 examples/s]1775 examples [00:02, 806.81 examples/s]1863 examples [00:02, 826.31 examples/s]1951 examples [00:02, 839.98 examples/s]2037 examples [00:02, 844.86 examples/s]2122 examples [00:02, 845.70 examples/s]2209 examples [00:02, 850.77 examples/s]2295 examples [00:02, 844.19 examples/s]2381 examples [00:02, 846.77 examples/s]2466 examples [00:02, 847.31 examples/s]2553 examples [00:03, 852.37 examples/s]2639 examples [00:03, 841.48 examples/s]2724 examples [00:03, 799.05 examples/s]2805 examples [00:03, 794.49 examples/s]2894 examples [00:03, 819.14 examples/s]2982 examples [00:03, 836.01 examples/s]3068 examples [00:03, 840.09 examples/s]3153 examples [00:03, 839.61 examples/s]3238 examples [00:03, 828.29 examples/s]3325 examples [00:04, 840.03 examples/s]3414 examples [00:04, 852.09 examples/s]3501 examples [00:04, 856.56 examples/s]3588 examples [00:04, 858.87 examples/s]3675 examples [00:04, 860.17 examples/s]3763 examples [00:04, 865.62 examples/s]3850 examples [00:04, 856.98 examples/s]3936 examples [00:04, 850.84 examples/s]4022 examples [00:04, 848.91 examples/s]4107 examples [00:04, 848.35 examples/s]4193 examples [00:05, 849.72 examples/s]4278 examples [00:05, 843.83 examples/s]4367 examples [00:05, 855.16 examples/s]4456 examples [00:05, 864.08 examples/s]4543 examples [00:05, 861.59 examples/s]4630 examples [00:05, 863.02 examples/s]4720 examples [00:05, 871.46 examples/s]4809 examples [00:05, 876.14 examples/s]4897 examples [00:05, 869.80 examples/s]4986 examples [00:05, 874.20 examples/s]5074 examples [00:06, 850.36 examples/s]5162 examples [00:06, 858.26 examples/s]5251 examples [00:06, 866.23 examples/s]5339 examples [00:06, 869.15 examples/s]5426 examples [00:06, 869.21 examples/s]5514 examples [00:06, 870.35 examples/s]5604 examples [00:06, 878.34 examples/s]5692 examples [00:06, 878.66 examples/s]5780 examples [00:06, 867.73 examples/s]5870 examples [00:06, 876.71 examples/s]5960 examples [00:07, 882.84 examples/s]6054 examples [00:07, 898.23 examples/s]6144 examples [00:07, 886.99 examples/s]6234 examples [00:07, 890.56 examples/s]6324 examples [00:07, 884.15 examples/s]6413 examples [00:07, 885.76 examples/s]6502 examples [00:07, 878.20 examples/s]6591 examples [00:07, 881.42 examples/s]6680 examples [00:07, 872.38 examples/s]6768 examples [00:07, 839.90 examples/s]6853 examples [00:08, 833.12 examples/s]6941 examples [00:08, 846.30 examples/s]7029 examples [00:08, 855.60 examples/s]7115 examples [00:08, 821.19 examples/s]7202 examples [00:08, 834.89 examples/s]7289 examples [00:08, 844.11 examples/s]7374 examples [00:08, 826.93 examples/s]7467 examples [00:08, 855.20 examples/s]7556 examples [00:08, 864.37 examples/s]7644 examples [00:09, 866.29 examples/s]7734 examples [00:09, 874.08 examples/s]7825 examples [00:09, 881.86 examples/s]7914 examples [00:09, 878.15 examples/s]8002 examples [00:09, 853.36 examples/s]8088 examples [00:09, 845.98 examples/s]8177 examples [00:09, 857.00 examples/s]8267 examples [00:09, 866.84 examples/s]8354 examples [00:09, 867.20 examples/s]8441 examples [00:09, 863.39 examples/s]8528 examples [00:10, 863.84 examples/s]8615 examples [00:10, 861.69 examples/s]8702 examples [00:10, 853.74 examples/s]8791 examples [00:10, 862.84 examples/s]8878 examples [00:10, 858.78 examples/s]8967 examples [00:10, 865.65 examples/s]9054 examples [00:10, 858.03 examples/s]9140 examples [00:10, 848.10 examples/s]9227 examples [00:10, 852.73 examples/s]9315 examples [00:10, 860.02 examples/s]9402 examples [00:11, 852.39 examples/s]9490 examples [00:11, 857.95 examples/s]9578 examples [00:11, 864.22 examples/s]9665 examples [00:11, 841.00 examples/s]9757 examples [00:11, 861.71 examples/s]9844 examples [00:11, 849.08 examples/s]9930 examples [00:11, 843.76 examples/s]10015 examples [00:11, 779.10 examples/s]10102 examples [00:11, 803.87 examples/s]10184 examples [00:12, 806.22 examples/s]10277 examples [00:12, 838.86 examples/s]10362 examples [00:12, 817.59 examples/s]10445 examples [00:12, 801.44 examples/s]10526 examples [00:12, 775.63 examples/s]10610 examples [00:12, 789.31 examples/s]10696 examples [00:12, 808.24 examples/s]10778 examples [00:12, 793.44 examples/s]10866 examples [00:12, 817.00 examples/s]10950 examples [00:12, 822.21 examples/s]11034 examples [00:13, 824.78 examples/s]11118 examples [00:13, 826.73 examples/s]11205 examples [00:13, 838.79 examples/s]11290 examples [00:13, 841.49 examples/s]11375 examples [00:13, 827.40 examples/s]11461 examples [00:13, 836.19 examples/s]11545 examples [00:13, 788.98 examples/s]11625 examples [00:13, 782.44 examples/s]11712 examples [00:13, 804.57 examples/s]11799 examples [00:13, 821.67 examples/s]11882 examples [00:14, 823.99 examples/s]11965 examples [00:14, 813.44 examples/s]12052 examples [00:14, 829.51 examples/s]12139 examples [00:14, 839.49 examples/s]12224 examples [00:14, 841.33 examples/s]12311 examples [00:14, 849.29 examples/s]12400 examples [00:14, 860.70 examples/s]12487 examples [00:14, 855.33 examples/s]12575 examples [00:14, 860.70 examples/s]12663 examples [00:14, 865.10 examples/s]12754 examples [00:15, 876.53 examples/s]12842 examples [00:15, 836.46 examples/s]12927 examples [00:15, 836.53 examples/s]13011 examples [00:15, 824.36 examples/s]13104 examples [00:15, 853.17 examples/s]13195 examples [00:15, 869.07 examples/s]13284 examples [00:15, 873.34 examples/s]13373 examples [00:15, 875.84 examples/s]13461 examples [00:15, 862.30 examples/s]13548 examples [00:16, 863.63 examples/s]13635 examples [00:16, 838.79 examples/s]13723 examples [00:16, 849.22 examples/s]13809 examples [00:16, 839.56 examples/s]13894 examples [00:16, 815.98 examples/s]13978 examples [00:16, 820.75 examples/s]14063 examples [00:16, 828.19 examples/s]14152 examples [00:16, 844.42 examples/s]14237 examples [00:16, 822.60 examples/s]14320 examples [00:16, 808.42 examples/s]14406 examples [00:17, 822.21 examples/s]14489 examples [00:17, 803.37 examples/s]14579 examples [00:17, 829.29 examples/s]14664 examples [00:17, 835.12 examples/s]14749 examples [00:17, 837.29 examples/s]14833 examples [00:17, 835.95 examples/s]14923 examples [00:17, 852.30 examples/s]15009 examples [00:17, 818.52 examples/s]15093 examples [00:17, 822.22 examples/s]15177 examples [00:18, 823.57 examples/s]15260 examples [00:18, 811.74 examples/s]15344 examples [00:18, 819.86 examples/s]15432 examples [00:18, 835.30 examples/s]15516 examples [00:18, 831.35 examples/s]15600 examples [00:18, 832.08 examples/s]15684 examples [00:18, 829.60 examples/s]15768 examples [00:18, 830.86 examples/s]15854 examples [00:18, 837.22 examples/s]15945 examples [00:18, 855.12 examples/s]16031 examples [00:19, 853.71 examples/s]16117 examples [00:19, 853.71 examples/s]16207 examples [00:19, 865.02 examples/s]16297 examples [00:19, 873.80 examples/s]16389 examples [00:19, 884.20 examples/s]16478 examples [00:19, 866.80 examples/s]16565 examples [00:19, 857.10 examples/s]16654 examples [00:19, 864.25 examples/s]16742 examples [00:19, 866.81 examples/s]16830 examples [00:19, 869.68 examples/s]16918 examples [00:20, 864.90 examples/s]17006 examples [00:20, 868.39 examples/s]17094 examples [00:20, 871.42 examples/s]17182 examples [00:20, 870.59 examples/s]17270 examples [00:20, 869.19 examples/s]17357 examples [00:20, 851.91 examples/s]17445 examples [00:20, 858.30 examples/s]17535 examples [00:20, 868.78 examples/s]17625 examples [00:20, 874.92 examples/s]17714 examples [00:20, 877.29 examples/s]17802 examples [00:21, 873.19 examples/s]17890 examples [00:21, 867.41 examples/s]17980 examples [00:21, 876.04 examples/s]18068 examples [00:21, 876.05 examples/s]18156 examples [00:21, 872.17 examples/s]18245 examples [00:21, 875.30 examples/s]18333 examples [00:21, 875.30 examples/s]18422 examples [00:21, 879.52 examples/s]18510 examples [00:21, 864.81 examples/s]18597 examples [00:21, 860.56 examples/s]18684 examples [00:22, 855.01 examples/s]18772 examples [00:22, 860.66 examples/s]18859 examples [00:22, 815.14 examples/s]18942 examples [00:22, 808.82 examples/s]19028 examples [00:22, 821.07 examples/s]19118 examples [00:22, 841.37 examples/s]19207 examples [00:22, 853.66 examples/s]19296 examples [00:22, 862.16 examples/s]19384 examples [00:22, 864.83 examples/s]19471 examples [00:23, 864.92 examples/s]19558 examples [00:23, 864.21 examples/s]19646 examples [00:23, 867.51 examples/s]19733 examples [00:23, 855.43 examples/s]19821 examples [00:23, 860.02 examples/s]19908 examples [00:23, 833.35 examples/s]19994 examples [00:23, 838.57 examples/s]20079 examples [00:23, 792.63 examples/s]20160 examples [00:23, 797.26 examples/s]20244 examples [00:23, 808.30 examples/s]20329 examples [00:24, 818.72 examples/s]20417 examples [00:24, 833.69 examples/s]20503 examples [00:24, 839.64 examples/s]20588 examples [00:24, 840.31 examples/s]20673 examples [00:24, 841.05 examples/s]20760 examples [00:24, 846.91 examples/s]20847 examples [00:24, 853.30 examples/s]20937 examples [00:24, 860.66 examples/s]21026 examples [00:24, 864.85 examples/s]21113 examples [00:24, 864.69 examples/s]21200 examples [00:25, 832.88 examples/s]21284 examples [00:25, 830.92 examples/s]21368 examples [00:25, 828.47 examples/s]21451 examples [00:25, 803.32 examples/s]21533 examples [00:25, 801.53 examples/s]21614 examples [00:25, 790.02 examples/s]21697 examples [00:25, 800.04 examples/s]21786 examples [00:25, 823.84 examples/s]21869 examples [00:25, 819.59 examples/s]21956 examples [00:25, 831.02 examples/s]22040 examples [00:26, 819.65 examples/s]22132 examples [00:26, 846.92 examples/s]22218 examples [00:26, 825.74 examples/s]22306 examples [00:26, 840.32 examples/s]22394 examples [00:26, 850.43 examples/s]22480 examples [00:26, 838.10 examples/s]22565 examples [00:26, 841.38 examples/s]22650 examples [00:26, 828.81 examples/s]22734 examples [00:26, 829.35 examples/s]22818 examples [00:27, 829.36 examples/s]22902 examples [00:27, 823.70 examples/s]22988 examples [00:27, 831.89 examples/s]23075 examples [00:27, 842.48 examples/s]23162 examples [00:27, 849.23 examples/s]23250 examples [00:27, 856.32 examples/s]23336 examples [00:27, 855.53 examples/s]23422 examples [00:27, 845.49 examples/s]23507 examples [00:27, 817.97 examples/s]23596 examples [00:27, 836.26 examples/s]23685 examples [00:28, 851.41 examples/s]23774 examples [00:28, 860.25 examples/s]23861 examples [00:28, 845.84 examples/s]23948 examples [00:28, 851.94 examples/s]24035 examples [00:28, 856.22 examples/s]24121 examples [00:28, 846.05 examples/s]24206 examples [00:28, 846.57 examples/s]24291 examples [00:28, 841.14 examples/s]24380 examples [00:28, 853.54 examples/s]24466 examples [00:28, 851.48 examples/s]24552 examples [00:29, 851.98 examples/s]24638 examples [00:29, 842.08 examples/s]24728 examples [00:29, 858.08 examples/s]24816 examples [00:29, 863.04 examples/s]24904 examples [00:29, 866.18 examples/s]24991 examples [00:29, 861.33 examples/s]25078 examples [00:29, 851.71 examples/s]25164 examples [00:29, 837.21 examples/s]25248 examples [00:29, 796.05 examples/s]25338 examples [00:30, 824.03 examples/s]25428 examples [00:30, 843.54 examples/s]25518 examples [00:30, 858.68 examples/s]25605 examples [00:30, 860.79 examples/s]25692 examples [00:30, 856.18 examples/s]25778 examples [00:30, 823.41 examples/s]25863 examples [00:30, 828.85 examples/s]25947 examples [00:30, 830.00 examples/s]26032 examples [00:30, 832.96 examples/s]26116 examples [00:30, 834.71 examples/s]26200 examples [00:31, 828.15 examples/s]26283 examples [00:31, 820.56 examples/s]26368 examples [00:31, 828.57 examples/s]26455 examples [00:31, 839.47 examples/s]26542 examples [00:31, 846.53 examples/s]26627 examples [00:31, 840.47 examples/s]26715 examples [00:31, 850.72 examples/s]26801 examples [00:31, 834.41 examples/s]26886 examples [00:31, 837.42 examples/s]26973 examples [00:31, 844.22 examples/s]27058 examples [00:32, 837.81 examples/s]27143 examples [00:32, 840.97 examples/s]27228 examples [00:32, 839.99 examples/s]27316 examples [00:32, 850.18 examples/s]27403 examples [00:32, 855.06 examples/s]27490 examples [00:32, 857.83 examples/s]27578 examples [00:32, 863.88 examples/s]27666 examples [00:32, 867.58 examples/s]27753 examples [00:32, 865.10 examples/s]27840 examples [00:32, 849.38 examples/s]27927 examples [00:33, 852.89 examples/s]28017 examples [00:33, 865.45 examples/s]28104 examples [00:33, 838.91 examples/s]28189 examples [00:33, 840.82 examples/s]28275 examples [00:33, 844.27 examples/s]28363 examples [00:33, 854.10 examples/s]28451 examples [00:33, 861.57 examples/s]28538 examples [00:33, 853.70 examples/s]28624 examples [00:33, 834.44 examples/s]28710 examples [00:33, 840.65 examples/s]28795 examples [00:34, 840.38 examples/s]28883 examples [00:34, 851.38 examples/s]28969 examples [00:34, 852.25 examples/s]29057 examples [00:34, 858.07 examples/s]29146 examples [00:34, 866.64 examples/s]29233 examples [00:34, 836.47 examples/s]29323 examples [00:34, 854.34 examples/s]29409 examples [00:34, 854.71 examples/s]29498 examples [00:34, 863.02 examples/s]29589 examples [00:35, 876.00 examples/s]29677 examples [00:35, 877.18 examples/s]29767 examples [00:35, 883.54 examples/s]29857 examples [00:35, 886.04 examples/s]29946 examples [00:35, 870.50 examples/s]30034 examples [00:35, 812.63 examples/s]30118 examples [00:35, 820.64 examples/s]30209 examples [00:35, 843.92 examples/s]30294 examples [00:35, 845.40 examples/s]30379 examples [00:35, 812.30 examples/s]30461 examples [00:36, 757.00 examples/s]30549 examples [00:36, 788.03 examples/s]30633 examples [00:36, 801.07 examples/s]30719 examples [00:36, 816.53 examples/s]30807 examples [00:36, 832.49 examples/s]30891 examples [00:36, 831.16 examples/s]30977 examples [00:36, 837.37 examples/s]31066 examples [00:36, 850.54 examples/s]31152 examples [00:36, 851.47 examples/s]31242 examples [00:36, 865.23 examples/s]31329 examples [00:37, 857.74 examples/s]31418 examples [00:37, 866.97 examples/s]31506 examples [00:37, 868.44 examples/s]31596 examples [00:37, 876.30 examples/s]31684 examples [00:37, 875.09 examples/s]31772 examples [00:37, 875.63 examples/s]31860 examples [00:37, 862.72 examples/s]31947 examples [00:37, 862.73 examples/s]32034 examples [00:37, 858.08 examples/s]32120 examples [00:38, 846.91 examples/s]32206 examples [00:38, 849.72 examples/s]32293 examples [00:38, 853.37 examples/s]32380 examples [00:38, 858.12 examples/s]32468 examples [00:38, 863.76 examples/s]32557 examples [00:38, 868.54 examples/s]32644 examples [00:38, 865.01 examples/s]32731 examples [00:38, 844.50 examples/s]32816 examples [00:38, 832.38 examples/s]32905 examples [00:38, 847.61 examples/s]32993 examples [00:39, 854.77 examples/s]33079 examples [00:39, 843.81 examples/s]33166 examples [00:39, 850.89 examples/s]33252 examples [00:39, 852.70 examples/s]33339 examples [00:39, 856.26 examples/s]33425 examples [00:39, 853.66 examples/s]33513 examples [00:39, 860.07 examples/s]33600 examples [00:39, 859.29 examples/s]33686 examples [00:39, 856.70 examples/s]33774 examples [00:39, 863.16 examples/s]33862 examples [00:40, 867.60 examples/s]33949 examples [00:40, 854.80 examples/s]34036 examples [00:40, 858.27 examples/s]34123 examples [00:40, 859.32 examples/s]34212 examples [00:40, 865.63 examples/s]34300 examples [00:40, 869.82 examples/s]34388 examples [00:40, 865.36 examples/s]34476 examples [00:40, 868.78 examples/s]34565 examples [00:40, 873.05 examples/s]34653 examples [00:40, 855.59 examples/s]34744 examples [00:41, 870.11 examples/s]34832 examples [00:41, 855.02 examples/s]34921 examples [00:41, 863.61 examples/s]35008 examples [00:41, 796.42 examples/s]35095 examples [00:41, 816.22 examples/s]35183 examples [00:41, 833.13 examples/s]35273 examples [00:41, 851.00 examples/s]35359 examples [00:41, 853.24 examples/s]35446 examples [00:41, 856.72 examples/s]35535 examples [00:42, 864.67 examples/s]35623 examples [00:42, 868.06 examples/s]35710 examples [00:42, 860.03 examples/s]35797 examples [00:42, 861.83 examples/s]35885 examples [00:42, 865.83 examples/s]35972 examples [00:42, 866.25 examples/s]36059 examples [00:42, 830.00 examples/s]36145 examples [00:42, 836.97 examples/s]36231 examples [00:42, 843.26 examples/s]36316 examples [00:42, 842.01 examples/s]36401 examples [00:43, 837.10 examples/s]36490 examples [00:43, 852.17 examples/s]36579 examples [00:43, 862.60 examples/s]36668 examples [00:43, 857.18 examples/s]36754 examples [00:43, 855.53 examples/s]36843 examples [00:43, 863.14 examples/s]36932 examples [00:43, 868.32 examples/s]37021 examples [00:43, 872.69 examples/s]37109 examples [00:43, 864.78 examples/s]37199 examples [00:43, 872.82 examples/s]37287 examples [00:44, 841.84 examples/s]37374 examples [00:44, 839.00 examples/s]37459 examples [00:44, 835.05 examples/s]37546 examples [00:44, 842.68 examples/s]37633 examples [00:44, 849.80 examples/s]37719 examples [00:44, 843.85 examples/s]37805 examples [00:44, 847.34 examples/s]37890 examples [00:44, 811.13 examples/s]37974 examples [00:44, 818.14 examples/s]38057 examples [00:44, 812.87 examples/s]38140 examples [00:45, 817.26 examples/s]38222 examples [00:45, 785.88 examples/s]38303 examples [00:45, 792.61 examples/s]38385 examples [00:45, 800.31 examples/s]38466 examples [00:45, 797.07 examples/s]38546 examples [00:45, 789.68 examples/s]38631 examples [00:45, 806.70 examples/s]38715 examples [00:45, 816.06 examples/s]38800 examples [00:45, 823.39 examples/s]38883 examples [00:46, 814.74 examples/s]38972 examples [00:46, 833.28 examples/s]39056 examples [00:46, 817.06 examples/s]39144 examples [00:46, 834.00 examples/s]39228 examples [00:46, 829.23 examples/s]39312 examples [00:46, 809.48 examples/s]39398 examples [00:46, 822.97 examples/s]39484 examples [00:46, 833.08 examples/s]39568 examples [00:46, 792.39 examples/s]39648 examples [00:46, 773.58 examples/s]39730 examples [00:47, 786.55 examples/s]39810 examples [00:47, 789.53 examples/s]39893 examples [00:47, 801.10 examples/s]39980 examples [00:47, 818.10 examples/s]40063 examples [00:47, 749.42 examples/s]40143 examples [00:47, 761.68 examples/s]40234 examples [00:47, 795.64 examples/s]40323 examples [00:47, 820.41 examples/s]40409 examples [00:47, 829.33 examples/s]40493 examples [00:48, 824.95 examples/s]40582 examples [00:48, 841.50 examples/s]40671 examples [00:48, 852.94 examples/s]40757 examples [00:48, 852.53 examples/s]40845 examples [00:48, 859.68 examples/s]40932 examples [00:48, 858.18 examples/s]41019 examples [00:48, 859.46 examples/s]41108 examples [00:48, 865.59 examples/s]41195 examples [00:48, 859.50 examples/s]41282 examples [00:48, 853.54 examples/s]41368 examples [00:49, 850.14 examples/s]41459 examples [00:49, 867.15 examples/s]41551 examples [00:49, 881.26 examples/s]41640 examples [00:49, 854.60 examples/s]41729 examples [00:49, 863.32 examples/s]41816 examples [00:49, 854.60 examples/s]41902 examples [00:49, 837.05 examples/s]41991 examples [00:49, 849.82 examples/s]42077 examples [00:49, 844.10 examples/s]42164 examples [00:49, 851.47 examples/s]42251 examples [00:50, 854.36 examples/s]42338 examples [00:50, 855.94 examples/s]42424 examples [00:50, 834.90 examples/s]42512 examples [00:50, 846.24 examples/s]42597 examples [00:50, 846.88 examples/s]42682 examples [00:50, 808.79 examples/s]42771 examples [00:50, 831.36 examples/s]42858 examples [00:50, 840.30 examples/s]42943 examples [00:50, 839.06 examples/s]43028 examples [00:50, 824.01 examples/s]43111 examples [00:51, 814.33 examples/s]43196 examples [00:51, 823.62 examples/s]43286 examples [00:51, 843.59 examples/s]43371 examples [00:51, 836.84 examples/s]43460 examples [00:51, 851.85 examples/s]43550 examples [00:51, 863.98 examples/s]43639 examples [00:51, 869.18 examples/s]43727 examples [00:51, 870.81 examples/s]43815 examples [00:51, 860.73 examples/s]43902 examples [00:52, 847.74 examples/s]43989 examples [00:52, 854.10 examples/s]44082 examples [00:52, 873.05 examples/s]44170 examples [00:52, 873.60 examples/s]44258 examples [00:52, 849.21 examples/s]44344 examples [00:52, 848.25 examples/s]44432 examples [00:52, 855.86 examples/s]44520 examples [00:52, 861.29 examples/s]44610 examples [00:52, 869.60 examples/s]44698 examples [00:52, 872.40 examples/s]44786 examples [00:53, 866.48 examples/s]44873 examples [00:53, 863.66 examples/s]44960 examples [00:53, 864.73 examples/s]45048 examples [00:53, 868.60 examples/s]45136 examples [00:53, 870.47 examples/s]45224 examples [00:53, 872.68 examples/s]45312 examples [00:53, 870.63 examples/s]45400 examples [00:53, 869.82 examples/s]45487 examples [00:53, 860.43 examples/s]45576 examples [00:53, 868.76 examples/s]45663 examples [00:54, 866.80 examples/s]45750 examples [00:54, 858.90 examples/s]45837 examples [00:54, 861.62 examples/s]45924 examples [00:54, 849.94 examples/s]46012 examples [00:54, 856.87 examples/s]46098 examples [00:54, 846.81 examples/s]46187 examples [00:54, 858.01 examples/s]46277 examples [00:54, 869.98 examples/s]46365 examples [00:54, 863.39 examples/s]46452 examples [00:54, 860.83 examples/s]46539 examples [00:55, 806.17 examples/s]46621 examples [00:55, 793.87 examples/s]46709 examples [00:55, 816.32 examples/s]46797 examples [00:55, 833.40 examples/s]46887 examples [00:55, 850.89 examples/s]46973 examples [00:55, 852.82 examples/s]47059 examples [00:55, 839.20 examples/s]47148 examples [00:55, 850.65 examples/s]47234 examples [00:55, 842.53 examples/s]47320 examples [00:56, 847.51 examples/s]47405 examples [00:56, 824.08 examples/s]47493 examples [00:56, 839.06 examples/s]47582 examples [00:56, 850.98 examples/s]47671 examples [00:56, 860.39 examples/s]47760 examples [00:56, 868.71 examples/s]47848 examples [00:56, 871.80 examples/s]47938 examples [00:56, 878.44 examples/s]48027 examples [00:56, 879.65 examples/s]48116 examples [00:56, 878.24 examples/s]48204 examples [00:57, 872.52 examples/s]48294 examples [00:57, 877.77 examples/s]48384 examples [00:57, 882.40 examples/s]48473 examples [00:57, 884.33 examples/s]48562 examples [00:57, 878.22 examples/s]48651 examples [00:57, 880.61 examples/s]48740 examples [00:57, 862.01 examples/s]48829 examples [00:57, 869.32 examples/s]48917 examples [00:57, 870.27 examples/s]49006 examples [00:57, 874.55 examples/s]49094 examples [00:58, 869.66 examples/s]49182 examples [00:58, 859.44 examples/s]49273 examples [00:58, 872.56 examples/s]49362 examples [00:58, 876.44 examples/s]49450 examples [00:58, 872.02 examples/s]49538 examples [00:58, 872.27 examples/s]49626 examples [00:58, 869.72 examples/s]49715 examples [00:58, 873.75 examples/s]49803 examples [00:58, 869.05 examples/s]49890 examples [00:58, 869.05 examples/s]49978 examples [00:59, 870.03 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5088/50000 [00:00<00:00, 50878.96 examples/s] 31%|███       | 15559/50000 [00:00<00:00, 60156.31 examples/s] 52%|█████▏    | 26112/50000 [00:00<00:00, 69064.44 examples/s] 73%|███████▎  | 36748/50000 [00:00<00:00, 77183.15 examples/s] 94%|█████████▍| 47232/50000 [00:00<00:00, 83815.88 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.06 examples/s]155 examples [00:00, 736.12 examples/s]241 examples [00:00, 768.47 examples/s]333 examples [00:00, 806.56 examples/s]421 examples [00:00, 825.67 examples/s]510 examples [00:00, 843.35 examples/s]602 examples [00:00, 864.06 examples/s]691 examples [00:00, 870.36 examples/s]780 examples [00:00, 874.20 examples/s]867 examples [00:01, 869.78 examples/s]956 examples [00:01, 873.81 examples/s]1043 examples [00:01, 864.89 examples/s]1134 examples [00:01, 875.51 examples/s]1221 examples [00:01, 864.39 examples/s]1308 examples [00:01, 845.78 examples/s]1398 examples [00:01, 859.58 examples/s]1484 examples [00:01, 842.04 examples/s]1573 examples [00:01, 855.04 examples/s]1662 examples [00:01, 864.43 examples/s]1749 examples [00:02, 850.89 examples/s]1835 examples [00:02, 850.79 examples/s]1921 examples [00:02, 818.24 examples/s]2008 examples [00:02, 833.00 examples/s]2096 examples [00:02, 843.74 examples/s]2183 examples [00:02, 850.85 examples/s]2269 examples [00:02, 851.46 examples/s]2357 examples [00:02, 857.55 examples/s]2443 examples [00:02, 815.88 examples/s]2538 examples [00:02, 849.63 examples/s]2624 examples [00:03, 842.19 examples/s]2709 examples [00:03, 814.87 examples/s]2798 examples [00:03, 833.87 examples/s]2882 examples [00:03, 820.83 examples/s]2968 examples [00:03, 831.26 examples/s]3053 examples [00:03, 835.88 examples/s]3139 examples [00:03, 840.60 examples/s]3225 examples [00:03, 844.91 examples/s]3315 examples [00:03, 859.75 examples/s]3402 examples [00:04, 862.38 examples/s]3489 examples [00:04, 858.62 examples/s]3580 examples [00:04, 870.61 examples/s]3671 examples [00:04, 880.11 examples/s]3760 examples [00:04, 875.00 examples/s]3848 examples [00:04, 873.84 examples/s]3936 examples [00:04, 858.93 examples/s]4022 examples [00:04, 853.56 examples/s]4108 examples [00:04, 843.55 examples/s]4193 examples [00:04, 842.56 examples/s]4278 examples [00:05, 840.30 examples/s]4363 examples [00:05, 837.23 examples/s]4448 examples [00:05, 837.85 examples/s]4534 examples [00:05, 843.92 examples/s]4628 examples [00:05, 868.77 examples/s]4717 examples [00:05, 874.27 examples/s]4806 examples [00:05, 878.02 examples/s]4899 examples [00:05, 890.58 examples/s]4989 examples [00:05, 889.95 examples/s]5079 examples [00:05, 885.68 examples/s]5168 examples [00:06, 880.55 examples/s]5257 examples [00:06, 822.16 examples/s]5345 examples [00:06, 836.82 examples/s]5431 examples [00:06, 841.24 examples/s]5517 examples [00:06, 844.34 examples/s]5604 examples [00:06, 849.55 examples/s]5690 examples [00:06, 838.59 examples/s]5780 examples [00:06, 855.80 examples/s]5871 examples [00:06, 869.99 examples/s]5961 examples [00:06, 878.21 examples/s]6053 examples [00:07, 888.55 examples/s]6144 examples [00:07, 894.73 examples/s]6235 examples [00:07, 898.03 examples/s]6325 examples [00:07, 888.25 examples/s]6414 examples [00:07, 877.48 examples/s]6502 examples [00:07, 865.98 examples/s]6593 examples [00:07, 877.76 examples/s]6681 examples [00:07, 877.13 examples/s]6769 examples [00:07, 870.96 examples/s]6857 examples [00:08, 835.33 examples/s]6948 examples [00:08, 840.14 examples/s]7038 examples [00:08, 854.99 examples/s]7126 examples [00:08, 861.76 examples/s]7213 examples [00:08, 856.47 examples/s]7299 examples [00:08, 815.45 examples/s]7387 examples [00:08, 832.47 examples/s]7475 examples [00:08, 844.87 examples/s]7560 examples [00:08, 845.99 examples/s]7645 examples [00:08, 839.92 examples/s]7736 examples [00:09, 858.90 examples/s]7824 examples [00:09, 862.43 examples/s]7911 examples [00:09, 848.52 examples/s]7997 examples [00:09, 846.30 examples/s]8082 examples [00:09, 847.04 examples/s]8172 examples [00:09, 861.17 examples/s]8259 examples [00:09, 862.49 examples/s]8346 examples [00:09, 859.15 examples/s]8432 examples [00:09, 853.38 examples/s]8518 examples [00:09, 855.15 examples/s]8604 examples [00:10, 851.49 examples/s]8690 examples [00:10, 837.58 examples/s]8777 examples [00:10, 846.93 examples/s]8868 examples [00:10, 863.37 examples/s]8956 examples [00:10, 867.41 examples/s]9043 examples [00:10, 867.81 examples/s]9130 examples [00:10, 858.88 examples/s]9218 examples [00:10, 863.69 examples/s]9309 examples [00:10, 874.61 examples/s]9397 examples [00:10, 870.90 examples/s]9486 examples [00:11, 875.63 examples/s]9574 examples [00:11, 827.83 examples/s]9666 examples [00:11, 851.81 examples/s]9752 examples [00:11, 830.42 examples/s]9836 examples [00:11, 815.62 examples/s]9920 examples [00:11, 820.94 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 97%|█████████▋| 9670/10000 [00:00<00:00, 96699.63 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCMMCG4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCMMCG4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ffa18514950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ffa18514950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ffa18514950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ffa83a39fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff99e898438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ffa18514950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ffa18514950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ffa18514950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ffa16a842b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ffa16a84048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff99c15ee18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff99c15ee18> 

  function with postional parmater data_info <function split_train_valid at 0x7ff99c15ee18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff99c15ef28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff99c15ef28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff99c15ef28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=f55b434aa2be7278d77626870a3f746ff399d6316d2803b607afc1f5bbcbb890
  Stored in directory: /tmp/pip-ephem-wheel-cache-pg_9r5eq/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:21:19, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:28:53, 16.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:11:24, 23.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:08:29, 33.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:59:11, 47.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.42M/862M [00:01<3:28:07, 68.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:24:49, 97.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:40:46, 139kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:01<1:10:09, 198kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<48:52, 283kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<34:04, 403kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<23:47, 574kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:02<16:37, 815kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<12:22, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<10:32, 1.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<10:59, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<08:35, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<09:54, 1.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<08:29, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<06:15, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:10, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<08:05, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<06:25, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<04:38, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<14:23, 919kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<11:24, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<08:15, 1.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:54, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<08:52, 1.48MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<06:52, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:55, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:35, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<05:52, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<04:15, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<12:24, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<10:01, 1.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<08:09, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<07:00, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<05:13, 2.47MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:42, 1.92MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<07:17, 1.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<05:39, 2.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<04:05, 3.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<14:46, 867kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<11:39, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<08:27, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<06:47, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:52, 2.60MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<15:30, 817kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:56, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<08:40, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:58, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:49, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:48, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:47, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:17, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:44, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<04:09, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:32:10, 135kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:05:44, 190kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<46:14, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<35:11, 352kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<27:07, 457kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<19:29, 635kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<13:48, 894kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<14:10, 870kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:11, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:05, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:30, 1.44MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:25, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:31, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:13, 1.96MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<04:58, 2.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<03:42, 3.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:07, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:47, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:22, 2.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:27, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:28, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:11, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:46, 3.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<14:04, 852kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:03, 1.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<07:58, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:22, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:15, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:16, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:32, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:23, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:49, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:44, 2.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:46, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:07, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:34, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:47, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:13, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:28, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:00, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:47, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:07, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:27, 3.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<15:46, 729kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<12:12, 942kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:50, 1.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:33, 1.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:34, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:43, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:24:35, 134kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:00:19, 188kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<42:25, 267kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<32:15, 350kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<24:46, 456kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<17:48, 634kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<12:32, 896kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<17:53, 628kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<13:40, 821kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<09:50, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:29, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:50, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:20, 2.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:30, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:07, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:41, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:18, 2.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:38, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:28, 3.16MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<02:36, 4.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<13:40, 801kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:40, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<07:40, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:57, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:46, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:59, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<04:17, 2.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:30:45, 17.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<7:22:22, 24.4kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<5:09:10, 34.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<3:38:13, 49.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<2:35:15, 69.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<1:49:45, 97.9kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<1:17:48, 138kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<55:21, 194kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<39:33, 271kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<28:24, 377kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<20:34, 520kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<15:00, 712kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<19:46, 540kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<16:52, 633kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<12:33, 850kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:28, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<07:21, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:37, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:27, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:28, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:36, 1.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:40, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:38, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:05, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:56, 2.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:28, 3.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:00, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:08, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:27, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:25, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:32, 2.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:02, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:49, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:08, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:27, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:13, 2.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:19, 3.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:10, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:00, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:45, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:52, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:30, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<02:54, 3.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<16:05, 642kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<14:50, 697kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<11:13, 920kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<08:14, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<06:07, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<04:37, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<23:01, 446kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<19:12, 534kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<14:15, 719kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<10:18, 994kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<07:34, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<05:36, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<35:04, 291kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<27:50, 366kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<20:17, 502kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<14:30, 701kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<10:36, 957kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<07:44, 1.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<11:08, 909kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<11:04, 915kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:32, 1.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:21, 1.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:44, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:42, 2.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<09:48, 1.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<09:53, 1.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:36, 1.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:41, 1.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:29, 2.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:45, 1.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:09, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:09, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:19, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:05, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:19, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<11:17, 879kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:24, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:57, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:19, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:01, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:31, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:23, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:52, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:24, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:21, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:55, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<10:05, 969kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:25, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:12, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:34, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<07:01, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<09:37, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:53, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:50, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<04:17, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:36, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<09:54, 974kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:48, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:44, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:13, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:11, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:53, 1.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:11, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:15, 1.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<03:49, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<11:41, 814kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:27, 829kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:43, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<06:20, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:39, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<08:20, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:54, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:52, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:00, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:41, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<18:12, 515kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<15:24, 608kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<11:27, 817kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<08:09, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:42, 1.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<08:44, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:44, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:52, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:40, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<07:01, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:23, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:53, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:14, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:48, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:28, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<06:27, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:30, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:03, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:38, 2.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<07:36, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<07:19, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<06:05, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:36, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:22, 2.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:15, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:02, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:39, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:22, 2.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<15:27, 575kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<12:28, 712kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:08, 971kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<06:28, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<27:31, 320kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<21:23, 412kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<15:28, 569kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<10:54, 803kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<13:41, 639kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<11:30, 760kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:29, 1.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<06:02, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<11:01, 787kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<09:42, 894kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<07:11, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:08, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:59, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:47, 1.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:10, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:44, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<08:58, 951kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<08:09, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<06:06, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:22, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<07:24, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<07:14, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<05:33, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:59, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:25, 997kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<07:49, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:58, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<04:16, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:14, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:41, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:46, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:09, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<05:55, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:58, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:34, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:19, 2.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:54, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:19, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:11, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:02, 2.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<07:59, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<07:33, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<05:46, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:07, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:55, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:43, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:57, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:33, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:18, 960kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:41, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<06:10, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:39, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:24, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:34, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<23:47, 333kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<18:14, 434kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<13:08, 601kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:25, 753kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:49, 889kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<06:33, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:49, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:12, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:42, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<04:12, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:41, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:49, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:45, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:51, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:14, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<02:24, 3.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<2:28:31, 51.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<1:45:27, 71.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<1:14:01, 102kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<51:43, 145kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<38:52, 193kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<28:42, 261kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<20:26, 366kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<15:24, 482kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<12:16, 605kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<08:55, 831kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:24, 994kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:08, 904kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:19, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:36, 1.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:51, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:24, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:17, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:40, 1.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:15, 1.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<03:07, 2.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:56, 1.82MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:02, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:06, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:14, 3.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:42, 814kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:43, 812kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:39, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:48, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:06, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:56, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:46, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:47, 1.83MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:06, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:39, 1.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:47, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:05, 3.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:56, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:59, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:05, 2.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:32, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:46, 2.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:03, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:39, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:46, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:45, 2.43MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:38, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:48, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:01, 3.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:27, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:48, 971kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:18, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:51, 1.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:19, 1.51MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:11, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:13, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:19, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<04:44, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:49, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:49, 2.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:31, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:37, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:01, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:15, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:33, 2.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:06, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:27, 2.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:44, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:14, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:27, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:34, 2.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:01, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:31, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:23, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:40, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:06, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:25, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:30, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:12, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:18, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:34, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:46, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:07, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:20, 1.75MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:28, 2.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:08, 1.84MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:14, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:31, 2.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:42, 2.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:55, 1.95MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:16, 2.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<01:38, 3.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<15:41, 360kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<11:59, 470kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<08:38, 652kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<06:55, 805kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:50, 953kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<04:19, 1.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:56, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:48, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:52, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:48, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:17, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:18, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:33, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:40, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:52, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:15, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:27, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:42, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:07, 2.48MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:21, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:36, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:01, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:12, 2.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:53, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:25, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:35, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:59, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:11, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:46, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:52, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:35, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:45, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:28, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:32, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:34, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:00, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:58, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:15, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:37, 2.96MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:50, 825kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:54, 816kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:34, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:18, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:27, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:18, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:30, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:47, 2.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<07:23, 633kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<06:03, 772kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:24, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<03:09, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:36, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:20, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:32, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:31, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:28, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:51, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:05, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:34, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:08, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:08, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:32, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:53, 2.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:08, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:47, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:17, 3.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:39, 1.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:10, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:16, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:21, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:37, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:35, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:00, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:04, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:11, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:41, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:13, 3.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:39, 717kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:28, 906kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:15, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:01, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:37, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:57, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:08, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:09, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:38, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:11, 3.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:35, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:22, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:17, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:00, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:06, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:33, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:57, 1.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:25, 2.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:15, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:41, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:47, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:31, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:04, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:30, 2.38MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:02, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:02, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:34, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:41, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:46, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:23, 2.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:33, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:39, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:18, 2.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:28, 2.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:30, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:12, 2.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<00:52, 3.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:19, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:10, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:38, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<01:09, 2.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<08:56, 362kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<07:22, 438kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<05:23, 599kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<03:48, 841kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:24, 930kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:54, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:08, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:30, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<08:42, 355kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<07:10, 432kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:16, 585kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<03:42, 824kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:23, 891kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:52, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:08, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:58, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:53, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:26, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:29, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:31, 1.89MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:11, 2.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:18, 2.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:53, 1.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:33, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:07, 2.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:32, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:32, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:20, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:01, 2.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:47, 3.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:24, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:05, 2.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:12, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:41, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:24, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:01, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:24, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:24, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:05, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:10, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:14, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:57, 2.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:04, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:10, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:54, 2.61MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:01, 2.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:06, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:52, 2.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:59, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:04, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:50, 2.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:57, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:02, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:48, 2.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:55, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:00, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:47, 2.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:53, 2.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:59, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:46, 2.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:52, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:44, 2.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:50, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:42, 2.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:48, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:12, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:00, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:43, 2.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:59, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:00, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:45, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:32, 3.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:59, 860kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:57, 879kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:29, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:03, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:10, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:05, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:49, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:34, 2.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:39, 953kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:25, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:03, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:58, 1.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:56, 1.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:42, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:44, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:58, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:48, 1.77MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:35, 2.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:46, 1.77MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:46, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:35, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:37, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:39, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:30, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:33, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:35, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:27, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:30, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:33, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:25, 2.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:18, 3.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<02:07, 515kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:52, 586kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:23, 782kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:58, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:57, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:44, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:31, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:37, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:36, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:27, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:28, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:21, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:33, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:27, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:19, 2.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:25, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:25, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:20, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:15, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:15, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:16, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:16, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:12, 2.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:10, 2.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:16, 1.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:08, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:02, 3.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:11, 722kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:10, 757kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 1.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.84MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 42/400000 [00:00<15:53, 419.56it/s]  0%|          | 712/400000 [00:00<11:24, 583.69it/s]  0%|          | 1420/400000 [00:00<08:14, 805.37it/s]  1%|          | 2128/400000 [00:00<06:02, 1097.03it/s]  1%|          | 2835/400000 [00:00<04:30, 1469.33it/s]  1%|          | 3546/400000 [00:00<03:25, 1928.16it/s]  1%|          | 4256/400000 [00:00<02:40, 2467.31it/s]  1%|          | 4961/400000 [00:00<02:08, 3064.54it/s]  1%|▏         | 5683/400000 [00:00<01:46, 3703.87it/s]  2%|▏         | 6406/400000 [00:01<01:30, 4338.44it/s]  2%|▏         | 7135/400000 [00:01<01:19, 4937.55it/s]  2%|▏         | 7845/400000 [00:01<01:12, 5433.41it/s]  2%|▏         | 8554/400000 [00:01<01:07, 5841.48it/s]  2%|▏         | 9259/400000 [00:01<01:03, 6156.77it/s]  2%|▏         | 9964/400000 [00:01<01:02, 6225.31it/s]  3%|▎         | 10659/400000 [00:01<01:00, 6425.67it/s]  3%|▎         | 11392/400000 [00:01<00:58, 6672.28it/s]  3%|▎         | 12111/400000 [00:01<00:56, 6818.12it/s]  3%|▎         | 12828/400000 [00:01<00:55, 6919.01it/s]  3%|▎         | 13538/400000 [00:02<00:55, 6903.58it/s]  4%|▎         | 14250/400000 [00:02<00:55, 6965.33it/s]  4%|▎         | 14960/400000 [00:02<00:54, 7002.87it/s]  4%|▍         | 15682/400000 [00:02<00:54, 7065.21it/s]  4%|▍         | 16400/400000 [00:02<00:54, 7099.20it/s]  4%|▍         | 17114/400000 [00:02<00:54, 7083.90it/s]  4%|▍         | 17825/400000 [00:02<00:53, 7080.47it/s]  5%|▍         | 18535/400000 [00:02<00:54, 7053.45it/s]  5%|▍         | 19257/400000 [00:02<00:53, 7100.11it/s]  5%|▍         | 19972/400000 [00:02<00:53, 7113.53it/s]  5%|▌         | 20684/400000 [00:03<00:53, 7087.39it/s]  5%|▌         | 21398/400000 [00:03<00:53, 7100.60it/s]  6%|▌         | 22119/400000 [00:03<00:52, 7130.74it/s]  6%|▌         | 22833/400000 [00:03<00:53, 7093.11it/s]  6%|▌         | 23546/400000 [00:03<00:53, 7101.68it/s]  6%|▌         | 24257/400000 [00:03<00:53, 6971.25it/s]  6%|▌         | 24976/400000 [00:03<00:53, 7034.97it/s]  6%|▋         | 25681/400000 [00:03<00:53, 7035.04it/s]  7%|▋         | 26393/400000 [00:03<00:52, 7059.43it/s]  7%|▋         | 27100/400000 [00:03<00:53, 7007.03it/s]  7%|▋         | 27802/400000 [00:04<00:55, 6666.42it/s]  7%|▋         | 28520/400000 [00:04<00:54, 6812.52it/s]  7%|▋         | 29205/400000 [00:04<00:55, 6738.86it/s]  7%|▋         | 29898/400000 [00:04<00:54, 6793.99it/s]  8%|▊         | 30620/400000 [00:04<00:53, 6899.55it/s]  8%|▊         | 31312/400000 [00:04<00:55, 6682.68it/s]  8%|▊         | 31998/400000 [00:04<00:54, 6732.92it/s]  8%|▊         | 32696/400000 [00:04<00:53, 6802.48it/s]  8%|▊         | 33388/400000 [00:04<00:53, 6835.45it/s]  9%|▊         | 34083/400000 [00:04<00:53, 6867.84it/s]  9%|▊         | 34771/400000 [00:05<00:55, 6594.07it/s]  9%|▉         | 35434/400000 [00:05<00:55, 6549.55it/s]  9%|▉         | 36139/400000 [00:05<00:54, 6690.82it/s]  9%|▉         | 36842/400000 [00:05<00:53, 6788.76it/s]  9%|▉         | 37547/400000 [00:05<00:52, 6863.04it/s] 10%|▉         | 38235/400000 [00:05<00:53, 6719.78it/s] 10%|▉         | 38957/400000 [00:05<00:52, 6861.42it/s] 10%|▉         | 39646/400000 [00:05<00:52, 6846.61it/s] 10%|█         | 40333/400000 [00:05<00:53, 6728.20it/s] 10%|█         | 41013/400000 [00:05<00:53, 6748.51it/s] 10%|█         | 41689/400000 [00:06<00:56, 6387.55it/s] 11%|█         | 42397/400000 [00:06<00:54, 6578.49it/s] 11%|█         | 43100/400000 [00:06<00:53, 6705.37it/s] 11%|█         | 43806/400000 [00:06<00:52, 6805.55it/s] 11%|█         | 44504/400000 [00:06<00:51, 6854.72it/s] 11%|█▏        | 45199/400000 [00:06<00:51, 6880.96it/s] 11%|█▏        | 45889/400000 [00:06<00:51, 6881.28it/s] 12%|█▏        | 46606/400000 [00:06<00:50, 6963.99it/s] 12%|█▏        | 47304/400000 [00:06<00:51, 6874.58it/s] 12%|█▏        | 48011/400000 [00:07<00:50, 6931.70it/s] 12%|█▏        | 48725/400000 [00:07<00:50, 6991.99it/s] 12%|█▏        | 49436/400000 [00:07<00:49, 7025.59it/s] 13%|█▎        | 50160/400000 [00:07<00:49, 7086.61it/s] 13%|█▎        | 50870/400000 [00:07<00:49, 7082.64it/s] 13%|█▎        | 51579/400000 [00:07<00:50, 6949.78it/s] 13%|█▎        | 52283/400000 [00:07<00:49, 6976.55it/s] 13%|█▎        | 53000/400000 [00:07<00:49, 7032.35it/s] 13%|█▎        | 53716/400000 [00:07<00:48, 7067.35it/s] 14%|█▎        | 54431/400000 [00:07<00:48, 7089.56it/s] 14%|█▍        | 55160/400000 [00:08<00:48, 7146.25it/s] 14%|█▍        | 55875/400000 [00:08<00:48, 7093.30it/s] 14%|█▍        | 56590/400000 [00:08<00:48, 7108.48it/s] 14%|█▍        | 57304/400000 [00:08<00:48, 7116.12it/s] 15%|█▍        | 58026/400000 [00:08<00:47, 7145.30it/s] 15%|█▍        | 58741/400000 [00:08<00:48, 6997.94it/s] 15%|█▍        | 59462/400000 [00:08<00:48, 7059.56it/s] 15%|█▌        | 60169/400000 [00:08<00:48, 6969.18it/s] 15%|█▌        | 60893/400000 [00:08<00:48, 7047.04it/s] 15%|█▌        | 61599/400000 [00:08<00:48, 6989.59it/s] 16%|█▌        | 62326/400000 [00:09<00:47, 7070.10it/s] 16%|█▌        | 63034/400000 [00:09<00:49, 6827.66it/s] 16%|█▌        | 63726/400000 [00:09<00:49, 6852.75it/s] 16%|█▌        | 64456/400000 [00:09<00:48, 6978.61it/s] 16%|█▋        | 65168/400000 [00:09<00:47, 7020.27it/s] 16%|█▋        | 65872/400000 [00:09<00:47, 6995.43it/s] 17%|█▋        | 66575/400000 [00:09<00:47, 7003.10it/s] 17%|█▋        | 67285/400000 [00:09<00:47, 7028.68it/s] 17%|█▋        | 68005/400000 [00:09<00:46, 7078.28it/s] 17%|█▋        | 68725/400000 [00:09<00:46, 7113.36it/s] 17%|█▋        | 69437/400000 [00:10<00:48, 6865.74it/s] 18%|█▊        | 70132/400000 [00:10<00:47, 6889.91it/s] 18%|█▊        | 70832/400000 [00:10<00:47, 6922.04it/s] 18%|█▊        | 71550/400000 [00:10<00:46, 6995.05it/s] 18%|█▊        | 72279/400000 [00:10<00:46, 7080.75it/s] 18%|█▊        | 72991/400000 [00:10<00:46, 7090.59it/s] 18%|█▊        | 73701/400000 [00:10<00:46, 6958.69it/s] 19%|█▊        | 74398/400000 [00:10<00:46, 6956.35it/s] 19%|█▉        | 75123/400000 [00:10<00:46, 7040.03it/s] 19%|█▉        | 75850/400000 [00:10<00:45, 7106.57it/s] 19%|█▉        | 76562/400000 [00:11<00:45, 7090.48it/s] 19%|█▉        | 77272/400000 [00:11<00:45, 7016.02it/s] 19%|█▉        | 77975/400000 [00:11<00:46, 6998.52it/s] 20%|█▉        | 78676/400000 [00:11<00:46, 6938.94it/s] 20%|█▉        | 79388/400000 [00:11<00:45, 6990.07it/s] 20%|██        | 80088/400000 [00:11<00:46, 6907.23it/s] 20%|██        | 80801/400000 [00:11<00:45, 6970.08it/s] 20%|██        | 81499/400000 [00:11<00:45, 6966.36it/s] 21%|██        | 82205/400000 [00:11<00:45, 6993.57it/s] 21%|██        | 82913/400000 [00:11<00:45, 7017.84it/s] 21%|██        | 83620/400000 [00:12<00:45, 7030.42it/s] 21%|██        | 84326/400000 [00:12<00:44, 7037.50it/s] 21%|██▏       | 85048/400000 [00:12<00:44, 7089.19it/s] 21%|██▏       | 85791/400000 [00:12<00:43, 7186.68it/s] 22%|██▏       | 86523/400000 [00:12<00:43, 7224.51it/s] 22%|██▏       | 87246/400000 [00:12<00:43, 7204.10it/s] 22%|██▏       | 87967/400000 [00:12<00:43, 7174.37it/s] 22%|██▏       | 88685/400000 [00:12<00:43, 7120.83it/s] 22%|██▏       | 89406/400000 [00:12<00:43, 7145.19it/s] 23%|██▎       | 90129/400000 [00:12<00:43, 7167.90it/s] 23%|██▎       | 90850/400000 [00:13<00:43, 7178.92it/s] 23%|██▎       | 91573/400000 [00:13<00:42, 7192.85it/s] 23%|██▎       | 92293/400000 [00:13<00:44, 6941.57it/s] 23%|██▎       | 93017/400000 [00:13<00:43, 7027.58it/s] 23%|██▎       | 93722/400000 [00:13<00:45, 6713.66it/s] 24%|██▎       | 94415/400000 [00:13<00:45, 6777.12it/s] 24%|██▍       | 95124/400000 [00:13<00:44, 6867.52it/s] 24%|██▍       | 95842/400000 [00:13<00:43, 6952.42it/s] 24%|██▍       | 96540/400000 [00:13<00:44, 6874.09it/s] 24%|██▍       | 97242/400000 [00:14<00:43, 6916.76it/s] 24%|██▍       | 97954/400000 [00:14<00:43, 6973.45it/s] 25%|██▍       | 98653/400000 [00:14<00:43, 6886.73it/s] 25%|██▍       | 99370/400000 [00:14<00:43, 6967.13it/s] 25%|██▌       | 100095/400000 [00:14<00:42, 7046.15it/s] 25%|██▌       | 100823/400000 [00:14<00:42, 7113.60it/s] 25%|██▌       | 101536/400000 [00:14<00:42, 7099.27it/s] 26%|██▌       | 102251/400000 [00:14<00:41, 7112.46it/s] 26%|██▌       | 102965/400000 [00:14<00:41, 7117.25it/s] 26%|██▌       | 103684/400000 [00:14<00:41, 7138.41it/s] 26%|██▌       | 104399/400000 [00:15<00:41, 7134.58it/s] 26%|██▋       | 105113/400000 [00:15<00:41, 7086.63it/s] 26%|██▋       | 105839/400000 [00:15<00:41, 7137.43it/s] 27%|██▋       | 106563/400000 [00:15<00:40, 7166.82it/s] 27%|██▋       | 107306/400000 [00:15<00:40, 7242.69it/s] 27%|██▋       | 108046/400000 [00:15<00:40, 7289.06it/s] 27%|██▋       | 108776/400000 [00:15<00:39, 7281.08it/s] 27%|██▋       | 109505/400000 [00:15<00:40, 7256.61it/s] 28%|██▊       | 110231/400000 [00:15<00:40, 7238.10it/s] 28%|██▊       | 110967/400000 [00:15<00:39, 7272.34it/s] 28%|██▊       | 111700/400000 [00:16<00:39, 7287.60it/s] 28%|██▊       | 112435/400000 [00:16<00:39, 7304.29it/s] 28%|██▊       | 113166/400000 [00:16<00:39, 7305.31it/s] 28%|██▊       | 113917/400000 [00:16<00:38, 7362.81it/s] 29%|██▊       | 114654/400000 [00:16<00:38, 7343.41it/s] 29%|██▉       | 115395/400000 [00:16<00:38, 7362.86it/s] 29%|██▉       | 116132/400000 [00:16<00:38, 7358.66it/s] 29%|██▉       | 116868/400000 [00:16<00:39, 7176.81it/s] 29%|██▉       | 117596/400000 [00:16<00:39, 7207.34it/s] 30%|██▉       | 118327/400000 [00:16<00:38, 7237.62it/s] 30%|██▉       | 119052/400000 [00:17<00:38, 7241.25it/s] 30%|██▉       | 119777/400000 [00:17<00:38, 7234.25it/s] 30%|███       | 120501/400000 [00:17<00:39, 7056.09it/s] 30%|███       | 121208/400000 [00:17<00:39, 7027.46it/s] 30%|███       | 121918/400000 [00:17<00:39, 7048.43it/s] 31%|███       | 122641/400000 [00:17<00:39, 7099.89it/s] 31%|███       | 123366/400000 [00:17<00:38, 7141.29it/s] 31%|███       | 124094/400000 [00:17<00:38, 7181.83it/s] 31%|███       | 124813/400000 [00:17<00:38, 7147.83it/s] 31%|███▏      | 125529/400000 [00:17<00:38, 7114.32it/s] 32%|███▏      | 126259/400000 [00:18<00:38, 7166.90it/s] 32%|███▏      | 126978/400000 [00:18<00:38, 7172.59it/s] 32%|███▏      | 127709/400000 [00:18<00:37, 7213.14it/s] 32%|███▏      | 128442/400000 [00:18<00:37, 7246.01it/s] 32%|███▏      | 129168/400000 [00:18<00:37, 7249.42it/s] 32%|███▏      | 129906/400000 [00:18<00:37, 7285.44it/s] 33%|███▎      | 130635/400000 [00:18<00:37, 7272.67it/s] 33%|███▎      | 131366/400000 [00:18<00:36, 7282.57it/s] 33%|███▎      | 132095/400000 [00:18<00:36, 7279.79it/s] 33%|███▎      | 132824/400000 [00:18<00:36, 7269.10it/s] 33%|███▎      | 133557/400000 [00:19<00:36, 7286.35it/s] 34%|███▎      | 134293/400000 [00:19<00:36, 7305.80it/s] 34%|███▍      | 135024/400000 [00:19<00:36, 7282.89it/s] 34%|███▍      | 135753/400000 [00:19<00:36, 7276.12it/s] 34%|███▍      | 136484/400000 [00:19<00:36, 7283.73it/s] 34%|███▍      | 137213/400000 [00:19<00:36, 7158.77it/s] 34%|███▍      | 137943/400000 [00:19<00:36, 7199.99it/s] 35%|███▍      | 138664/400000 [00:19<00:36, 7195.37it/s] 35%|███▍      | 139384/400000 [00:19<00:36, 7097.70it/s] 35%|███▌      | 140097/400000 [00:19<00:36, 7105.17it/s] 35%|███▌      | 140823/400000 [00:20<00:36, 7148.30it/s] 35%|███▌      | 141545/400000 [00:20<00:36, 7169.12it/s] 36%|███▌      | 142263/400000 [00:20<00:36, 7118.13it/s] 36%|███▌      | 142980/400000 [00:20<00:36, 7132.92it/s] 36%|███▌      | 143695/400000 [00:20<00:35, 7137.25it/s] 36%|███▌      | 144409/400000 [00:20<00:37, 6806.26it/s] 36%|███▋      | 145128/400000 [00:20<00:36, 6915.17it/s] 36%|███▋      | 145823/400000 [00:20<00:37, 6852.79it/s] 37%|███▋      | 146556/400000 [00:20<00:36, 6987.14it/s] 37%|███▋      | 147283/400000 [00:21<00:35, 7068.35it/s] 37%|███▋      | 147995/400000 [00:21<00:35, 7082.49it/s] 37%|███▋      | 148705/400000 [00:21<00:35, 7040.94it/s] 37%|███▋      | 149410/400000 [00:21<00:37, 6732.71it/s] 38%|███▊      | 150128/400000 [00:21<00:36, 6860.23it/s] 38%|███▊      | 150847/400000 [00:21<00:35, 6955.57it/s] 38%|███▊      | 151570/400000 [00:21<00:35, 7035.03it/s] 38%|███▊      | 152298/400000 [00:21<00:34, 7105.08it/s] 38%|███▊      | 153029/400000 [00:21<00:34, 7163.35it/s] 38%|███▊      | 153756/400000 [00:21<00:34, 7194.55it/s] 39%|███▊      | 154490/400000 [00:22<00:33, 7237.14it/s] 39%|███▉      | 155223/400000 [00:22<00:33, 7261.53it/s] 39%|███▉      | 155952/400000 [00:22<00:33, 7268.05it/s] 39%|███▉      | 156680/400000 [00:22<00:33, 7219.41it/s] 39%|███▉      | 157413/400000 [00:22<00:33, 7250.16it/s] 40%|███▉      | 158148/400000 [00:22<00:33, 7278.53it/s] 40%|███▉      | 158877/400000 [00:22<00:33, 7275.11it/s] 40%|███▉      | 159605/400000 [00:22<00:33, 7274.52it/s] 40%|████      | 160333/400000 [00:22<00:32, 7273.70it/s] 40%|████      | 161061/400000 [00:22<00:33, 7062.56it/s] 40%|████      | 161801/400000 [00:23<00:33, 7158.44it/s] 41%|████      | 162532/400000 [00:23<00:32, 7201.43it/s] 41%|████      | 163256/400000 [00:23<00:32, 7212.40it/s] 41%|████      | 163978/400000 [00:23<00:32, 7187.93it/s] 41%|████      | 164698/400000 [00:23<00:33, 6952.78it/s] 41%|████▏     | 165396/400000 [00:23<00:34, 6896.74it/s] 42%|████▏     | 166128/400000 [00:23<00:33, 7016.42it/s] 42%|████▏     | 166863/400000 [00:23<00:32, 7112.32it/s] 42%|████▏     | 167588/400000 [00:23<00:32, 7150.10it/s] 42%|████▏     | 168305/400000 [00:23<00:32, 7083.99it/s] 42%|████▏     | 169026/400000 [00:24<00:32, 7118.87it/s] 42%|████▏     | 169749/400000 [00:24<00:32, 7151.31it/s] 43%|████▎     | 170474/400000 [00:24<00:31, 7178.02it/s] 43%|████▎     | 171195/400000 [00:24<00:31, 7185.45it/s] 43%|████▎     | 171914/400000 [00:24<00:32, 7125.97it/s] 43%|████▎     | 172636/400000 [00:24<00:31, 7153.13it/s] 43%|████▎     | 173354/400000 [00:24<00:31, 7160.74it/s] 44%|████▎     | 174071/400000 [00:24<00:31, 7147.15it/s] 44%|████▎     | 174786/400000 [00:24<00:31, 7048.26it/s] 44%|████▍     | 175501/400000 [00:24<00:31, 7077.53it/s] 44%|████▍     | 176220/400000 [00:25<00:31, 7109.01it/s] 44%|████▍     | 176938/400000 [00:25<00:31, 7129.34it/s] 44%|████▍     | 177652/400000 [00:25<00:31, 7120.99it/s] 45%|████▍     | 178368/400000 [00:25<00:31, 7130.39it/s] 45%|████▍     | 179082/400000 [00:25<00:31, 7086.29it/s] 45%|████▍     | 179809/400000 [00:25<00:30, 7139.76it/s] 45%|████▌     | 180529/400000 [00:25<00:30, 7154.93it/s] 45%|████▌     | 181250/400000 [00:25<00:30, 7171.29it/s] 45%|████▌     | 181968/400000 [00:25<00:30, 7148.71it/s] 46%|████▌     | 182683/400000 [00:25<00:30, 7099.92it/s] 46%|████▌     | 183399/400000 [00:26<00:30, 7117.29it/s] 46%|████▌     | 184111/400000 [00:26<00:30, 7117.15it/s] 46%|████▌     | 184835/400000 [00:26<00:30, 7151.32it/s] 46%|████▋     | 185551/400000 [00:26<00:30, 6929.78it/s] 47%|████▋     | 186247/400000 [00:26<00:30, 6937.20it/s] 47%|████▋     | 186969/400000 [00:26<00:30, 7018.16it/s] 47%|████▋     | 187684/400000 [00:26<00:30, 7055.12it/s] 47%|████▋     | 188401/400000 [00:26<00:29, 7088.16it/s] 47%|████▋     | 189127/400000 [00:26<00:29, 7136.32it/s] 47%|████▋     | 189857/400000 [00:26<00:29, 7182.47it/s] 48%|████▊     | 190576/400000 [00:27<00:29, 7135.19it/s] 48%|████▊     | 191310/400000 [00:27<00:29, 7194.02it/s] 48%|████▊     | 192032/400000 [00:27<00:28, 7201.14it/s] 48%|████▊     | 192760/400000 [00:27<00:28, 7223.33it/s] 48%|████▊     | 193483/400000 [00:27<00:28, 7172.06it/s] 49%|████▊     | 194210/400000 [00:27<00:28, 7199.98it/s] 49%|████▊     | 194942/400000 [00:27<00:28, 7232.27it/s] 49%|████▉     | 195666/400000 [00:27<00:28, 7228.72it/s] 49%|████▉     | 196391/400000 [00:27<00:28, 7233.87it/s] 49%|████▉     | 197115/400000 [00:27<00:28, 7181.59it/s] 49%|████▉     | 197834/400000 [00:28<00:28, 7177.22it/s] 50%|████▉     | 198563/400000 [00:28<00:27, 7210.17it/s] 50%|████▉     | 199285/400000 [00:28<00:28, 7148.21it/s] 50%|█████     | 200031/400000 [00:28<00:27, 7237.76it/s] 50%|█████     | 200780/400000 [00:28<00:27, 7310.21it/s] 50%|█████     | 201519/400000 [00:28<00:27, 7332.96it/s] 51%|█████     | 202253/400000 [00:28<00:27, 7297.06it/s] 51%|█████     | 202988/400000 [00:28<00:26, 7312.35it/s] 51%|█████     | 203720/400000 [00:28<00:26, 7280.69it/s] 51%|█████     | 204451/400000 [00:28<00:26, 7288.93it/s] 51%|█████▏    | 205188/400000 [00:29<00:26, 7312.67it/s] 51%|█████▏    | 205923/400000 [00:29<00:26, 7321.11it/s] 52%|█████▏    | 206656/400000 [00:29<00:26, 7314.83it/s] 52%|█████▏    | 207393/400000 [00:29<00:26, 7328.91it/s] 52%|█████▏    | 208126/400000 [00:29<00:26, 7298.81it/s] 52%|█████▏    | 208856/400000 [00:29<00:26, 7276.19it/s] 52%|█████▏    | 209584/400000 [00:29<00:26, 7272.80it/s] 53%|█████▎    | 210312/400000 [00:29<00:26, 7270.15it/s] 53%|█████▎    | 211040/400000 [00:29<00:26, 7223.89it/s] 53%|█████▎    | 211763/400000 [00:30<00:26, 7189.73it/s] 53%|█████▎    | 212494/400000 [00:30<00:25, 7224.70it/s] 53%|█████▎    | 213221/400000 [00:30<00:25, 7235.34it/s] 53%|█████▎    | 213967/400000 [00:30<00:25, 7298.69it/s] 54%|█████▎    | 214698/400000 [00:30<00:25, 7256.13it/s] 54%|█████▍    | 215424/400000 [00:30<00:25, 7242.84it/s] 54%|█████▍    | 216149/400000 [00:30<00:25, 7192.55it/s] 54%|█████▍    | 216869/400000 [00:30<00:25, 7189.42it/s] 54%|█████▍    | 217601/400000 [00:30<00:25, 7227.18it/s] 55%|█████▍    | 218324/400000 [00:30<00:25, 7169.15it/s] 55%|█████▍    | 219044/400000 [00:31<00:25, 7176.14it/s] 55%|█████▍    | 219762/400000 [00:31<00:25, 7107.67it/s] 55%|█████▌    | 220474/400000 [00:31<00:25, 7098.23it/s] 55%|█████▌    | 221184/400000 [00:31<00:25, 6887.97it/s] 55%|█████▌    | 221898/400000 [00:31<00:25, 6960.39it/s] 56%|█████▌    | 222628/400000 [00:31<00:25, 7057.11it/s] 56%|█████▌    | 223335/400000 [00:31<00:25, 7042.61it/s] 56%|█████▌    | 224060/400000 [00:31<00:24, 7102.30it/s] 56%|█████▌    | 224778/400000 [00:31<00:24, 7124.10it/s] 56%|█████▋    | 225491/400000 [00:31<00:24, 7093.71it/s] 57%|█████▋    | 226203/400000 [00:32<00:24, 7101.15it/s] 57%|█████▋    | 226914/400000 [00:32<00:24, 7065.63it/s] 57%|█████▋    | 227628/400000 [00:32<00:24, 7085.06it/s] 57%|█████▋    | 228346/400000 [00:32<00:24, 7110.90it/s] 57%|█████▋    | 229063/400000 [00:32<00:23, 7127.53it/s] 57%|█████▋    | 229787/400000 [00:32<00:23, 7158.33it/s] 58%|█████▊    | 230503/400000 [00:32<00:24, 7045.69it/s] 58%|█████▊    | 231209/400000 [00:32<00:23, 7039.71it/s] 58%|█████▊    | 231929/400000 [00:32<00:23, 7086.91it/s] 58%|█████▊    | 232638/400000 [00:32<00:24, 6950.98it/s] 58%|█████▊    | 233334/400000 [00:33<00:24, 6934.99it/s] 59%|█████▊    | 234065/400000 [00:33<00:23, 7043.03it/s] 59%|█████▊    | 234804/400000 [00:33<00:23, 7142.33it/s] 59%|█████▉    | 235529/400000 [00:33<00:22, 7173.69it/s] 59%|█████▉    | 236267/400000 [00:33<00:22, 7233.18it/s] 59%|█████▉    | 236991/400000 [00:33<00:22, 7231.25it/s] 59%|█████▉    | 237715/400000 [00:33<00:22, 7189.46it/s] 60%|█████▉    | 238441/400000 [00:33<00:22, 7207.91it/s] 60%|█████▉    | 239163/400000 [00:33<00:22, 7206.85it/s] 60%|█████▉    | 239884/400000 [00:33<00:22, 7204.90it/s] 60%|██████    | 240605/400000 [00:34<00:22, 7192.40it/s] 60%|██████    | 241331/400000 [00:34<00:22, 7211.87it/s] 61%|██████    | 242053/400000 [00:34<00:22, 7012.53it/s] 61%|██████    | 242773/400000 [00:34<00:22, 7066.65it/s] 61%|██████    | 243491/400000 [00:34<00:22, 7099.98it/s] 61%|██████    | 244202/400000 [00:34<00:21, 7082.27it/s] 61%|██████    | 244912/400000 [00:34<00:21, 7085.19it/s] 61%|██████▏   | 245621/400000 [00:34<00:21, 7062.56it/s] 62%|██████▏   | 246330/400000 [00:34<00:21, 7068.88it/s] 62%|██████▏   | 247061/400000 [00:34<00:21, 7138.07it/s] 62%|██████▏   | 247782/400000 [00:35<00:21, 7159.06it/s] 62%|██████▏   | 248499/400000 [00:35<00:21, 7142.29it/s] 62%|██████▏   | 249237/400000 [00:35<00:20, 7210.29it/s] 62%|██████▏   | 249962/400000 [00:35<00:20, 7220.61it/s] 63%|██████▎   | 250685/400000 [00:35<00:20, 7218.62it/s] 63%|██████▎   | 251407/400000 [00:35<00:20, 7216.24it/s] 63%|██████▎   | 252129/400000 [00:35<00:20, 7099.61it/s] 63%|██████▎   | 252840/400000 [00:35<00:20, 7097.54it/s] 63%|██████▎   | 253562/400000 [00:35<00:20, 7132.05it/s] 64%|██████▎   | 254276/400000 [00:35<00:20, 7089.25it/s] 64%|██████▎   | 254986/400000 [00:36<00:20, 7063.60it/s] 64%|██████▍   | 255693/400000 [00:36<00:20, 7005.77it/s] 64%|██████▍   | 256408/400000 [00:36<00:20, 7047.25it/s] 64%|██████▍   | 257128/400000 [00:36<00:20, 7088.01it/s] 64%|██████▍   | 257846/400000 [00:36<00:19, 7112.71it/s] 65%|██████▍   | 258573/400000 [00:36<00:19, 7157.52it/s] 65%|██████▍   | 259289/400000 [00:36<00:19, 7116.95it/s] 65%|██████▌   | 260001/400000 [00:36<00:19, 7065.61it/s] 65%|██████▌   | 260723/400000 [00:36<00:19, 7110.91it/s] 65%|██████▌   | 261438/400000 [00:36<00:19, 7121.10it/s] 66%|██████▌   | 262151/400000 [00:37<00:19, 6924.83it/s] 66%|██████▌   | 262845/400000 [00:37<00:20, 6819.67it/s] 66%|██████▌   | 263529/400000 [00:37<00:20, 6729.74it/s] 66%|██████▌   | 264237/400000 [00:37<00:19, 6829.47it/s] 66%|██████▌   | 264937/400000 [00:37<00:19, 6877.19it/s] 66%|██████▋   | 265626/400000 [00:37<00:19, 6738.15it/s] 67%|██████▋   | 266302/400000 [00:37<00:20, 6436.39it/s] 67%|██████▋   | 267017/400000 [00:37<00:20, 6634.26it/s] 67%|██████▋   | 267714/400000 [00:37<00:19, 6729.37it/s] 67%|██████▋   | 268410/400000 [00:38<00:19, 6794.19it/s] 67%|██████▋   | 269092/400000 [00:38<00:19, 6785.54it/s] 67%|██████▋   | 269774/400000 [00:38<00:19, 6795.06it/s] 68%|██████▊   | 270463/400000 [00:38<00:18, 6819.62it/s] 68%|██████▊   | 271146/400000 [00:38<00:19, 6736.55it/s] 68%|██████▊   | 271821/400000 [00:38<00:20, 6391.38it/s] 68%|██████▊   | 272465/400000 [00:38<00:19, 6389.99it/s] 68%|██████▊   | 273134/400000 [00:38<00:19, 6474.98it/s] 68%|██████▊   | 273820/400000 [00:38<00:19, 6584.54it/s] 69%|██████▊   | 274528/400000 [00:38<00:18, 6725.08it/s] 69%|██████▉   | 275228/400000 [00:39<00:18, 6805.02it/s] 69%|██████▉   | 275911/400000 [00:39<00:18, 6762.55it/s] 69%|██████▉   | 276623/400000 [00:39<00:17, 6864.52it/s] 69%|██████▉   | 277349/400000 [00:39<00:17, 6978.33it/s] 70%|██████▉   | 278049/400000 [00:39<00:17, 6861.70it/s] 70%|██████▉   | 278745/400000 [00:39<00:17, 6890.76it/s] 70%|██████▉   | 279436/400000 [00:39<00:18, 6576.71it/s] 70%|███████   | 280159/400000 [00:39<00:17, 6758.36it/s] 70%|███████   | 280857/400000 [00:39<00:17, 6822.35it/s] 70%|███████   | 281549/400000 [00:39<00:17, 6850.85it/s] 71%|███████   | 282255/400000 [00:40<00:17, 6909.65it/s] 71%|███████   | 282966/400000 [00:40<00:16, 6968.02it/s] 71%|███████   | 283681/400000 [00:40<00:16, 7019.31it/s] 71%|███████   | 284384/400000 [00:40<00:16, 7019.86it/s] 71%|███████▏  | 285087/400000 [00:40<00:16, 6993.47it/s] 71%|███████▏  | 285787/400000 [00:40<00:16, 6842.79it/s] 72%|███████▏  | 286490/400000 [00:40<00:16, 6896.53it/s] 72%|███████▏  | 287191/400000 [00:40<00:16, 6930.15it/s] 72%|███████▏  | 287885/400000 [00:40<00:16, 6900.91it/s] 72%|███████▏  | 288602/400000 [00:40<00:15, 6978.68it/s] 72%|███████▏  | 289329/400000 [00:41<00:15, 7061.65it/s] 73%|███████▎  | 290038/400000 [00:41<00:15, 7069.68it/s] 73%|███████▎  | 290770/400000 [00:41<00:15, 7141.67it/s] 73%|███████▎  | 291514/400000 [00:41<00:15, 7226.90it/s] 73%|███████▎  | 292249/400000 [00:41<00:14, 7260.58it/s] 73%|███████▎  | 292976/400000 [00:41<00:14, 7225.40it/s] 73%|███████▎  | 293699/400000 [00:41<00:14, 7156.78it/s] 74%|███████▎  | 294416/400000 [00:41<00:14, 7157.18it/s] 74%|███████▍  | 295156/400000 [00:41<00:14, 7228.03it/s] 74%|███████▍  | 295880/400000 [00:41<00:14, 7217.36it/s] 74%|███████▍  | 296615/400000 [00:42<00:14, 7256.19it/s] 74%|███████▍  | 297341/400000 [00:42<00:14, 7212.59it/s] 75%|███████▍  | 298064/400000 [00:42<00:14, 7217.45it/s] 75%|███████▍  | 298800/400000 [00:42<00:13, 7257.60it/s] 75%|███████▍  | 299526/400000 [00:42<00:13, 7228.95it/s] 75%|███████▌  | 300258/400000 [00:42<00:13, 7254.94it/s] 75%|███████▌  | 300984/400000 [00:42<00:13, 7223.60it/s] 75%|███████▌  | 301707/400000 [00:42<00:13, 7099.13it/s] 76%|███████▌  | 302419/400000 [00:42<00:13, 7104.15it/s] 76%|███████▌  | 303133/400000 [00:43<00:13, 7112.26it/s] 76%|███████▌  | 303845/400000 [00:43<00:13, 7110.19it/s] 76%|███████▌  | 304578/400000 [00:43<00:13, 7174.13it/s] 76%|███████▋  | 305314/400000 [00:43<00:13, 7227.50it/s] 77%|███████▋  | 306038/400000 [00:43<00:13, 7139.03it/s] 77%|███████▋  | 306753/400000 [00:43<00:13, 7139.79it/s] 77%|███████▋  | 307473/400000 [00:43<00:12, 7156.12it/s] 77%|███████▋  | 308189/400000 [00:43<00:13, 7061.38it/s] 77%|███████▋  | 308896/400000 [00:43<00:12, 7063.83it/s] 77%|███████▋  | 309603/400000 [00:43<00:12, 7010.68it/s] 78%|███████▊  | 310305/400000 [00:44<00:12, 6977.99it/s] 78%|███████▊  | 311004/400000 [00:44<00:13, 6777.97it/s] 78%|███████▊  | 311702/400000 [00:44<00:12, 6832.92it/s] 78%|███████▊  | 312388/400000 [00:44<00:12, 6840.09it/s] 78%|███████▊  | 313088/400000 [00:44<00:12, 6884.81it/s] 78%|███████▊  | 313800/400000 [00:44<00:12, 6951.32it/s] 79%|███████▊  | 314496/400000 [00:44<00:12, 6892.25it/s] 79%|███████▉  | 315203/400000 [00:44<00:12, 6943.36it/s] 79%|███████▉  | 315915/400000 [00:44<00:12, 6992.95it/s] 79%|███████▉  | 316638/400000 [00:44<00:11, 7061.79it/s] 79%|███████▉  | 317369/400000 [00:45<00:11, 7133.51it/s] 80%|███████▉  | 318083/400000 [00:45<00:11, 6967.45it/s] 80%|███████▉  | 318788/400000 [00:45<00:11, 6989.89it/s] 80%|███████▉  | 319510/400000 [00:45<00:11, 7056.80it/s] 80%|████████  | 320241/400000 [00:45<00:11, 7127.86it/s] 80%|████████  | 320955/400000 [00:45<00:11, 6718.84it/s] 80%|████████  | 321695/400000 [00:45<00:11, 6909.46it/s] 81%|████████  | 322392/400000 [00:45<00:11, 6698.95it/s] 81%|████████  | 323110/400000 [00:45<00:11, 6836.09it/s] 81%|████████  | 323819/400000 [00:45<00:11, 6909.31it/s] 81%|████████  | 324524/400000 [00:46<00:10, 6948.97it/s] 81%|████████▏ | 325249/400000 [00:46<00:10, 7035.01it/s] 81%|████████▏ | 325958/400000 [00:46<00:10, 7050.85it/s] 82%|████████▏ | 326666/400000 [00:46<00:10, 7058.21it/s] 82%|████████▏ | 327376/400000 [00:46<00:10, 7069.47it/s] 82%|████████▏ | 328092/400000 [00:46<00:10, 7094.17it/s] 82%|████████▏ | 328809/400000 [00:46<00:10, 7114.63it/s] 82%|████████▏ | 329521/400000 [00:46<00:10, 6968.99it/s] 83%|████████▎ | 330257/400000 [00:46<00:09, 7079.89it/s] 83%|████████▎ | 330989/400000 [00:46<00:09, 7147.91it/s] 83%|████████▎ | 331705/400000 [00:47<00:09, 7086.38it/s] 83%|████████▎ | 332416/400000 [00:47<00:09, 7091.70it/s] 83%|████████▎ | 333126/400000 [00:47<00:09, 6921.39it/s] 83%|████████▎ | 333824/400000 [00:47<00:09, 6938.26it/s] 84%|████████▎ | 334554/400000 [00:47<00:09, 7040.84it/s] 84%|████████▍ | 335277/400000 [00:47<00:09, 7093.33it/s] 84%|████████▍ | 336003/400000 [00:47<00:08, 7140.53it/s] 84%|████████▍ | 336718/400000 [00:47<00:08, 7081.17it/s] 84%|████████▍ | 337427/400000 [00:47<00:09, 6841.42it/s] 85%|████████▍ | 338152/400000 [00:48<00:08, 6957.23it/s] 85%|████████▍ | 338888/400000 [00:48<00:08, 7072.93it/s] 85%|████████▍ | 339613/400000 [00:48<00:08, 7121.51it/s] 85%|████████▌ | 340327/400000 [00:48<00:08, 7119.45it/s] 85%|████████▌ | 341055/400000 [00:48<00:08, 7166.24it/s] 85%|████████▌ | 341781/400000 [00:48<00:08, 7192.67it/s] 86%|████████▌ | 342501/400000 [00:48<00:07, 7192.66it/s] 86%|████████▌ | 343221/400000 [00:48<00:07, 7192.41it/s] 86%|████████▌ | 343941/400000 [00:48<00:07, 7144.48it/s] 86%|████████▌ | 344656/400000 [00:48<00:07, 6973.13it/s] 86%|████████▋ | 345360/400000 [00:49<00:07, 6991.31it/s] 87%|████████▋ | 346083/400000 [00:49<00:07, 7058.48it/s] 87%|████████▋ | 346797/400000 [00:49<00:07, 7081.02it/s] 87%|████████▋ | 347506/400000 [00:49<00:07, 7029.70it/s] 87%|████████▋ | 348224/400000 [00:49<00:07, 7071.73it/s] 87%|████████▋ | 348932/400000 [00:49<00:07, 6851.58it/s] 87%|████████▋ | 349672/400000 [00:49<00:07, 7006.23it/s] 88%|████████▊ | 350418/400000 [00:49<00:06, 7135.32it/s] 88%|████████▊ | 351149/400000 [00:49<00:06, 7185.53it/s] 88%|████████▊ | 351870/400000 [00:49<00:06, 7182.70it/s] 88%|████████▊ | 352590/400000 [00:50<00:06, 7152.98it/s] 88%|████████▊ | 353307/400000 [00:50<00:06, 7121.68it/s] 89%|████████▊ | 354024/400000 [00:50<00:06, 7133.20it/s] 89%|████████▊ | 354738/400000 [00:50<00:06, 7090.74it/s] 89%|████████▉ | 355448/400000 [00:50<00:06, 7085.73it/s] 89%|████████▉ | 356157/400000 [00:50<00:06, 7048.97it/s] 89%|████████▉ | 356871/400000 [00:50<00:06, 7075.52it/s] 89%|████████▉ | 357579/400000 [00:50<00:06, 7068.88it/s] 90%|████████▉ | 358308/400000 [00:50<00:05, 7131.20it/s] 90%|████████▉ | 359043/400000 [00:50<00:05, 7195.09it/s] 90%|████████▉ | 359773/400000 [00:51<00:05, 7224.47it/s] 90%|█████████ | 360496/400000 [00:51<00:05, 7220.35it/s] 90%|█████████ | 361219/400000 [00:51<00:05, 7135.33it/s] 90%|█████████ | 361933/400000 [00:51<00:05, 7112.58it/s] 91%|█████████ | 362650/400000 [00:51<00:05, 7128.12it/s] 91%|█████████ | 363363/400000 [00:51<00:05, 7119.62it/s] 91%|█████████ | 364079/400000 [00:51<00:05, 7131.19it/s] 91%|█████████ | 364793/400000 [00:51<00:04, 7105.59it/s] 91%|█████████▏| 365507/400000 [00:51<00:04, 7115.73it/s] 92%|█████████▏| 366230/400000 [00:51<00:04, 7148.94it/s] 92%|█████████▏| 366945/400000 [00:52<00:04, 7143.39it/s] 92%|█████████▏| 367660/400000 [00:52<00:04, 7099.50it/s] 92%|█████████▏| 368383/400000 [00:52<00:04, 7137.78it/s] 92%|█████████▏| 369097/400000 [00:52<00:04, 7086.01it/s] 92%|█████████▏| 369806/400000 [00:52<00:04, 7048.24it/s] 93%|█████████▎| 370534/400000 [00:52<00:04, 7116.08it/s] 93%|█████████▎| 371246/400000 [00:52<00:04, 7108.52it/s] 93%|█████████▎| 371972/400000 [00:52<00:03, 7151.63it/s] 93%|█████████▎| 372688/400000 [00:52<00:03, 7128.03it/s] 93%|█████████▎| 373401/400000 [00:52<00:03, 6995.45it/s] 94%|█████████▎| 374118/400000 [00:53<00:03, 7044.43it/s] 94%|█████████▎| 374835/400000 [00:53<00:03, 7081.21it/s] 94%|█████████▍| 375549/400000 [00:53<00:03, 7098.55it/s] 94%|█████████▍| 376261/400000 [00:53<00:03, 7104.28it/s] 94%|█████████▍| 376974/400000 [00:53<00:03, 7106.01it/s] 94%|█████████▍| 377690/400000 [00:53<00:03, 7121.54it/s] 95%|█████████▍| 378403/400000 [00:53<00:03, 7118.53it/s] 95%|█████████▍| 379115/400000 [00:53<00:02, 7115.78it/s] 95%|█████████▍| 379827/400000 [00:53<00:02, 7096.37it/s] 95%|█████████▌| 380539/400000 [00:53<00:02, 7100.99it/s] 95%|█████████▌| 381261/400000 [00:54<00:02, 7135.33it/s] 95%|█████████▌| 381975/400000 [00:54<00:02, 7062.83it/s] 96%|█████████▌| 382682/400000 [00:54<00:02, 7058.19it/s] 96%|█████████▌| 383392/400000 [00:54<00:02, 7069.75it/s] 96%|█████████▌| 384111/400000 [00:54<00:02, 7105.34it/s] 96%|█████████▌| 384834/400000 [00:54<00:02, 7141.19it/s] 96%|█████████▋| 385560/400000 [00:54<00:02, 7175.18it/s] 97%|█████████▋| 386278/400000 [00:54<00:01, 7159.47it/s] 97%|█████████▋| 386995/400000 [00:54<00:01, 7135.71it/s] 97%|█████████▋| 387720/400000 [00:54<00:01, 7167.88it/s] 97%|█████████▋| 388443/400000 [00:55<00:01, 7184.68it/s] 97%|█████████▋| 389166/400000 [00:55<00:01, 7198.14it/s] 97%|█████████▋| 389886/400000 [00:55<00:01, 7191.56it/s] 98%|█████████▊| 390606/400000 [00:55<00:01, 7171.43it/s] 98%|█████████▊| 391324/400000 [00:55<00:01, 7142.81it/s] 98%|█████████▊| 392041/400000 [00:55<00:01, 7148.18it/s] 98%|█████████▊| 392775/400000 [00:55<00:01, 7204.02it/s] 98%|█████████▊| 393517/400000 [00:55<00:00, 7266.74it/s] 99%|█████████▊| 394244/400000 [00:55<00:00, 7233.05it/s] 99%|█████████▊| 394968/400000 [00:55<00:00, 7065.38it/s] 99%|█████████▉| 395676/400000 [00:56<00:00, 7042.16it/s] 99%|█████████▉| 396386/400000 [00:56<00:00, 7056.72it/s] 99%|█████████▉| 397093/400000 [00:56<00:00, 7049.00it/s] 99%|█████████▉| 397812/400000 [00:56<00:00, 7089.58it/s]100%|█████████▉| 398522/400000 [00:56<00:00, 7064.19it/s]100%|█████████▉| 399243/400000 [00:56<00:00, 7105.75it/s]100%|█████████▉| 399966/400000 [00:56<00:00, 7140.87it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7055.69it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff99c080e10>, <torchtext.data.dataset.TabularDataset object at 0x7ff99c080f60>, <torchtext.vocab.Vocab object at 0x7ff99c080e80>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 26.67 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 47.10 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.75 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.31 file/s]2020-07-18 18:58:50.623252: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-18 18:58:50.627770: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-18 18:58:50.628631: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561ddefce280 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 18:58:50.628656: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▎      | 3325952/9912422 [00:00<00:00, 33255349.51it/s]9920512it [00:00, 33224800.05it/s]                             
0it [00:00, ?it/s]32768it [00:00, 512579.45it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160098.31it/s]1654784it [00:00, 11298792.01it/s]                         
0it [00:00, ?it/s]8192it [00:00, 177352.48it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
