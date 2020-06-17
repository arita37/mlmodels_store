
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb4f4b1bea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb4f4b1bea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb5600d30d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb5600d30d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb579dffea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb579dffea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb50d94e950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb50d94e950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb50d94e950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|██        | 1990656/9912422 [00:00<00:00, 19225617.27it/s]9920512it [00:00, 32740966.58it/s]                             
0it [00:00, ?it/s]32768it [00:00, 688285.70it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1005985.33it/s]1654784it [00:00, 12827053.15it/s]                           
0it [00:00, ?it/s]8192it [00:00, 261569.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50af9c710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50af92978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb50d94e598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb50d94e598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb50d94e598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:05,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:04,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:04,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:04,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:44,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:43,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:30,  4.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:30,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:30,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:14,  8.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 12.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 26.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 38.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 44.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 49.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 53.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 61.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.82s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.82s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.26s/ url]
0 examples [00:00, ? examples/s]2020-06-17 00:09:34.664505: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-17 00:09:34.677130: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-17 00:09:34.677341: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d601c37740 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 00:09:34.677391: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
41 examples [00:00, 406.96 examples/s]130 examples [00:00, 485.95 examples/s]221 examples [00:00, 564.20 examples/s]313 examples [00:00, 637.30 examples/s]409 examples [00:00, 706.25 examples/s]504 examples [00:00, 763.32 examples/s]590 examples [00:00, 789.09 examples/s]673 examples [00:00, 800.33 examples/s]764 examples [00:00, 828.26 examples/s]853 examples [00:01, 845.18 examples/s]939 examples [00:01, 847.11 examples/s]1025 examples [00:01, 841.80 examples/s]1113 examples [00:01, 851.02 examples/s]1202 examples [00:01, 861.21 examples/s]1290 examples [00:01, 866.01 examples/s]1384 examples [00:01, 884.57 examples/s]1473 examples [00:01, 859.91 examples/s]1560 examples [00:01, 851.97 examples/s]1652 examples [00:01, 869.95 examples/s]1743 examples [00:02, 881.53 examples/s]1833 examples [00:02, 886.41 examples/s]1927 examples [00:02, 899.70 examples/s]2027 examples [00:02, 927.30 examples/s]2121 examples [00:02, 916.34 examples/s]2213 examples [00:02, 913.03 examples/s]2307 examples [00:02, 918.56 examples/s]2399 examples [00:02, 891.55 examples/s]2491 examples [00:02, 898.15 examples/s]2582 examples [00:02, 899.84 examples/s]2673 examples [00:03, 888.04 examples/s]2769 examples [00:03, 906.17 examples/s]2860 examples [00:03, 892.98 examples/s]2951 examples [00:03, 896.33 examples/s]3041 examples [00:03, 880.20 examples/s]3133 examples [00:03, 891.31 examples/s]3228 examples [00:03, 907.89 examples/s]3319 examples [00:03, 902.63 examples/s]3415 examples [00:03, 917.09 examples/s]3513 examples [00:03, 934.94 examples/s]3607 examples [00:04, 935.68 examples/s]3705 examples [00:04, 946.81 examples/s]3800 examples [00:04, 929.29 examples/s]3894 examples [00:04, 913.32 examples/s]3986 examples [00:04, 909.83 examples/s]4081 examples [00:04, 920.32 examples/s]4178 examples [00:04, 934.67 examples/s]4272 examples [00:04, 924.73 examples/s]4368 examples [00:04, 934.58 examples/s]4468 examples [00:04, 951.09 examples/s]4564 examples [00:05, 949.68 examples/s]4660 examples [00:05, 918.54 examples/s]4753 examples [00:05, 912.12 examples/s]4848 examples [00:05, 922.63 examples/s]4942 examples [00:05, 922.62 examples/s]5035 examples [00:05, 919.29 examples/s]5131 examples [00:05, 928.52 examples/s]5224 examples [00:05, 925.71 examples/s]5320 examples [00:05, 933.06 examples/s]5415 examples [00:06, 936.90 examples/s]5509 examples [00:06, 912.88 examples/s]5606 examples [00:06, 927.60 examples/s]5704 examples [00:06, 940.67 examples/s]5799 examples [00:06, 941.09 examples/s]5895 examples [00:06, 945.24 examples/s]5990 examples [00:06, 929.43 examples/s]6087 examples [00:06, 940.08 examples/s]6182 examples [00:06, 932.75 examples/s]6276 examples [00:06, 924.58 examples/s]6369 examples [00:07, 920.31 examples/s]6462 examples [00:07, 912.99 examples/s]6554 examples [00:07, 902.44 examples/s]6648 examples [00:07, 910.97 examples/s]6740 examples [00:07, 901.86 examples/s]6837 examples [00:07, 918.96 examples/s]6934 examples [00:07, 932.24 examples/s]7028 examples [00:07, 929.03 examples/s]7121 examples [00:07, 921.06 examples/s]7219 examples [00:07, 936.51 examples/s]7317 examples [00:08, 948.89 examples/s]7415 examples [00:08, 957.09 examples/s]7511 examples [00:08, 954.97 examples/s]7607 examples [00:08, 953.72 examples/s]7705 examples [00:08, 959.37 examples/s]7803 examples [00:08, 965.41 examples/s]7900 examples [00:08, 965.21 examples/s]7997 examples [00:08, 957.72 examples/s]8093 examples [00:08, 947.14 examples/s]8188 examples [00:08, 944.28 examples/s]8284 examples [00:09, 948.06 examples/s]8379 examples [00:09, 948.33 examples/s]8474 examples [00:09, 931.26 examples/s]8569 examples [00:09, 934.23 examples/s]8663 examples [00:09, 928.18 examples/s]8758 examples [00:09, 933.59 examples/s]8852 examples [00:09, 902.17 examples/s]8943 examples [00:09, 880.52 examples/s]9032 examples [00:09, 878.49 examples/s]9121 examples [00:10, 865.18 examples/s]9215 examples [00:10, 883.84 examples/s]9308 examples [00:10, 897.07 examples/s]9399 examples [00:10, 900.58 examples/s]9492 examples [00:10, 907.60 examples/s]9583 examples [00:10, 881.14 examples/s]9676 examples [00:10, 893.47 examples/s]9773 examples [00:10, 912.85 examples/s]9867 examples [00:10, 919.92 examples/s]9960 examples [00:10, 893.01 examples/s]10050 examples [00:11, 867.64 examples/s]10141 examples [00:11, 878.96 examples/s]10230 examples [00:11, 873.03 examples/s]10318 examples [00:11, 867.13 examples/s]10409 examples [00:11, 877.25 examples/s]10497 examples [00:11, 870.85 examples/s]10587 examples [00:11, 878.66 examples/s]10679 examples [00:11, 889.33 examples/s]10772 examples [00:11, 900.62 examples/s]10869 examples [00:11, 918.36 examples/s]10968 examples [00:12, 938.12 examples/s]11063 examples [00:12, 937.40 examples/s]11159 examples [00:12, 942.05 examples/s]11254 examples [00:12, 939.98 examples/s]11349 examples [00:12, 932.07 examples/s]11444 examples [00:12, 934.58 examples/s]11538 examples [00:12, 929.86 examples/s]11632 examples [00:12, 932.22 examples/s]11726 examples [00:12, 920.46 examples/s]11823 examples [00:12, 933.94 examples/s]11917 examples [00:13, 918.60 examples/s]12010 examples [00:13, 920.52 examples/s]12103 examples [00:13, 898.60 examples/s]12194 examples [00:13, 862.91 examples/s]12281 examples [00:13, 859.27 examples/s]12368 examples [00:13, 854.49 examples/s]12461 examples [00:13, 873.87 examples/s]12549 examples [00:13, 871.41 examples/s]12640 examples [00:13, 880.33 examples/s]12729 examples [00:14, 877.35 examples/s]12822 examples [00:14, 891.64 examples/s]12918 examples [00:14, 910.02 examples/s]13010 examples [00:14, 898.17 examples/s]13100 examples [00:14, 892.23 examples/s]13195 examples [00:14, 907.67 examples/s]13290 examples [00:14, 916.83 examples/s]13386 examples [00:14, 925.44 examples/s]13482 examples [00:14, 933.52 examples/s]13578 examples [00:14, 939.57 examples/s]13673 examples [00:15, 935.42 examples/s]13769 examples [00:15, 940.03 examples/s]13864 examples [00:15, 910.78 examples/s]13956 examples [00:15, 910.96 examples/s]14053 examples [00:15, 925.19 examples/s]14146 examples [00:15, 912.64 examples/s]14240 examples [00:15, 919.03 examples/s]14336 examples [00:15, 928.69 examples/s]14429 examples [00:15, 927.89 examples/s]14522 examples [00:15, 904.90 examples/s]14616 examples [00:16, 912.96 examples/s]14712 examples [00:16, 925.54 examples/s]14809 examples [00:16, 935.92 examples/s]14907 examples [00:16, 947.61 examples/s]15003 examples [00:16, 949.70 examples/s]15099 examples [00:16, 936.02 examples/s]15193 examples [00:16, 917.98 examples/s]15290 examples [00:16, 929.87 examples/s]15387 examples [00:16, 940.41 examples/s]15482 examples [00:17, 912.52 examples/s]15575 examples [00:17, 915.45 examples/s]15667 examples [00:17, 914.64 examples/s]15759 examples [00:17, 905.24 examples/s]15850 examples [00:17, 905.51 examples/s]15944 examples [00:17, 911.16 examples/s]16042 examples [00:17, 930.30 examples/s]16139 examples [00:17, 940.53 examples/s]16236 examples [00:17, 949.10 examples/s]16334 examples [00:17, 956.98 examples/s]16430 examples [00:18, 915.87 examples/s]16523 examples [00:18, 914.14 examples/s]16616 examples [00:18, 917.32 examples/s]16711 examples [00:18, 924.16 examples/s]16807 examples [00:18, 925.71 examples/s]16900 examples [00:18, 921.23 examples/s]16993 examples [00:18, 905.80 examples/s]17089 examples [00:18, 918.08 examples/s]17187 examples [00:18, 933.89 examples/s]17285 examples [00:18, 944.16 examples/s]17380 examples [00:19, 935.78 examples/s]17474 examples [00:19, 920.23 examples/s]17567 examples [00:19, 914.70 examples/s]17661 examples [00:19, 920.71 examples/s]17757 examples [00:19, 930.96 examples/s]17853 examples [00:19, 938.50 examples/s]17951 examples [00:19, 949.73 examples/s]18050 examples [00:19, 958.76 examples/s]18150 examples [00:19, 970.27 examples/s]18249 examples [00:19, 970.82 examples/s]18347 examples [00:20, 943.65 examples/s]18442 examples [00:20, 937.88 examples/s]18536 examples [00:20, 928.15 examples/s]18629 examples [00:20, 905.21 examples/s]18720 examples [00:20, 891.75 examples/s]18810 examples [00:20, 893.81 examples/s]18908 examples [00:20, 917.11 examples/s]19005 examples [00:20, 930.92 examples/s]19103 examples [00:20, 942.61 examples/s]19201 examples [00:20, 952.78 examples/s]19297 examples [00:21, 951.88 examples/s]19393 examples [00:21, 943.72 examples/s]19488 examples [00:21, 945.51 examples/s]19583 examples [00:21, 946.05 examples/s]19678 examples [00:21, 940.91 examples/s]19773 examples [00:21, 930.90 examples/s]19867 examples [00:21, 927.54 examples/s]19961 examples [00:21, 929.11 examples/s]20054 examples [00:21, 892.07 examples/s]20150 examples [00:22, 908.83 examples/s]20242 examples [00:22, 868.32 examples/s]20331 examples [00:22, 873.56 examples/s]20421 examples [00:22, 880.22 examples/s]20510 examples [00:22, 875.00 examples/s]20605 examples [00:22, 893.77 examples/s]20695 examples [00:22, 891.15 examples/s]20787 examples [00:22, 896.88 examples/s]20883 examples [00:22, 913.37 examples/s]20978 examples [00:22, 923.99 examples/s]21071 examples [00:23, 910.95 examples/s]21163 examples [00:23, 877.04 examples/s]21252 examples [00:23, 877.71 examples/s]21341 examples [00:23, 868.77 examples/s]21430 examples [00:23, 871.95 examples/s]21518 examples [00:23, 853.76 examples/s]21605 examples [00:23, 858.43 examples/s]21691 examples [00:23, 857.50 examples/s]21781 examples [00:23, 868.93 examples/s]21876 examples [00:23, 889.26 examples/s]21970 examples [00:24, 902.47 examples/s]22065 examples [00:24, 915.36 examples/s]22164 examples [00:24, 935.60 examples/s]22262 examples [00:24, 947.33 examples/s]22357 examples [00:24, 946.36 examples/s]22452 examples [00:24, 918.65 examples/s]22549 examples [00:24, 932.61 examples/s]22643 examples [00:24, 917.85 examples/s]22736 examples [00:24, 907.77 examples/s]22829 examples [00:25, 912.92 examples/s]22925 examples [00:25, 925.82 examples/s]23020 examples [00:25, 932.44 examples/s]23114 examples [00:25, 925.49 examples/s]23210 examples [00:25, 935.13 examples/s]23308 examples [00:25, 946.43 examples/s]23403 examples [00:25, 933.46 examples/s]23497 examples [00:25, 909.04 examples/s]23589 examples [00:25, 892.15 examples/s]23681 examples [00:25, 898.83 examples/s]23774 examples [00:26, 906.81 examples/s]23865 examples [00:26, 900.10 examples/s]23959 examples [00:26, 910.01 examples/s]24052 examples [00:26, 913.35 examples/s]24146 examples [00:26, 920.77 examples/s]24243 examples [00:26, 934.35 examples/s]24340 examples [00:26, 943.90 examples/s]24438 examples [00:26, 953.07 examples/s]24534 examples [00:26, 944.86 examples/s]24629 examples [00:26, 938.96 examples/s]24723 examples [00:27, 922.40 examples/s]24816 examples [00:27, 891.81 examples/s]24912 examples [00:27, 910.50 examples/s]25004 examples [00:27, 908.80 examples/s]25101 examples [00:27, 924.70 examples/s]25197 examples [00:27, 933.98 examples/s]25293 examples [00:27, 939.78 examples/s]25388 examples [00:27, 938.50 examples/s]25482 examples [00:27, 911.23 examples/s]25577 examples [00:27, 922.32 examples/s]25670 examples [00:28, 876.46 examples/s]25759 examples [00:28, 877.40 examples/s]25852 examples [00:28, 890.17 examples/s]25946 examples [00:28, 902.26 examples/s]26037 examples [00:28, 899.38 examples/s]26132 examples [00:28, 911.33 examples/s]26224 examples [00:28, 904.33 examples/s]26318 examples [00:28, 914.00 examples/s]26410 examples [00:28, 905.85 examples/s]26501 examples [00:29, 884.80 examples/s]26595 examples [00:29, 899.64 examples/s]26686 examples [00:29, 876.59 examples/s]26778 examples [00:29, 888.00 examples/s]26873 examples [00:29, 903.28 examples/s]26968 examples [00:29, 915.78 examples/s]27065 examples [00:29, 930.60 examples/s]27159 examples [00:29, 931.11 examples/s]27257 examples [00:29, 942.78 examples/s]27356 examples [00:29, 953.98 examples/s]27452 examples [00:30, 939.25 examples/s]27549 examples [00:30, 948.07 examples/s]27644 examples [00:30, 929.52 examples/s]27738 examples [00:30, 930.54 examples/s]27835 examples [00:30, 939.70 examples/s]27930 examples [00:30, 926.76 examples/s]28024 examples [00:30, 928.09 examples/s]28118 examples [00:30, 930.98 examples/s]28213 examples [00:30, 935.26 examples/s]28307 examples [00:30, 916.00 examples/s]28399 examples [00:31, 897.88 examples/s]28489 examples [00:31, 880.81 examples/s]28580 examples [00:31, 888.22 examples/s]28669 examples [00:31, 858.77 examples/s]28762 examples [00:31, 875.84 examples/s]28858 examples [00:31, 898.96 examples/s]28949 examples [00:31, 898.45 examples/s]29040 examples [00:31, 897.64 examples/s]29130 examples [00:31, 893.53 examples/s]29225 examples [00:32, 908.51 examples/s]29323 examples [00:32, 927.74 examples/s]29421 examples [00:32, 941.11 examples/s]29516 examples [00:32, 869.66 examples/s]29610 examples [00:32, 887.56 examples/s]29707 examples [00:32, 909.10 examples/s]29799 examples [00:32, 902.09 examples/s]29890 examples [00:32, 894.23 examples/s]29985 examples [00:32, 908.25 examples/s]30077 examples [00:32, 871.41 examples/s]30172 examples [00:33, 891.17 examples/s]30270 examples [00:33, 915.16 examples/s]30363 examples [00:33, 915.79 examples/s]30458 examples [00:33, 925.72 examples/s]30555 examples [00:33, 936.77 examples/s]30651 examples [00:33, 941.88 examples/s]30747 examples [00:33, 946.52 examples/s]30842 examples [00:33, 935.72 examples/s]30940 examples [00:33, 946.50 examples/s]31035 examples [00:33, 943.34 examples/s]31132 examples [00:34, 950.83 examples/s]31229 examples [00:34, 954.65 examples/s]31325 examples [00:34, 937.33 examples/s]31419 examples [00:34, 925.11 examples/s]31512 examples [00:34, 909.52 examples/s]31604 examples [00:34, 898.17 examples/s]31694 examples [00:34, 898.11 examples/s]31785 examples [00:34, 901.22 examples/s]31883 examples [00:34, 921.94 examples/s]31976 examples [00:35, 915.29 examples/s]32068 examples [00:35, 905.91 examples/s]32159 examples [00:35, 898.55 examples/s]32249 examples [00:35, 861.10 examples/s]32336 examples [00:35, 861.61 examples/s]32423 examples [00:35, 842.58 examples/s]32508 examples [00:35, 836.56 examples/s]32600 examples [00:35, 858.21 examples/s]32687 examples [00:35, 856.31 examples/s]32778 examples [00:35, 871.51 examples/s]32869 examples [00:36, 881.94 examples/s]32959 examples [00:36, 885.34 examples/s]33048 examples [00:36, 884.32 examples/s]33137 examples [00:36, 885.22 examples/s]33230 examples [00:36, 897.86 examples/s]33327 examples [00:36, 917.22 examples/s]33420 examples [00:36, 919.38 examples/s]33515 examples [00:36, 925.60 examples/s]33608 examples [00:36, 913.15 examples/s]33700 examples [00:36, 909.97 examples/s]33793 examples [00:37, 913.28 examples/s]33885 examples [00:37, 909.59 examples/s]33977 examples [00:37, 904.01 examples/s]34069 examples [00:37, 908.55 examples/s]34160 examples [00:37, 908.98 examples/s]34255 examples [00:37, 918.26 examples/s]34351 examples [00:37, 927.66 examples/s]34447 examples [00:37, 935.43 examples/s]34541 examples [00:37, 935.93 examples/s]34638 examples [00:37, 943.26 examples/s]34736 examples [00:38, 951.65 examples/s]34832 examples [00:38, 931.16 examples/s]34926 examples [00:38, 915.24 examples/s]35018 examples [00:38, 867.87 examples/s]35111 examples [00:38, 882.83 examples/s]35208 examples [00:38, 905.23 examples/s]35302 examples [00:38, 910.54 examples/s]35398 examples [00:38, 924.13 examples/s]35491 examples [00:38, 916.16 examples/s]35583 examples [00:39, 901.09 examples/s]35674 examples [00:39, 896.25 examples/s]35764 examples [00:39, 896.93 examples/s]35858 examples [00:39, 907.66 examples/s]35954 examples [00:39, 921.59 examples/s]36047 examples [00:39, 920.96 examples/s]36141 examples [00:39, 925.28 examples/s]36236 examples [00:39, 931.69 examples/s]36331 examples [00:39, 934.74 examples/s]36425 examples [00:39, 935.49 examples/s]36519 examples [00:40, 932.04 examples/s]36614 examples [00:40, 936.04 examples/s]36710 examples [00:40, 940.60 examples/s]36809 examples [00:40, 953.20 examples/s]36906 examples [00:40, 955.27 examples/s]37004 examples [00:40, 961.58 examples/s]37101 examples [00:40, 956.54 examples/s]37198 examples [00:40, 959.55 examples/s]37294 examples [00:40, 955.05 examples/s]37390 examples [00:40, 949.31 examples/s]37485 examples [00:41, 948.09 examples/s]37580 examples [00:41, 934.01 examples/s]37677 examples [00:41, 942.60 examples/s]37772 examples [00:41, 935.11 examples/s]37866 examples [00:41, 895.60 examples/s]37963 examples [00:41, 914.73 examples/s]38062 examples [00:41, 934.58 examples/s]38157 examples [00:41, 937.19 examples/s]38251 examples [00:41, 908.98 examples/s]38343 examples [00:41, 898.75 examples/s]38434 examples [00:42, 899.24 examples/s]38528 examples [00:42, 909.22 examples/s]38621 examples [00:42, 914.22 examples/s]38713 examples [00:42, 912.35 examples/s]38805 examples [00:42, 896.24 examples/s]38896 examples [00:42, 897.96 examples/s]38986 examples [00:42, 896.24 examples/s]39076 examples [00:42, 888.27 examples/s]39165 examples [00:42, 875.64 examples/s]39261 examples [00:42, 897.16 examples/s]39352 examples [00:43, 900.33 examples/s]39447 examples [00:43, 913.02 examples/s]39542 examples [00:43, 921.03 examples/s]39635 examples [00:43, 914.63 examples/s]39727 examples [00:43, 906.82 examples/s]39818 examples [00:43, 895.38 examples/s]39908 examples [00:43, 879.23 examples/s]39997 examples [00:43, 867.34 examples/s]40084 examples [00:43, 817.58 examples/s]40176 examples [00:44, 844.85 examples/s]40262 examples [00:44, 848.09 examples/s]40357 examples [00:44, 874.80 examples/s]40448 examples [00:44, 883.84 examples/s]40537 examples [00:44, 847.14 examples/s]40624 examples [00:44, 852.47 examples/s]40710 examples [00:44, 847.02 examples/s]40798 examples [00:44, 853.15 examples/s]40891 examples [00:44, 874.33 examples/s]40983 examples [00:44, 885.41 examples/s]41080 examples [00:45, 908.41 examples/s]41173 examples [00:45, 914.05 examples/s]41265 examples [00:45, 909.25 examples/s]41357 examples [00:45, 897.53 examples/s]41447 examples [00:45, 888.72 examples/s]41545 examples [00:45, 911.90 examples/s]41637 examples [00:45, 886.48 examples/s]41727 examples [00:45, 889.22 examples/s]41817 examples [00:45, 879.46 examples/s]41906 examples [00:45, 877.32 examples/s]41994 examples [00:46, 877.28 examples/s]42084 examples [00:46, 883.55 examples/s]42175 examples [00:46, 889.72 examples/s]42266 examples [00:46, 895.04 examples/s]42356 examples [00:46, 892.58 examples/s]42446 examples [00:46, 892.21 examples/s]42540 examples [00:46, 903.13 examples/s]42635 examples [00:46, 915.66 examples/s]42729 examples [00:46, 922.53 examples/s]42823 examples [00:46, 926.04 examples/s]42916 examples [00:47, 919.67 examples/s]43010 examples [00:47, 925.22 examples/s]43107 examples [00:47, 936.86 examples/s]43201 examples [00:47, 927.06 examples/s]43294 examples [00:47, 900.39 examples/s]43386 examples [00:47, 906.01 examples/s]43477 examples [00:47, 896.71 examples/s]43567 examples [00:47, 883.03 examples/s]43660 examples [00:47, 895.37 examples/s]43750 examples [00:48, 887.38 examples/s]43840 examples [00:48, 889.05 examples/s]43930 examples [00:48, 889.94 examples/s]44020 examples [00:48, 872.99 examples/s]44113 examples [00:48, 887.50 examples/s]44203 examples [00:48, 891.14 examples/s]44297 examples [00:48, 904.46 examples/s]44388 examples [00:48, 904.01 examples/s]44479 examples [00:48, 902.87 examples/s]44575 examples [00:48, 918.82 examples/s]44668 examples [00:49, 920.82 examples/s]44761 examples [00:49, 918.18 examples/s]44853 examples [00:49, 894.76 examples/s]44943 examples [00:49, 854.29 examples/s]45029 examples [00:49, 828.29 examples/s]45113 examples [00:49, 826.30 examples/s]45202 examples [00:49, 843.05 examples/s]45299 examples [00:49, 876.09 examples/s]45396 examples [00:49, 901.62 examples/s]45490 examples [00:49, 911.64 examples/s]45582 examples [00:50, 908.52 examples/s]45674 examples [00:50, 910.75 examples/s]45766 examples [00:50, 907.17 examples/s]45857 examples [00:50, 905.70 examples/s]45950 examples [00:50, 912.21 examples/s]46042 examples [00:50, 902.28 examples/s]46137 examples [00:50, 915.53 examples/s]46234 examples [00:50, 928.87 examples/s]46330 examples [00:50, 937.43 examples/s]46424 examples [00:50, 937.75 examples/s]46518 examples [00:51, 935.57 examples/s]46612 examples [00:51, 913.90 examples/s]46704 examples [00:51, 902.85 examples/s]46795 examples [00:51, 902.09 examples/s]46887 examples [00:51, 906.48 examples/s]46978 examples [00:51, 901.61 examples/s]47069 examples [00:51, 901.73 examples/s]47160 examples [00:51, 899.93 examples/s]47252 examples [00:51, 903.74 examples/s]47343 examples [00:52, 902.48 examples/s]47437 examples [00:52, 910.57 examples/s]47534 examples [00:52, 927.21 examples/s]47630 examples [00:52, 934.34 examples/s]47726 examples [00:52, 941.26 examples/s]47824 examples [00:52, 950.10 examples/s]47922 examples [00:52, 957.33 examples/s]48018 examples [00:52, 952.54 examples/s]48114 examples [00:52, 954.64 examples/s]48211 examples [00:52, 958.29 examples/s]48308 examples [00:53, 958.36 examples/s]48404 examples [00:53, 945.96 examples/s]48500 examples [00:53, 947.73 examples/s]48597 examples [00:53, 953.08 examples/s]48693 examples [00:53, 947.97 examples/s]48790 examples [00:53, 951.22 examples/s]48886 examples [00:53, 908.57 examples/s]48983 examples [00:53, 925.80 examples/s]49080 examples [00:53, 936.67 examples/s]49177 examples [00:53, 944.35 examples/s]49272 examples [00:54, 926.27 examples/s]49371 examples [00:54, 942.40 examples/s]49466 examples [00:54, 935.43 examples/s]49561 examples [00:54, 939.01 examples/s]49658 examples [00:54, 946.13 examples/s]49753 examples [00:54, 944.06 examples/s]49848 examples [00:54, 945.50 examples/s]49947 examples [00:54, 958.25 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6556/50000 [00:00<00:00, 65558.81 examples/s] 35%|███▌      | 17740/50000 [00:00<00:00, 74851.14 examples/s] 57%|█████▋    | 28727/50000 [00:00<00:00, 82764.76 examples/s] 81%|████████  | 40270/50000 [00:00<00:00, 90442.75 examples/s]                                                               0 examples [00:00, ? examples/s]81 examples [00:00, 804.90 examples/s]176 examples [00:00, 841.80 examples/s]272 examples [00:00, 872.63 examples/s]370 examples [00:00, 900.00 examples/s]469 examples [00:00, 923.90 examples/s]566 examples [00:00, 936.99 examples/s]665 examples [00:00, 951.96 examples/s]758 examples [00:00, 943.76 examples/s]857 examples [00:00, 956.74 examples/s]956 examples [00:01, 963.77 examples/s]1053 examples [00:01, 964.23 examples/s]1148 examples [00:01, 936.28 examples/s]1243 examples [00:01, 938.85 examples/s]1337 examples [00:01, 927.27 examples/s]1434 examples [00:01, 938.97 examples/s]1532 examples [00:01, 950.08 examples/s]1627 examples [00:01, 947.18 examples/s]1723 examples [00:01, 949.51 examples/s]1822 examples [00:01, 956.79 examples/s]1920 examples [00:02, 962.87 examples/s]2018 examples [00:02, 965.83 examples/s]2115 examples [00:02, 938.25 examples/s]2209 examples [00:02, 924.25 examples/s]2303 examples [00:02, 926.46 examples/s]2399 examples [00:02, 934.99 examples/s]2497 examples [00:02, 946.42 examples/s]2597 examples [00:02, 959.10 examples/s]2694 examples [00:02, 943.79 examples/s]2793 examples [00:02, 955.71 examples/s]2892 examples [00:03, 963.77 examples/s]2990 examples [00:03, 965.97 examples/s]3088 examples [00:03, 967.66 examples/s]3185 examples [00:03, 951.33 examples/s]3282 examples [00:03, 956.19 examples/s]3379 examples [00:03, 959.28 examples/s]3475 examples [00:03, 948.09 examples/s]3572 examples [00:03, 953.42 examples/s]3668 examples [00:03, 933.12 examples/s]3762 examples [00:03, 927.05 examples/s]3855 examples [00:04, 917.29 examples/s]3948 examples [00:04, 919.28 examples/s]4040 examples [00:04, 917.15 examples/s]4132 examples [00:04, 881.71 examples/s]4225 examples [00:04, 894.80 examples/s]4326 examples [00:04, 924.09 examples/s]4425 examples [00:04, 942.71 examples/s]4524 examples [00:04, 955.67 examples/s]4620 examples [00:04, 943.96 examples/s]4715 examples [00:05, 922.78 examples/s]4808 examples [00:05, 922.84 examples/s]4901 examples [00:05, 905.57 examples/s]4995 examples [00:05, 912.24 examples/s]5087 examples [00:05, 873.48 examples/s]5180 examples [00:05, 888.93 examples/s]5270 examples [00:05, 881.72 examples/s]5359 examples [00:05, 868.49 examples/s]5447 examples [00:05, 861.11 examples/s]5534 examples [00:05, 857.30 examples/s]5625 examples [00:06, 869.67 examples/s]5716 examples [00:06, 878.82 examples/s]5805 examples [00:06, 868.99 examples/s]5899 examples [00:06, 888.30 examples/s]5989 examples [00:06, 891.51 examples/s]6083 examples [00:06, 903.35 examples/s]6184 examples [00:06, 930.74 examples/s]6283 examples [00:06, 945.75 examples/s]6378 examples [00:06, 911.24 examples/s]6472 examples [00:06, 917.94 examples/s]6569 examples [00:07, 930.47 examples/s]6663 examples [00:07, 918.48 examples/s]6763 examples [00:07, 940.39 examples/s]6859 examples [00:07, 943.66 examples/s]6954 examples [00:07, 914.44 examples/s]7054 examples [00:07, 934.66 examples/s]7153 examples [00:07, 949.75 examples/s]7249 examples [00:07, 946.38 examples/s]7346 examples [00:07, 952.36 examples/s]7442 examples [00:08, 936.66 examples/s]7536 examples [00:08, 923.83 examples/s]7629 examples [00:08, 910.92 examples/s]7721 examples [00:08, 892.29 examples/s]7811 examples [00:08, 867.02 examples/s]7898 examples [00:08, 851.47 examples/s]7987 examples [00:08, 862.11 examples/s]8081 examples [00:08, 881.82 examples/s]8177 examples [00:08, 903.88 examples/s]8271 examples [00:08, 913.55 examples/s]8363 examples [00:09, 904.16 examples/s]8454 examples [00:09, 879.67 examples/s]8545 examples [00:09, 887.80 examples/s]8634 examples [00:09, 886.93 examples/s]8725 examples [00:09, 891.62 examples/s]8820 examples [00:09, 908.35 examples/s]8920 examples [00:09, 931.92 examples/s]9014 examples [00:09, 916.84 examples/s]9106 examples [00:09, 894.57 examples/s]9196 examples [00:09, 894.73 examples/s]9290 examples [00:10, 906.36 examples/s]9384 examples [00:10, 915.59 examples/s]9476 examples [00:10, 895.95 examples/s]9566 examples [00:10, 884.55 examples/s]9655 examples [00:10, 839.31 examples/s]9750 examples [00:10, 869.14 examples/s]9841 examples [00:10, 880.37 examples/s]9931 examples [00:10, 885.02 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6W3QT3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6W3QT3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb50d94e950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb50d94e950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb50d94e950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb55933b320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb496e3fc50>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb50d94e950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb50d94e950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb50d94e950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50d93cba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50af920b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb485444620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb485444620> 

  function with postional parmater data_info <function split_train_valid at 0x7fb485444620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb485444730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb485444730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb485444730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=07c5d42a59620592cb303e7fb40b62921861ef120239d6a5cb069f5796b53618
  Stored in directory: /tmp/pip-ephem-wheel-cache-ct53c24s/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<27:05:55, 8.84kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<19:12:18, 12.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<13:29:50, 17.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<9:27:17, 25.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:36:03, 36.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.10M/862M [00:01<4:35:33, 51.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<3:12:22, 73.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<2:14:01, 105kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.5M/862M [00:01<1:33:30, 150kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.0M/862M [00:01<1:05:11, 214kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.1M/862M [00:02<45:31, 305kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:02<31:54, 434kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:02<22:22, 616kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.6M/862M [00:02<15:38, 875kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<11:01, 1.24MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<07:44, 1.75MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:37, 2.40MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.1M/862M [00:04<06:00, 2.24MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:04<04:37, 2.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<03:24, 3.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:06<10:54, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:06<10:47, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:06<08:21, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<05:59, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:08<12:00, 1.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:08<09:55, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<07:18, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<07:56, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:10<06:58, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<05:10, 2.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<06:37, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:12<07:20, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:12<05:44, 2.29MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<04:12, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<08:29, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:14<07:17, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<05:22, 2.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<06:49, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:16<07:29, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:16<05:53, 2.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<04:18, 3.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<09:04, 1.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<07:43, 1.68MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<05:43, 2.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:19<07:01, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<04:41, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:21<06:19, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<05:42, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<04:19, 2.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<05:31, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<04:11, 3.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:25<05:55, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:26, 2.33MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:07, 3.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:52, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:13, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:55, 3.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:27<02:54, 4.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<49:30, 254kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<35:56, 350kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<25:23, 494kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<20:41, 604kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<15:45, 793kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<11:16, 1.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<10:49, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:33<08:51, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:30, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:28, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:11, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:03, 2.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<03:41, 3.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<09:17, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<07:31, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:39, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:37<04:05, 2.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<38:07, 321kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<29:10, 419kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<21:02, 581kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<16:38, 731kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<12:55, 940kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<09:21, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<09:19, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<07:46, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:44, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:51, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:48, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:38, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<04:12, 2.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:43, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:15, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<03:58, 2.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<05:34, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<05:07, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<03:51, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:04, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:50, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<05:01, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<03:48, 3.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<05:25, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<05:00, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<03:48, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<05:24, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<04:57, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<03:45, 3.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<05:21, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:45, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<03:43, 3.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<02:44, 4.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<27:49, 413kB/s] .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<24:12, 474kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<18:07, 633kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<12:56, 884kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<10:53, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:54, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:55, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:54, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:59, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<05:29, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:09, 2.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:19, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<06:08, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:11, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:30, 2.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<06:20, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<07:03, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:36, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<04:04, 2.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<10:52, 1.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:52, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<06:54, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:11, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:31, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:05, 2.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:52, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:15, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:57, 2.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:16, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:38, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:34, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<02:36, 4.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<42:28, 254kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<30:49, 349kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<21:46, 493kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<17:44, 603kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<13:30, 792kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:42, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<09:17, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:24, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:25, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:55, 2.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<42:42, 247kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<32:02, 330kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<22:57, 460kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<16:05, 652kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<39:05, 269kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<28:25, 369kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<20:14, 518kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<14:26, 724kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<14:04, 741kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<17:10, 607kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<13:37, 765kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:56, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:13, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<08:28, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<08:56, 1.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:59, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:09, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:52, 2.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<14:45, 698kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<15:57, 645kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<12:35, 817kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<09:10, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:39, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<08:47, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<08:40, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:39, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:55, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:40, 2.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<56:22, 180kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<44:29, 228kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<32:22, 313kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<22:58, 441kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<16:16, 621kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<16:30, 611kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<16:33, 609kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<12:49, 786kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<09:19, 1.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:44, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<09:34, 1.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<11:40, 858kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<09:20, 1.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:52, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:00, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:35, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<10:31, 945kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<08:31, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<09:42, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<11:14, 878kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:58, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:34, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:48, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:46, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<11:17, 868kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:02, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:36, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:47, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<08:14, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<10:08, 960kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<08:02, 1.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:53, 1.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:24, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:17, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<15:15, 634kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<15:02, 643kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<11:29, 841kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:17, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:05, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:27, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<34:13, 281kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<28:14, 340kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<20:41, 464kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<14:46, 648kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<10:31, 907kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<13:25, 710kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<13:39, 697kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<10:37, 896kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:41, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:34, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:38, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<09:33, 990kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:43, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:40, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:09, 2.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<09:37, 976kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:57, 858kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<08:42, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<06:21, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:38, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<11:33, 807kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<12:12, 763kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<09:34, 973kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:59, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:54, 934kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<11:01, 839kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<08:44, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:22, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:38, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<09:47, 938kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:54, 842kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<08:39, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<06:20, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<04:37, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<10:12, 893kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<11:10, 815kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<08:49, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<06:28, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<04:42, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<10:00, 904kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<11:00, 822kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:41, 1.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:22, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<04:38, 1.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<09:44, 922kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<10:47, 833kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:29, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:11, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<04:28, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<14:43, 605kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<14:14, 625kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:54, 816kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:51, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<05:39, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<08:50, 1.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<10:06, 874kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:00, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:52, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<04:16, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<13:03, 672kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<12:41, 691kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<09:38, 909kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<06:59, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<05:03, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<13:27, 647kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<12:40, 686kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<09:38, 902kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<06:59, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<05:01, 1.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<09:59, 864kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<10:29, 823kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:03, 1.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:51, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<04:13, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<26:27, 324kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<21:45, 394kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<16:00, 534kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<11:24, 748kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<08:07, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<55:35, 153kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<42:04, 202kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<30:09, 282kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<21:17, 398kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<15:00, 562kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<17:17, 487kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<15:03, 560kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<11:10, 753kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<08:01, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<05:44, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<14:12, 588kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<12:40, 660kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<09:34, 872kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:51, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:49, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<07:29, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:55, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:17, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:08, 2.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<27:20, 301kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<21:39, 379kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<15:41, 523kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<11:07, 736kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<07:55, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<12:22, 659kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<11:18, 721kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:28, 961kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:04, 1.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<16:48, 481kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<13:56, 579kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<10:19, 782kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<07:20, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<07:34, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:28, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:45, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:10, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:37, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:52, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:36, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:20, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:16, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:36, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:20, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:09, 2.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:37, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:14, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:10, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:01, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:17, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:48, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:29, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:16, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<05:10, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<05:41, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:30, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:17, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:00, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:02, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:51, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:46, 2.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<09:28, 795kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<08:06, 929kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:58, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<07:52, 946kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<07:14, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:54, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:24, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<05:25, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:12, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:01, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<08:34, 854kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<07:16, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:23, 1.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<05:10, 1.40MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:58, 1.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:51, 2.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<06:15, 1.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:20, 1.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:01, 1.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:53, 2.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<07:00, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<07:47, 913kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:12, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:29, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:40, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:41, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:38, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:37, 2.66MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<51:43, 135kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<37:36, 185kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<26:35, 262kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<18:37, 372kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<16:07, 429kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<12:14, 564kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<08:45, 786kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<06:09, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<31:30, 217kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<23:19, 293kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<16:34, 412kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<11:36, 584kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<14:34, 464kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<11:28, 589kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<08:16, 816kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:52, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:57, 962kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:07, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:35, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:17, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:15, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:16, 2.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:22, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:35, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:49, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:02, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:21, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:38, 2.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:54, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:12, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:32, 2.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:49, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:58, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:18, 2.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<01:48, 3.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<04:21, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:37, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:40, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:23, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:33, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:46, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:57, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:13, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:32, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:25, 2.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<01:45, 3.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<05:07, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:41, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:27, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:53, 2.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:07, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:24, 2.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<01:44, 3.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<09:24, 617kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:40, 757kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:37, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:51, 1.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:29, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:23, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:18, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:23, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:37, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:45, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<04:01, 1.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:20, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:27, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<03:05, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:12, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:27, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<02:37, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:51, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:13, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:27, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:44, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:09, 2.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:23, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:39, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:58, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:12, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:49, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:57, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:18, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:27, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:41, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:07, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:19, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:34, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:54, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:07, 2.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:48, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:09, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:34, 3.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:05, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:05, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:21, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:42, 2.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:51, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:37, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:45, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:43, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:48, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:09, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:33, 3.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:14, 911kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:27, 873kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:12, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:02, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:16, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:10, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:26, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:27, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:36, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:01, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:10, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:22, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:50, 2.46MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:20, 3.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:47, 939kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:11, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:08, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:54, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:51, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:10, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:33, 2.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:00, 1.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:27, 978kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:27, 1.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:29, 1.74MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:44, 1.56MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:43, 1.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:06, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:09, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:18, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:46, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:17, 3.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:41, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:21, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:32, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:26, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:28, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:54, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:59, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:02, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:36, 2.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:10, 3.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:51, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:28, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:44, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:59, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:16, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:19, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:48, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:53, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:02, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:35, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:43, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:54, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:30, 2.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:39, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:27, 2.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:36, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:48, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:25, 2.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:01, 3.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<1:43:58, 34.0kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<1:13:20, 48.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<51:18, 68.6kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<35:58, 96.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<25:46, 134kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<18:07, 190kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<13:04, 260kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<10:25, 326kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<07:35, 447kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<05:20, 630kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<04:31, 736kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:47, 876kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:48, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:28, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:20, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:47, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:45, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:48, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:23, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:00, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:11, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:05, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:36, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:37, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:42, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:19, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:24, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:32, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:11, 2.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<00:51, 3.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:32, 1.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:19, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:44, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:14, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:02, 1.39MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:30, 1.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:59, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:26, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:40, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:40, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:17, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:55, 2.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:02, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:55, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:28, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:27, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:30, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:10, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:14, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:20, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:02, 2.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:45, 3.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:40, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:38, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:15, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:16, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:20, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:03, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:06, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:40, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:21, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:58, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:14, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:18, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:00, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:03, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:09, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:54, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:59, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:05, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:50, 2.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:36, 3.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:37, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:31, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:08, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:49, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:24, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:22, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:01, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:43, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:00, 971kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:46, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:19, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:12, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:33, 1.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:14, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:53, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:01, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:03, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:49, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:50, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:55, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:42, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:30, 3.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:41, 990kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:29, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:07, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:01, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:58, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:45, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:32, 2.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:34, 980kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:40, 917kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:18, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:56, 1.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:00, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:58, 1.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:44, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:36, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:37, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:41, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:31, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:23, 3.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:42, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:43, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:33, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:34, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:37, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:29, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:30, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:34, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:26, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:28, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:31, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:24, 2.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:26, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:29, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:23, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:24, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:21, 2.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:14, 3.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:59, 856kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:51, 993kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:37, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:32, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:31, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:23, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:16, 2.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:33, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:31, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:23, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:21, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:22, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:17, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:14, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:15, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:11, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<01:51, 165kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<01:20, 226kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:54, 318kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:33, 422kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:25, 542kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:17, 747kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:11, 906kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.75MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 1.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.67MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<89:24:54,  1.24it/s]  0%|          | 654/400000 [00:00<62:29:36,  1.78it/s]  0%|          | 1314/400000 [00:01<43:40:41,  2.54it/s]  0%|          | 1980/400000 [00:01<30:31:43,  3.62it/s]  1%|          | 2690/400000 [00:01<21:20:11,  5.17it/s]  1%|          | 3317/400000 [00:01<14:55:02,  7.39it/s]  1%|          | 4003/400000 [00:01<10:25:43, 10.55it/s]  1%|          | 4742/400000 [00:01<7:17:27, 15.06it/s]   1%|▏         | 5498/400000 [00:01<5:05:53, 21.49it/s]  2%|▏         | 6250/400000 [00:01<3:33:59, 30.67it/s]  2%|▏         | 6981/400000 [00:01<2:29:46, 43.73it/s]  2%|▏         | 7687/400000 [00:01<1:44:56, 62.31it/s]  2%|▏         | 8423/400000 [00:02<1:13:35, 88.69it/s]  2%|▏         | 9173/400000 [00:02<51:40, 126.06it/s]   2%|▏         | 9929/400000 [00:02<36:21, 178.81it/s]  3%|▎         | 10664/400000 [00:02<25:40, 252.68it/s]  3%|▎         | 11388/400000 [00:02<18:13, 355.46it/s]  3%|▎         | 12148/400000 [00:02<12:59, 497.82it/s]  3%|▎         | 12878/400000 [00:02<09:21, 688.94it/s]  3%|▎         | 13587/400000 [00:02<06:51, 939.98it/s]  4%|▎         | 14272/400000 [00:02<05:05, 1263.22it/s]  4%|▎         | 14947/400000 [00:02<03:50, 1670.47it/s]  4%|▍         | 15623/400000 [00:03<02:58, 2157.83it/s]  4%|▍         | 16321/400000 [00:03<02:20, 2721.86it/s]  4%|▍         | 17076/400000 [00:03<01:53, 3367.96it/s]  4%|▍         | 17810/400000 [00:03<01:35, 4020.51it/s]  5%|▍         | 18548/400000 [00:03<01:21, 4656.04it/s]  5%|▍         | 19268/400000 [00:03<01:14, 5079.88it/s]  5%|▌         | 20027/400000 [00:03<01:07, 5639.20it/s]  5%|▌         | 20769/400000 [00:03<01:02, 6075.03it/s]  5%|▌         | 21493/400000 [00:03<01:00, 6262.84it/s]  6%|▌         | 22202/400000 [00:03<00:58, 6420.48it/s]  6%|▌         | 22903/400000 [00:04<00:58, 6457.19it/s]  6%|▌         | 23609/400000 [00:04<00:56, 6626.70it/s]  6%|▌         | 24353/400000 [00:04<00:54, 6850.03it/s]  6%|▋         | 25081/400000 [00:04<00:53, 6973.31it/s]  6%|▋         | 25839/400000 [00:04<00:52, 7142.94it/s]  7%|▋         | 26579/400000 [00:04<00:51, 7215.99it/s]  7%|▋         | 27310/400000 [00:04<00:51, 7171.86it/s]  7%|▋         | 28048/400000 [00:04<00:51, 7232.96it/s]  7%|▋         | 28776/400000 [00:04<00:51, 7141.11it/s]  7%|▋         | 29497/400000 [00:04<00:51, 7158.46it/s]  8%|▊         | 30239/400000 [00:05<00:51, 7234.34it/s]  8%|▊         | 30968/400000 [00:05<00:50, 7249.68it/s]  8%|▊         | 31717/400000 [00:05<00:50, 7318.82it/s]  8%|▊         | 32450/400000 [00:05<00:50, 7293.72it/s]  8%|▊         | 33201/400000 [00:05<00:49, 7355.38it/s]  8%|▊         | 33967/400000 [00:05<00:49, 7442.45it/s]  9%|▊         | 34719/400000 [00:05<00:48, 7464.59it/s]  9%|▉         | 35475/400000 [00:05<00:48, 7492.45it/s]  9%|▉         | 36225/400000 [00:05<00:49, 7418.75it/s]  9%|▉         | 36968/400000 [00:05<00:49, 7402.30it/s]  9%|▉         | 37727/400000 [00:06<00:48, 7457.08it/s] 10%|▉         | 38474/400000 [00:06<00:48, 7453.99it/s] 10%|▉         | 39233/400000 [00:06<00:48, 7492.16it/s] 10%|▉         | 39987/400000 [00:06<00:47, 7503.87it/s] 10%|█         | 40738/400000 [00:06<00:48, 7466.72it/s] 10%|█         | 41485/400000 [00:06<00:48, 7342.32it/s] 11%|█         | 42220/400000 [00:06<00:49, 7291.27it/s] 11%|█         | 42950/400000 [00:06<00:48, 7289.28it/s] 11%|█         | 43711/400000 [00:06<00:48, 7381.86it/s] 11%|█         | 44450/400000 [00:07<00:49, 7134.01it/s] 11%|█▏        | 45166/400000 [00:07<00:50, 7000.58it/s] 11%|█▏        | 45869/400000 [00:07<00:51, 6897.30it/s] 12%|█▏        | 46588/400000 [00:07<00:50, 6981.73it/s] 12%|█▏        | 47347/400000 [00:07<00:49, 7153.36it/s] 12%|█▏        | 48094/400000 [00:07<00:48, 7243.13it/s] 12%|█▏        | 48851/400000 [00:07<00:47, 7335.72it/s] 12%|█▏        | 49614/400000 [00:07<00:47, 7419.92it/s] 13%|█▎        | 50358/400000 [00:07<00:47, 7354.64it/s] 13%|█▎        | 51095/400000 [00:07<00:47, 7350.08it/s] 13%|█▎        | 51831/400000 [00:08<00:47, 7315.07it/s] 13%|█▎        | 52597/400000 [00:08<00:46, 7413.20it/s] 13%|█▎        | 53340/400000 [00:08<00:46, 7386.19it/s] 14%|█▎        | 54090/400000 [00:08<00:46, 7418.14it/s] 14%|█▎        | 54850/400000 [00:08<00:46, 7470.88it/s] 14%|█▍        | 55598/400000 [00:08<00:46, 7391.77it/s] 14%|█▍        | 56354/400000 [00:08<00:46, 7439.85it/s] 14%|█▍        | 57104/400000 [00:08<00:45, 7455.11it/s] 14%|█▍        | 57850/400000 [00:08<00:46, 7390.64it/s] 15%|█▍        | 58590/400000 [00:08<00:46, 7323.27it/s] 15%|█▍        | 59323/400000 [00:09<00:46, 7267.53it/s] 15%|█▌        | 60051/400000 [00:09<00:48, 7061.13it/s] 15%|█▌        | 60759/400000 [00:09<00:48, 6970.53it/s] 15%|█▌        | 61458/400000 [00:09<00:48, 6924.76it/s] 16%|█▌        | 62152/400000 [00:09<00:49, 6886.09it/s] 16%|█▌        | 62889/400000 [00:09<00:48, 7022.57it/s] 16%|█▌        | 63645/400000 [00:09<00:46, 7174.22it/s] 16%|█▌        | 64397/400000 [00:09<00:46, 7273.70it/s] 16%|█▋        | 65126/400000 [00:09<00:46, 7262.56it/s] 16%|█▋        | 65854/400000 [00:09<00:47, 6989.45it/s] 17%|█▋        | 66556/400000 [00:10<00:49, 6771.31it/s] 17%|█▋        | 67237/400000 [00:10<00:49, 6740.24it/s] 17%|█▋        | 67983/400000 [00:10<00:47, 6940.46it/s] 17%|█▋        | 68681/400000 [00:10<00:48, 6891.04it/s] 17%|█▋        | 69373/400000 [00:10<00:48, 6798.30it/s] 18%|█▊        | 70071/400000 [00:10<00:48, 6850.10it/s] 18%|█▊        | 70804/400000 [00:10<00:47, 6984.93it/s] 18%|█▊        | 71539/400000 [00:10<00:46, 7089.03it/s] 18%|█▊        | 72304/400000 [00:10<00:45, 7247.25it/s] 18%|█▊        | 73031/400000 [00:11<00:45, 7244.19it/s] 18%|█▊        | 73776/400000 [00:11<00:44, 7302.93it/s] 19%|█▊        | 74529/400000 [00:11<00:44, 7368.93it/s] 19%|█▉        | 75283/400000 [00:11<00:43, 7416.98it/s] 19%|█▉        | 76026/400000 [00:11<00:44, 7302.68it/s] 19%|█▉        | 76763/400000 [00:11<00:44, 7322.33it/s] 19%|█▉        | 77505/400000 [00:11<00:43, 7350.90it/s] 20%|█▉        | 78257/400000 [00:11<00:43, 7398.74it/s] 20%|█▉        | 78998/400000 [00:11<00:43, 7332.43it/s] 20%|█▉        | 79744/400000 [00:11<00:43, 7367.92it/s] 20%|██        | 80482/400000 [00:12<00:43, 7312.94it/s] 20%|██        | 81227/400000 [00:12<00:43, 7350.99it/s] 20%|██        | 81963/400000 [00:12<00:43, 7352.12it/s] 21%|██        | 82699/400000 [00:12<00:43, 7335.03it/s] 21%|██        | 83453/400000 [00:12<00:42, 7392.75it/s] 21%|██        | 84193/400000 [00:12<00:42, 7371.39it/s] 21%|██        | 84934/400000 [00:12<00:42, 7380.47it/s] 21%|██▏       | 85673/400000 [00:12<00:42, 7352.27it/s] 22%|██▏       | 86409/400000 [00:12<00:42, 7295.28it/s] 22%|██▏       | 87155/400000 [00:12<00:42, 7343.77it/s] 22%|██▏       | 87890/400000 [00:13<00:42, 7291.31it/s] 22%|██▏       | 88620/400000 [00:13<00:42, 7280.44it/s] 22%|██▏       | 89363/400000 [00:13<00:42, 7322.84it/s] 23%|██▎       | 90126/400000 [00:13<00:41, 7411.10it/s] 23%|██▎       | 90868/400000 [00:13<00:41, 7384.75it/s] 23%|██▎       | 91607/400000 [00:13<00:42, 7177.99it/s] 23%|██▎       | 92342/400000 [00:13<00:42, 7226.74it/s] 23%|██▎       | 93088/400000 [00:13<00:42, 7293.84it/s] 23%|██▎       | 93833/400000 [00:13<00:41, 7339.89it/s] 24%|██▎       | 94568/400000 [00:13<00:42, 7191.28it/s] 24%|██▍       | 95289/400000 [00:14<00:43, 6976.29it/s] 24%|██▍       | 96049/400000 [00:14<00:42, 7151.49it/s] 24%|██▍       | 96786/400000 [00:14<00:42, 7171.60it/s] 24%|██▍       | 97506/400000 [00:14<00:42, 7161.18it/s] 25%|██▍       | 98224/400000 [00:14<00:43, 6998.62it/s] 25%|██▍       | 98926/400000 [00:14<00:43, 6875.57it/s] 25%|██▍       | 99629/400000 [00:14<00:43, 6919.69it/s] 25%|██▌       | 100333/400000 [00:14<00:43, 6954.39it/s] 25%|██▌       | 101078/400000 [00:14<00:42, 7094.32it/s] 25%|██▌       | 101818/400000 [00:14<00:41, 7182.33it/s] 26%|██▌       | 102553/400000 [00:15<00:41, 7231.38it/s] 26%|██▌       | 103303/400000 [00:15<00:40, 7308.16it/s] 26%|██▌       | 104045/400000 [00:15<00:40, 7339.20it/s] 26%|██▌       | 104781/400000 [00:15<00:40, 7342.64it/s] 26%|██▋       | 105538/400000 [00:15<00:39, 7407.77it/s] 27%|██▋       | 106280/400000 [00:15<00:39, 7407.67it/s] 27%|██▋       | 107022/400000 [00:15<00:39, 7363.90it/s] 27%|██▋       | 107759/400000 [00:15<00:40, 7288.59it/s] 27%|██▋       | 108506/400000 [00:15<00:39, 7340.09it/s] 27%|██▋       | 109241/400000 [00:15<00:39, 7310.67it/s] 27%|██▋       | 109973/400000 [00:16<00:40, 7243.96it/s] 28%|██▊       | 110714/400000 [00:16<00:39, 7291.14it/s] 28%|██▊       | 111476/400000 [00:16<00:39, 7386.24it/s] 28%|██▊       | 112228/400000 [00:16<00:38, 7423.91it/s] 28%|██▊       | 112992/400000 [00:16<00:38, 7484.85it/s] 28%|██▊       | 113741/400000 [00:16<00:38, 7464.26it/s] 29%|██▊       | 114488/400000 [00:16<00:38, 7360.41it/s] 29%|██▉       | 115225/400000 [00:16<00:38, 7303.66it/s] 29%|██▉       | 115980/400000 [00:16<00:38, 7374.02it/s] 29%|██▉       | 116735/400000 [00:16<00:38, 7425.03it/s] 29%|██▉       | 117482/400000 [00:17<00:37, 7437.51it/s] 30%|██▉       | 118235/400000 [00:17<00:37, 7463.78it/s] 30%|██▉       | 119004/400000 [00:17<00:37, 7529.89it/s] 30%|██▉       | 119767/400000 [00:17<00:37, 7558.32it/s] 30%|███       | 120524/400000 [00:17<00:37, 7463.13it/s] 30%|███       | 121271/400000 [00:17<00:38, 7245.45it/s] 31%|███       | 122026/400000 [00:17<00:37, 7332.76it/s] 31%|███       | 122761/400000 [00:17<00:38, 7286.13it/s] 31%|███       | 123521/400000 [00:17<00:37, 7375.43it/s] 31%|███       | 124277/400000 [00:18<00:37, 7427.13it/s] 31%|███▏      | 125021/400000 [00:18<00:37, 7392.64it/s] 31%|███▏      | 125761/400000 [00:18<00:37, 7279.64it/s] 32%|███▏      | 126490/400000 [00:18<00:38, 7129.51it/s] 32%|███▏      | 127205/400000 [00:18<00:38, 7019.34it/s] 32%|███▏      | 127909/400000 [00:18<00:39, 6971.62it/s] 32%|███▏      | 128638/400000 [00:18<00:38, 7061.80it/s] 32%|███▏      | 129346/400000 [00:18<00:38, 6971.28it/s] 33%|███▎      | 130045/400000 [00:18<00:39, 6860.72it/s] 33%|███▎      | 130798/400000 [00:18<00:38, 7046.70it/s] 33%|███▎      | 131550/400000 [00:19<00:37, 7180.00it/s] 33%|███▎      | 132288/400000 [00:19<00:36, 7238.24it/s] 33%|███▎      | 133014/400000 [00:19<00:37, 7146.57it/s] 33%|███▎      | 133730/400000 [00:19<00:37, 7038.42it/s] 34%|███▎      | 134436/400000 [00:19<00:38, 6979.49it/s] 34%|███▍      | 135135/400000 [00:19<00:38, 6938.28it/s] 34%|███▍      | 135830/400000 [00:19<00:38, 6868.71it/s] 34%|███▍      | 136553/400000 [00:19<00:37, 6971.31it/s] 34%|███▍      | 137308/400000 [00:19<00:36, 7134.96it/s] 35%|███▍      | 138044/400000 [00:19<00:36, 7198.37it/s] 35%|███▍      | 138787/400000 [00:20<00:35, 7263.86it/s] 35%|███▍      | 139515/400000 [00:20<00:35, 7263.85it/s] 35%|███▌      | 140243/400000 [00:20<00:36, 7079.74it/s] 35%|███▌      | 140953/400000 [00:20<00:37, 6959.60it/s] 35%|███▌      | 141717/400000 [00:20<00:36, 7150.21it/s] 36%|███▌      | 142479/400000 [00:20<00:35, 7283.67it/s] 36%|███▌      | 143210/400000 [00:20<00:35, 7277.84it/s] 36%|███▌      | 143940/400000 [00:20<00:35, 7283.49it/s] 36%|███▌      | 144670/400000 [00:20<00:35, 7150.23it/s] 36%|███▋      | 145387/400000 [00:20<00:36, 7048.61it/s] 37%|███▋      | 146124/400000 [00:21<00:35, 7140.56it/s] 37%|███▋      | 146865/400000 [00:21<00:35, 7217.33it/s] 37%|███▋      | 147632/400000 [00:21<00:34, 7345.21it/s] 37%|███▋      | 148405/400000 [00:21<00:33, 7456.07it/s] 37%|███▋      | 149168/400000 [00:21<00:33, 7505.27it/s] 37%|███▋      | 149941/400000 [00:21<00:33, 7571.21it/s] 38%|███▊      | 150699/400000 [00:21<00:33, 7365.31it/s] 38%|███▊      | 151438/400000 [00:21<00:35, 7099.41it/s] 38%|███▊      | 152152/400000 [00:21<00:35, 7039.54it/s] 38%|███▊      | 152880/400000 [00:22<00:34, 7108.58it/s] 38%|███▊      | 153593/400000 [00:22<00:35, 6974.61it/s] 39%|███▊      | 154293/400000 [00:22<00:35, 6834.37it/s] 39%|███▉      | 155037/400000 [00:22<00:34, 7002.95it/s] 39%|███▉      | 155794/400000 [00:22<00:34, 7161.72it/s] 39%|███▉      | 156558/400000 [00:22<00:33, 7298.40it/s] 39%|███▉      | 157304/400000 [00:22<00:33, 7345.39it/s] 40%|███▉      | 158048/400000 [00:22<00:32, 7371.30it/s] 40%|███▉      | 158794/400000 [00:22<00:32, 7396.36it/s] 40%|███▉      | 159557/400000 [00:22<00:32, 7463.07it/s] 40%|████      | 160305/400000 [00:23<00:32, 7379.54it/s] 40%|████      | 161070/400000 [00:23<00:32, 7456.32it/s] 40%|████      | 161817/400000 [00:23<00:32, 7219.68it/s] 41%|████      | 162542/400000 [00:23<00:33, 7095.73it/s] 41%|████      | 163254/400000 [00:23<00:33, 7034.99it/s] 41%|████      | 163959/400000 [00:23<00:33, 7016.28it/s] 41%|████      | 164711/400000 [00:23<00:32, 7158.40it/s] 41%|████▏     | 165433/400000 [00:23<00:32, 7175.18it/s] 42%|████▏     | 166176/400000 [00:23<00:32, 7246.56it/s] 42%|████▏     | 166928/400000 [00:23<00:31, 7322.81it/s] 42%|████▏     | 167681/400000 [00:24<00:31, 7382.73it/s] 42%|████▏     | 168420/400000 [00:24<00:31, 7331.53it/s] 42%|████▏     | 169154/400000 [00:24<00:31, 7240.12it/s] 42%|████▏     | 169912/400000 [00:24<00:31, 7337.63it/s] 43%|████▎     | 170660/400000 [00:24<00:31, 7377.24it/s] 43%|████▎     | 171432/400000 [00:24<00:30, 7475.29it/s] 43%|████▎     | 172191/400000 [00:24<00:30, 7506.48it/s] 43%|████▎     | 172951/400000 [00:24<00:30, 7534.20it/s] 43%|████▎     | 173705/400000 [00:24<00:30, 7508.18it/s] 44%|████▎     | 174473/400000 [00:24<00:29, 7556.25it/s] 44%|████▍     | 175235/400000 [00:25<00:29, 7573.97it/s] 44%|████▍     | 175995/400000 [00:25<00:29, 7579.57it/s] 44%|████▍     | 176754/400000 [00:25<00:29, 7521.54it/s] 44%|████▍     | 177513/400000 [00:25<00:29, 7540.37it/s] 45%|████▍     | 178268/400000 [00:25<00:29, 7445.56it/s] 45%|████▍     | 179037/400000 [00:25<00:29, 7515.87it/s] 45%|████▍     | 179798/400000 [00:25<00:29, 7542.29it/s] 45%|████▌     | 180553/400000 [00:25<00:29, 7458.33it/s] 45%|████▌     | 181314/400000 [00:25<00:29, 7501.19it/s] 46%|████▌     | 182065/400000 [00:25<00:29, 7420.55it/s] 46%|████▌     | 182808/400000 [00:26<00:30, 7203.96it/s] 46%|████▌     | 183531/400000 [00:26<00:31, 6960.15it/s] 46%|████▌     | 184230/400000 [00:26<00:31, 6787.30it/s] 46%|████▌     | 184916/400000 [00:26<00:31, 6808.55it/s] 46%|████▋     | 185599/400000 [00:26<00:31, 6733.00it/s] 47%|████▋     | 186302/400000 [00:26<00:31, 6816.95it/s] 47%|████▋     | 187049/400000 [00:26<00:30, 6999.27it/s] 47%|████▋     | 187813/400000 [00:26<00:29, 7179.73it/s] 47%|████▋     | 188570/400000 [00:26<00:28, 7291.16it/s] 47%|████▋     | 189302/400000 [00:27<00:29, 7060.60it/s] 48%|████▊     | 190012/400000 [00:27<00:30, 6870.68it/s] 48%|████▊     | 190703/400000 [00:27<00:31, 6607.90it/s] 48%|████▊     | 191369/400000 [00:27<00:32, 6489.20it/s] 48%|████▊     | 192022/400000 [00:27<00:32, 6397.06it/s] 48%|████▊     | 192674/400000 [00:27<00:32, 6430.61it/s] 48%|████▊     | 193355/400000 [00:27<00:31, 6538.58it/s] 49%|████▊     | 194024/400000 [00:27<00:31, 6580.92it/s] 49%|████▊     | 194707/400000 [00:27<00:30, 6651.57it/s] 49%|████▉     | 195374/400000 [00:27<00:30, 6605.24it/s] 49%|████▉     | 196095/400000 [00:28<00:30, 6773.72it/s] 49%|████▉     | 196846/400000 [00:28<00:29, 6976.58it/s] 49%|████▉     | 197592/400000 [00:28<00:28, 7114.84it/s] 50%|████▉     | 198358/400000 [00:28<00:27, 7267.63it/s] 50%|████▉     | 199114/400000 [00:28<00:27, 7349.23it/s] 50%|████▉     | 199851/400000 [00:28<00:27, 7322.84it/s] 50%|█████     | 200606/400000 [00:28<00:26, 7389.05it/s] 50%|█████     | 201347/400000 [00:28<00:27, 7175.00it/s] 51%|█████     | 202067/400000 [00:28<00:28, 7047.34it/s] 51%|█████     | 202784/400000 [00:28<00:27, 7083.37it/s] 51%|█████     | 203529/400000 [00:29<00:27, 7188.48it/s] 51%|█████     | 204279/400000 [00:29<00:26, 7278.93it/s] 51%|█████▏    | 205009/400000 [00:29<00:27, 7197.56it/s] 51%|█████▏    | 205730/400000 [00:29<00:27, 7042.90it/s] 52%|█████▏    | 206488/400000 [00:29<00:26, 7194.51it/s] 52%|█████▏    | 207210/400000 [00:29<00:27, 7090.16it/s] 52%|█████▏    | 207921/400000 [00:29<00:27, 7024.78it/s] 52%|█████▏    | 208625/400000 [00:29<00:27, 6936.73it/s] 52%|█████▏    | 209320/400000 [00:29<00:28, 6799.00it/s] 53%|█████▎    | 210024/400000 [00:30<00:27, 6867.12it/s] 53%|█████▎    | 210754/400000 [00:30<00:27, 6989.09it/s] 53%|█████▎    | 211524/400000 [00:30<00:26, 7185.90it/s] 53%|█████▎    | 212279/400000 [00:30<00:25, 7290.49it/s] 53%|█████▎    | 213041/400000 [00:30<00:25, 7385.30it/s] 53%|█████▎    | 213807/400000 [00:30<00:24, 7465.47it/s] 54%|█████▎    | 214555/400000 [00:30<00:24, 7441.58it/s] 54%|█████▍    | 215301/400000 [00:30<00:25, 7235.50it/s] 54%|█████▍    | 216027/400000 [00:30<00:26, 7005.29it/s] 54%|█████▍    | 216731/400000 [00:30<00:26, 6845.42it/s] 54%|█████▍    | 217419/400000 [00:31<00:26, 6793.66it/s] 55%|█████▍    | 218101/400000 [00:31<00:26, 6795.40it/s] 55%|█████▍    | 218783/400000 [00:31<00:26, 6795.84it/s] 55%|█████▍    | 219464/400000 [00:31<00:26, 6727.49it/s] 55%|█████▌    | 220143/400000 [00:31<00:26, 6745.61it/s] 55%|█████▌    | 220819/400000 [00:31<00:26, 6743.36it/s] 55%|█████▌    | 221540/400000 [00:31<00:25, 6876.71it/s] 56%|█████▌    | 222303/400000 [00:31<00:25, 7084.78it/s] 56%|█████▌    | 223043/400000 [00:31<00:24, 7175.74it/s] 56%|█████▌    | 223801/400000 [00:31<00:24, 7291.17it/s] 56%|█████▌    | 224567/400000 [00:32<00:23, 7395.95it/s] 56%|█████▋    | 225313/400000 [00:32<00:23, 7414.98it/s] 57%|█████▋    | 226056/400000 [00:32<00:23, 7366.90it/s] 57%|█████▋    | 226794/400000 [00:32<00:24, 7182.39it/s] 57%|█████▋    | 227539/400000 [00:32<00:23, 7259.85it/s] 57%|█████▋    | 228267/400000 [00:32<00:23, 7206.80it/s] 57%|█████▋    | 228989/400000 [00:32<00:24, 6841.16it/s] 57%|█████▋    | 229699/400000 [00:32<00:24, 6914.58it/s] 58%|█████▊    | 230394/400000 [00:32<00:24, 6905.20it/s] 58%|█████▊    | 231143/400000 [00:32<00:23, 7068.76it/s] 58%|█████▊    | 231899/400000 [00:33<00:23, 7207.25it/s] 58%|█████▊    | 232659/400000 [00:33<00:22, 7318.45it/s] 58%|█████▊    | 233411/400000 [00:33<00:22, 7376.53it/s] 59%|█████▊    | 234151/400000 [00:33<00:22, 7309.97it/s] 59%|█████▊    | 234884/400000 [00:33<00:23, 7107.21it/s] 59%|█████▉    | 235597/400000 [00:33<00:23, 6983.24it/s] 59%|█████▉    | 236326/400000 [00:33<00:23, 7070.90it/s] 59%|█████▉    | 237058/400000 [00:33<00:22, 7143.08it/s] 59%|█████▉    | 237774/400000 [00:33<00:23, 7032.92it/s] 60%|█████▉    | 238533/400000 [00:34<00:22, 7189.61it/s] 60%|█████▉    | 239272/400000 [00:34<00:22, 7248.20it/s] 60%|██████    | 240019/400000 [00:34<00:21, 7312.31it/s] 60%|██████    | 240752/400000 [00:34<00:21, 7304.31it/s] 60%|██████    | 241484/400000 [00:34<00:21, 7274.10it/s] 61%|██████    | 242250/400000 [00:34<00:21, 7385.19it/s] 61%|██████    | 243012/400000 [00:34<00:21, 7452.23it/s] 61%|██████    | 243773/400000 [00:34<00:20, 7496.98it/s] 61%|██████    | 244524/400000 [00:34<00:21, 7335.20it/s] 61%|██████▏   | 245269/400000 [00:34<00:21, 7367.17it/s] 62%|██████▏   | 246033/400000 [00:35<00:20, 7446.90it/s] 62%|██████▏   | 246792/400000 [00:35<00:20, 7487.86it/s] 62%|██████▏   | 247550/400000 [00:35<00:20, 7514.06it/s] 62%|██████▏   | 248302/400000 [00:35<00:20, 7435.55it/s] 62%|██████▏   | 249047/400000 [00:35<00:20, 7401.59it/s] 62%|██████▏   | 249807/400000 [00:35<00:20, 7458.79it/s] 63%|██████▎   | 250554/400000 [00:35<00:20, 7407.27it/s] 63%|██████▎   | 251316/400000 [00:35<00:19, 7469.74it/s] 63%|██████▎   | 252064/400000 [00:35<00:19, 7443.34it/s] 63%|██████▎   | 252809/400000 [00:35<00:20, 7320.28it/s] 63%|██████▎   | 253569/400000 [00:36<00:19, 7400.05it/s] 64%|██████▎   | 254341/400000 [00:36<00:19, 7490.60it/s] 64%|██████▍   | 255091/400000 [00:36<00:19, 7468.68it/s] 64%|██████▍   | 255858/400000 [00:36<00:19, 7526.50it/s] 64%|██████▍   | 256612/400000 [00:36<00:19, 7316.81it/s] 64%|██████▍   | 257374/400000 [00:36<00:19, 7397.65it/s] 65%|██████▍   | 258141/400000 [00:36<00:18, 7476.27it/s] 65%|██████▍   | 258903/400000 [00:36<00:18, 7517.70it/s] 65%|██████▍   | 259665/400000 [00:36<00:18, 7546.58it/s] 65%|██████▌   | 260421/400000 [00:36<00:18, 7402.67it/s] 65%|██████▌   | 261163/400000 [00:37<00:18, 7324.34it/s] 65%|██████▌   | 261920/400000 [00:37<00:18, 7395.96it/s] 66%|██████▌   | 262681/400000 [00:37<00:18, 7457.28it/s] 66%|██████▌   | 263428/400000 [00:37<00:18, 7435.08it/s] 66%|██████▌   | 264173/400000 [00:37<00:18, 7409.05it/s] 66%|██████▌   | 264929/400000 [00:37<00:18, 7451.93it/s] 66%|██████▋   | 265685/400000 [00:37<00:17, 7483.32it/s] 67%|██████▋   | 266443/400000 [00:37<00:17, 7509.68it/s] 67%|██████▋   | 267195/400000 [00:37<00:17, 7488.92it/s] 67%|██████▋   | 267945/400000 [00:37<00:17, 7454.91it/s] 67%|██████▋   | 268709/400000 [00:38<00:17, 7509.26it/s] 67%|██████▋   | 269461/400000 [00:38<00:17, 7387.96it/s] 68%|██████▊   | 270201/400000 [00:38<00:18, 7187.77it/s] 68%|██████▊   | 270922/400000 [00:38<00:17, 7191.79it/s] 68%|██████▊   | 271664/400000 [00:38<00:17, 7258.06it/s] 68%|██████▊   | 272391/400000 [00:38<00:17, 7231.30it/s] 68%|██████▊   | 273149/400000 [00:38<00:17, 7330.28it/s] 68%|██████▊   | 273883/400000 [00:38<00:17, 7306.76it/s] 69%|██████▊   | 274615/400000 [00:38<00:17, 7162.09it/s] 69%|██████▉   | 275333/400000 [00:38<00:17, 7131.58it/s] 69%|██████▉   | 276047/400000 [00:39<00:17, 7045.63it/s] 69%|██████▉   | 276753/400000 [00:39<00:17, 6954.09it/s] 69%|██████▉   | 277450/400000 [00:39<00:17, 6833.60it/s] 70%|██████▉   | 278135/400000 [00:39<00:17, 6820.46it/s] 70%|██████▉   | 278830/400000 [00:39<00:17, 6857.23it/s] 70%|██████▉   | 279569/400000 [00:39<00:17, 7006.77it/s] 70%|███████   | 280330/400000 [00:39<00:16, 7174.78it/s] 70%|███████   | 281091/400000 [00:39<00:16, 7297.98it/s] 70%|███████   | 281849/400000 [00:39<00:16, 7362.45it/s] 71%|███████   | 282587/400000 [00:40<00:16, 7335.66it/s] 71%|███████   | 283328/400000 [00:40<00:15, 7356.12it/s] 71%|███████   | 284065/400000 [00:40<00:16, 7116.98it/s] 71%|███████   | 284779/400000 [00:40<00:16, 7013.08it/s] 71%|███████▏  | 285497/400000 [00:40<00:16, 7061.58it/s] 72%|███████▏  | 286226/400000 [00:40<00:15, 7126.55it/s] 72%|███████▏  | 286940/400000 [00:40<00:15, 7093.44it/s] 72%|███████▏  | 287651/400000 [00:40<00:16, 6992.63it/s] 72%|███████▏  | 288352/400000 [00:40<00:16, 6815.06it/s] 72%|███████▏  | 289104/400000 [00:40<00:15, 7010.51it/s] 72%|███████▏  | 289808/400000 [00:41<00:15, 6988.92it/s] 73%|███████▎  | 290549/400000 [00:41<00:15, 7109.25it/s] 73%|███████▎  | 291310/400000 [00:41<00:14, 7251.12it/s] 73%|███████▎  | 292051/400000 [00:41<00:14, 7297.67it/s] 73%|███████▎  | 292803/400000 [00:41<00:14, 7362.01it/s] 73%|███████▎  | 293558/400000 [00:41<00:14, 7417.08it/s] 74%|███████▎  | 294315/400000 [00:41<00:14, 7459.01it/s] 74%|███████▍  | 295068/400000 [00:41<00:14, 7477.21it/s] 74%|███████▍  | 295820/400000 [00:41<00:13, 7477.04it/s] 74%|███████▍  | 296569/400000 [00:41<00:13, 7476.88it/s] 74%|███████▍  | 297317/400000 [00:42<00:14, 7227.57it/s] 75%|███████▍  | 298059/400000 [00:42<00:13, 7283.63it/s] 75%|███████▍  | 298811/400000 [00:42<00:13, 7351.61it/s] 75%|███████▍  | 299568/400000 [00:42<00:13, 7414.89it/s] 75%|███████▌  | 300317/400000 [00:42<00:13, 7436.73it/s] 75%|███████▌  | 301062/400000 [00:42<00:13, 7275.99it/s] 75%|███████▌  | 301791/400000 [00:42<00:13, 7165.15it/s] 76%|███████▌  | 302542/400000 [00:42<00:13, 7263.64it/s] 76%|███████▌  | 303270/400000 [00:42<00:13, 7216.51it/s] 76%|███████▌  | 304030/400000 [00:42<00:13, 7325.72it/s] 76%|███████▌  | 304764/400000 [00:43<00:13, 7272.01it/s] 76%|███████▋  | 305513/400000 [00:43<00:12, 7335.68it/s] 77%|███████▋  | 306266/400000 [00:43<00:12, 7392.68it/s] 77%|███████▋  | 307023/400000 [00:43<00:12, 7444.82it/s] 77%|███████▋  | 307777/400000 [00:43<00:12, 7473.10it/s] 77%|███████▋  | 308525/400000 [00:43<00:12, 7423.79it/s] 77%|███████▋  | 309285/400000 [00:43<00:12, 7475.69it/s] 78%|███████▊  | 310052/400000 [00:43<00:11, 7532.30it/s] 78%|███████▊  | 310806/400000 [00:43<00:11, 7534.32it/s] 78%|███████▊  | 311560/400000 [00:43<00:11, 7510.88it/s] 78%|███████▊  | 312312/400000 [00:44<00:11, 7364.20it/s] 78%|███████▊  | 313050/400000 [00:44<00:12, 7162.69it/s] 78%|███████▊  | 313774/400000 [00:44<00:12, 7185.37it/s] 79%|███████▊  | 314533/400000 [00:44<00:11, 7300.57it/s] 79%|███████▉  | 315294/400000 [00:44<00:11, 7388.72it/s] 79%|███████▉  | 316035/400000 [00:44<00:11, 7359.45it/s] 79%|███████▉  | 316772/400000 [00:44<00:11, 7337.52it/s] 79%|███████▉  | 317507/400000 [00:44<00:11, 7248.49it/s] 80%|███████▉  | 318233/400000 [00:44<00:11, 7025.66it/s] 80%|███████▉  | 318938/400000 [00:45<00:11, 6967.96it/s] 80%|███████▉  | 319637/400000 [00:45<00:11, 6881.90it/s] 80%|████████  | 320327/400000 [00:45<00:11, 6857.90it/s] 80%|████████  | 321053/400000 [00:45<00:11, 6971.22it/s] 80%|████████  | 321815/400000 [00:45<00:10, 7153.35it/s] 81%|████████  | 322580/400000 [00:45<00:10, 7294.61it/s] 81%|████████  | 323337/400000 [00:45<00:10, 7373.40it/s] 81%|████████  | 324091/400000 [00:45<00:10, 7420.18it/s] 81%|████████  | 324835/400000 [00:45<00:10, 7141.87it/s] 81%|████████▏ | 325586/400000 [00:45<00:10, 7248.32it/s] 82%|████████▏ | 326347/400000 [00:46<00:10, 7350.85it/s] 82%|████████▏ | 327085/400000 [00:46<00:10, 7123.04it/s] 82%|████████▏ | 327801/400000 [00:46<00:10, 7026.57it/s] 82%|████████▏ | 328507/400000 [00:46<00:10, 6969.12it/s] 82%|████████▏ | 329206/400000 [00:46<00:10, 6859.28it/s] 82%|████████▏ | 329948/400000 [00:46<00:09, 7017.85it/s] 83%|████████▎ | 330699/400000 [00:46<00:09, 7157.44it/s] 83%|████████▎ | 331467/400000 [00:46<00:09, 7304.64it/s] 83%|████████▎ | 332232/400000 [00:46<00:09, 7402.46it/s] 83%|████████▎ | 332998/400000 [00:46<00:08, 7476.55it/s] 83%|████████▎ | 333767/400000 [00:47<00:08, 7538.87it/s] 84%|████████▎ | 334523/400000 [00:47<00:08, 7378.70it/s] 84%|████████▍ | 335263/400000 [00:47<00:09, 7190.12it/s] 84%|████████▍ | 335985/400000 [00:47<00:09, 7004.63it/s] 84%|████████▍ | 336689/400000 [00:47<00:09, 6908.76it/s] 84%|████████▍ | 337382/400000 [00:47<00:09, 6715.04it/s] 85%|████████▍ | 338057/400000 [00:47<00:09, 6651.66it/s] 85%|████████▍ | 338725/400000 [00:47<00:09, 6587.53it/s] 85%|████████▍ | 339448/400000 [00:47<00:08, 6767.20it/s] 85%|████████▌ | 340132/400000 [00:48<00:08, 6788.23it/s] 85%|████████▌ | 340813/400000 [00:48<00:08, 6729.03it/s] 85%|████████▌ | 341488/400000 [00:48<00:08, 6728.46it/s] 86%|████████▌ | 342231/400000 [00:48<00:08, 6923.97it/s] 86%|████████▌ | 342991/400000 [00:48<00:08, 7111.69it/s] 86%|████████▌ | 343757/400000 [00:48<00:07, 7266.32it/s] 86%|████████▌ | 344487/400000 [00:48<00:07, 7262.13it/s] 86%|████████▋ | 345216/400000 [00:48<00:07, 7242.96it/s] 86%|████████▋ | 345942/400000 [00:48<00:07, 7058.27it/s] 87%|████████▋ | 346650/400000 [00:48<00:07, 6740.33it/s] 87%|████████▋ | 347415/400000 [00:49<00:07, 6987.29it/s] 87%|████████▋ | 348155/400000 [00:49<00:07, 7104.40it/s] 87%|████████▋ | 348903/400000 [00:49<00:07, 7212.13it/s] 87%|████████▋ | 349628/400000 [00:49<00:07, 7119.11it/s] 88%|████████▊ | 350343/400000 [00:49<00:07, 7018.56it/s] 88%|████████▊ | 351048/400000 [00:49<00:07, 6773.99it/s] 88%|████████▊ | 351729/400000 [00:49<00:07, 6687.48it/s] 88%|████████▊ | 352471/400000 [00:49<00:06, 6889.60it/s] 88%|████████▊ | 353213/400000 [00:49<00:06, 7038.81it/s] 88%|████████▊ | 353921/400000 [00:49<00:06, 6934.50it/s] 89%|████████▊ | 354617/400000 [00:50<00:06, 6887.58it/s] 89%|████████▉ | 355314/400000 [00:50<00:06, 6909.43it/s] 89%|████████▉ | 356063/400000 [00:50<00:06, 7071.80it/s] 89%|████████▉ | 356811/400000 [00:50<00:06, 7187.75it/s] 89%|████████▉ | 357532/400000 [00:50<00:05, 7157.44it/s] 90%|████████▉ | 358274/400000 [00:50<00:05, 7232.19it/s] 90%|████████▉ | 359001/400000 [00:50<00:05, 7242.60it/s] 90%|████████▉ | 359761/400000 [00:50<00:05, 7344.55it/s] 90%|█████████ | 360526/400000 [00:50<00:05, 7430.11it/s] 90%|█████████ | 361281/400000 [00:50<00:05, 7464.77it/s] 91%|█████████ | 362042/400000 [00:51<00:05, 7504.82it/s] 91%|█████████ | 362793/400000 [00:51<00:04, 7445.64it/s] 91%|█████████ | 363539/400000 [00:51<00:05, 7198.61it/s] 91%|█████████ | 364261/400000 [00:51<00:05, 6914.75it/s] 91%|█████████ | 364957/400000 [00:51<00:05, 6811.07it/s] 91%|█████████▏| 365642/400000 [00:51<00:05, 6763.73it/s] 92%|█████████▏| 366321/400000 [00:51<00:05, 6701.17it/s] 92%|█████████▏| 367069/400000 [00:51<00:04, 6916.56it/s] 92%|█████████▏| 367811/400000 [00:51<00:04, 7059.76it/s] 92%|█████████▏| 368520/400000 [00:52<00:04, 7010.98it/s] 92%|█████████▏| 369224/400000 [00:52<00:04, 6962.59it/s] 92%|█████████▏| 369937/400000 [00:52<00:04, 7010.92it/s] 93%|█████████▎| 370709/400000 [00:52<00:04, 7208.72it/s] 93%|█████████▎| 371448/400000 [00:52<00:03, 7260.20it/s] 93%|█████████▎| 372204/400000 [00:52<00:03, 7346.43it/s] 93%|█████████▎| 372947/400000 [00:52<00:03, 7369.19it/s] 93%|█████████▎| 373687/400000 [00:52<00:03, 7377.15it/s] 94%|█████████▎| 374426/400000 [00:52<00:03, 7373.50it/s] 94%|█████████▍| 375171/400000 [00:52<00:03, 7394.02it/s] 94%|█████████▍| 375937/400000 [00:53<00:03, 7470.75it/s] 94%|█████████▍| 376694/400000 [00:53<00:03, 7498.08it/s] 94%|█████████▍| 377445/400000 [00:53<00:03, 7451.97it/s] 95%|█████████▍| 378197/400000 [00:53<00:02, 7470.93it/s] 95%|█████████▍| 378945/400000 [00:53<00:02, 7435.57it/s] 95%|█████████▍| 379706/400000 [00:53<00:02, 7486.22it/s] 95%|█████████▌| 380467/400000 [00:53<00:02, 7520.52it/s] 95%|█████████▌| 381220/400000 [00:53<00:02, 7369.16it/s] 95%|█████████▌| 381990/400000 [00:53<00:02, 7463.73it/s] 96%|█████████▌| 382754/400000 [00:53<00:02, 7515.68it/s] 96%|█████████▌| 383518/400000 [00:54<00:02, 7551.71it/s] 96%|█████████▌| 384274/400000 [00:54<00:02, 7430.19it/s] 96%|█████████▋| 385018/400000 [00:54<00:02, 7383.89it/s] 96%|█████████▋| 385776/400000 [00:54<00:01, 7440.88it/s] 97%|█████████▋| 386537/400000 [00:54<00:01, 7490.09it/s] 97%|█████████▋| 387307/400000 [00:54<00:01, 7551.48it/s] 97%|█████████▋| 388063/400000 [00:54<00:01, 7520.12it/s] 97%|█████████▋| 388816/400000 [00:54<00:01, 7261.25it/s] 97%|█████████▋| 389545/400000 [00:54<00:01, 7116.85it/s] 98%|█████████▊| 390259/400000 [00:54<00:01, 6981.29it/s] 98%|█████████▊| 390960/400000 [00:55<00:01, 6957.16it/s] 98%|█████████▊| 391685/400000 [00:55<00:01, 7039.79it/s] 98%|█████████▊| 392391/400000 [00:55<00:01, 7000.78it/s] 98%|█████████▊| 393155/400000 [00:55<00:00, 7180.63it/s] 98%|█████████▊| 393897/400000 [00:55<00:00, 7249.75it/s] 99%|█████████▊| 394624/400000 [00:55<00:00, 7185.53it/s] 99%|█████████▉| 395344/400000 [00:55<00:00, 7080.10it/s] 99%|█████████▉| 396083/400000 [00:55<00:00, 7169.19it/s] 99%|█████████▉| 396844/400000 [00:55<00:00, 7295.40it/s] 99%|█████████▉| 397575/400000 [00:55<00:00, 7164.40it/s]100%|█████████▉| 398327/400000 [00:56<00:00, 7266.38it/s]100%|█████████▉| 399055/400000 [00:56<00:00, 7154.16it/s]100%|█████████▉| 399782/400000 [00:56<00:00, 7186.97it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7101.66it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb49403ac50>, <torchtext.data.dataset.TabularDataset object at 0x7fb49403acf8>, <torchtext.vocab.Vocab object at 0x7fb49403c128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.14 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.14 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.14 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.07 file/s]2020-06-17 00:19:02.183580: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-17 00:19:02.188013: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-17 00:19:02.188274: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563896340090 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 00:19:02.188331: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:17, 127332.57it/s] 75%|███████▌  | 7462912/9912422 [00:00<00:13, 181761.89it/s]9920512it [00:00, 38981144.58it/s]                           
0it [00:00, ?it/s]32768it [00:00, 562030.56it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1029234.84it/s]1654784it [00:00, 12861780.93it/s]                           
0it [00:00, ?it/s]8192it [00:00, 257897.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
