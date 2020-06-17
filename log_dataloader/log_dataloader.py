
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe676f15ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe676f15ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe6e24cd0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe6e24cd0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe6fc1f9ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe6fc1f9ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe68fd48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe68fd48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe68fd48950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147066.20it/s] 83%|████████▎ | 8216576/9912422 [00:00<00:08, 209933.13it/s]9920512it [00:00, 43618995.83it/s]                           
0it [00:00, ?it/s]32768it [00:00, 592976.70it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141367.85it/s]1654784it [00:00, 10170384.68it/s]                         
0it [00:00, ?it/s]8192it [00:00, 183340.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe68d471710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe68d441978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe68fd48598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe68fd48598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe68fd48598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:25,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:25,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:40,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:40,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  5.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:27,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:18,  7.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:13,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:13,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 12.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 43.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 51.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 48.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 48.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 59.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 59.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 60.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 60.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.04s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  3.04s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 65.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.05 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.90s/ url]
0 examples [00:00, ? examples/s]2020-06-17 12:09:08.784552: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 12:09:08.797276: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095210000 Hz
2020-06-17 12:09:08.797506: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a92832f610 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 12:09:08.797523: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
50 examples [00:00, 497.38 examples/s]174 examples [00:00, 606.00 examples/s]287 examples [00:00, 702.83 examples/s]391 examples [00:00, 778.52 examples/s]488 examples [00:00, 826.39 examples/s]597 examples [00:00, 890.33 examples/s]707 examples [00:00, 942.66 examples/s]820 examples [00:00, 991.20 examples/s]940 examples [00:00, 1045.56 examples/s]1052 examples [00:01, 1065.90 examples/s]1174 examples [00:01, 1106.10 examples/s]1295 examples [00:01, 1132.92 examples/s]1412 examples [00:01, 1141.86 examples/s]1528 examples [00:01, 1119.26 examples/s]1641 examples [00:01, 1119.92 examples/s]1754 examples [00:01, 1109.94 examples/s]1871 examples [00:01, 1124.92 examples/s]1989 examples [00:01, 1138.79 examples/s]2109 examples [00:01, 1156.17 examples/s]2227 examples [00:02, 1162.71 examples/s]2348 examples [00:02, 1175.43 examples/s]2466 examples [00:02, 1144.04 examples/s]2581 examples [00:02, 1109.11 examples/s]2693 examples [00:02, 1099.76 examples/s]2804 examples [00:02, 1088.99 examples/s]2923 examples [00:02, 1104.22 examples/s]3042 examples [00:02, 1128.37 examples/s]3166 examples [00:02, 1158.69 examples/s]3284 examples [00:02, 1163.03 examples/s]3401 examples [00:03, 1125.10 examples/s]3519 examples [00:03, 1138.93 examples/s]3639 examples [00:03, 1156.40 examples/s]3755 examples [00:03, 1106.33 examples/s]3869 examples [00:03, 1114.75 examples/s]3986 examples [00:03, 1129.48 examples/s]4100 examples [00:03, 1125.47 examples/s]4217 examples [00:03, 1135.74 examples/s]4338 examples [00:03, 1155.85 examples/s]4464 examples [00:03, 1182.48 examples/s]4583 examples [00:04, 1174.40 examples/s]4701 examples [00:04, 1151.07 examples/s]4822 examples [00:04, 1165.53 examples/s]4939 examples [00:04, 1128.51 examples/s]5058 examples [00:04, 1144.33 examples/s]5174 examples [00:04, 1147.09 examples/s]5296 examples [00:04, 1167.06 examples/s]5414 examples [00:04, 1168.11 examples/s]5537 examples [00:04, 1184.87 examples/s]5663 examples [00:05, 1206.11 examples/s]5784 examples [00:05, 1198.85 examples/s]5905 examples [00:05, 1200.19 examples/s]6026 examples [00:05, 1155.83 examples/s]6143 examples [00:05, 1127.52 examples/s]6260 examples [00:05, 1138.57 examples/s]6377 examples [00:05, 1147.37 examples/s]6498 examples [00:05, 1163.31 examples/s]6617 examples [00:05, 1168.79 examples/s]6735 examples [00:05, 1155.32 examples/s]6851 examples [00:06, 1133.22 examples/s]6965 examples [00:06, 1130.17 examples/s]7082 examples [00:06, 1141.40 examples/s]7197 examples [00:06, 1102.51 examples/s]7308 examples [00:06, 1093.96 examples/s]7419 examples [00:06, 1095.13 examples/s]7529 examples [00:06, 1095.54 examples/s]7643 examples [00:06, 1106.28 examples/s]7758 examples [00:06, 1118.75 examples/s]7875 examples [00:06, 1133.03 examples/s]7989 examples [00:07, 1117.35 examples/s]8108 examples [00:07, 1135.68 examples/s]8232 examples [00:07, 1164.89 examples/s]8349 examples [00:07, 1119.53 examples/s]8464 examples [00:07, 1127.18 examples/s]8578 examples [00:07, 1097.63 examples/s]8700 examples [00:07, 1129.20 examples/s]8824 examples [00:07, 1159.16 examples/s]8949 examples [00:07, 1182.68 examples/s]9075 examples [00:08, 1204.83 examples/s]9196 examples [00:08, 1198.34 examples/s]9317 examples [00:08, 1176.25 examples/s]9435 examples [00:08, 1169.57 examples/s]9553 examples [00:08, 1132.86 examples/s]9668 examples [00:08, 1136.48 examples/s]9783 examples [00:08, 1140.17 examples/s]9902 examples [00:08, 1154.08 examples/s]10018 examples [00:08, 1119.12 examples/s]10141 examples [00:08, 1147.61 examples/s]10268 examples [00:09, 1181.75 examples/s]10387 examples [00:09, 1143.71 examples/s]10504 examples [00:09, 1150.56 examples/s]10620 examples [00:09, 1121.81 examples/s]10733 examples [00:09, 1120.34 examples/s]10849 examples [00:09, 1129.24 examples/s]10966 examples [00:09, 1140.92 examples/s]11087 examples [00:09, 1160.47 examples/s]11204 examples [00:09, 1155.38 examples/s]11327 examples [00:09, 1175.18 examples/s]11445 examples [00:10, 1167.94 examples/s]11562 examples [00:10, 1167.58 examples/s]11679 examples [00:10, 1145.80 examples/s]11794 examples [00:10, 1078.41 examples/s]11911 examples [00:10, 1103.00 examples/s]12031 examples [00:10, 1129.93 examples/s]12145 examples [00:10, 1125.76 examples/s]12259 examples [00:10, 1117.26 examples/s]12377 examples [00:10, 1134.41 examples/s]12504 examples [00:11, 1169.42 examples/s]12622 examples [00:11, 1134.61 examples/s]12746 examples [00:11, 1162.51 examples/s]12865 examples [00:11, 1167.77 examples/s]12983 examples [00:11, 1112.87 examples/s]13096 examples [00:11, 1117.01 examples/s]13209 examples [00:11, 1105.23 examples/s]13323 examples [00:11, 1114.01 examples/s]13438 examples [00:11, 1122.91 examples/s]13556 examples [00:11, 1136.95 examples/s]13670 examples [00:12, 1137.14 examples/s]13792 examples [00:12, 1159.21 examples/s]13909 examples [00:12, 1133.70 examples/s]14023 examples [00:12, 1093.89 examples/s]14133 examples [00:12, 1094.70 examples/s]14249 examples [00:12, 1112.96 examples/s]14361 examples [00:12, 1108.89 examples/s]14475 examples [00:12, 1117.35 examples/s]14590 examples [00:12, 1126.04 examples/s]14711 examples [00:12, 1149.89 examples/s]14831 examples [00:13, 1163.46 examples/s]14948 examples [00:13, 1137.06 examples/s]15064 examples [00:13, 1141.61 examples/s]15179 examples [00:13, 1104.06 examples/s]15297 examples [00:13, 1125.51 examples/s]15416 examples [00:13, 1142.44 examples/s]15533 examples [00:13, 1150.03 examples/s]15654 examples [00:13, 1166.02 examples/s]15778 examples [00:13, 1184.17 examples/s]15902 examples [00:13, 1197.65 examples/s]16028 examples [00:14, 1213.90 examples/s]16152 examples [00:14, 1218.82 examples/s]16275 examples [00:14, 1204.80 examples/s]16396 examples [00:14, 1164.31 examples/s]16516 examples [00:14, 1174.48 examples/s]16634 examples [00:14, 1141.84 examples/s]16751 examples [00:14, 1149.82 examples/s]16867 examples [00:14, 1135.07 examples/s]16981 examples [00:14, 1122.89 examples/s]17094 examples [00:15, 1091.68 examples/s]17211 examples [00:15, 1113.81 examples/s]17323 examples [00:15, 1088.76 examples/s]17433 examples [00:15, 1058.78 examples/s]17542 examples [00:15, 1067.23 examples/s]17661 examples [00:15, 1098.57 examples/s]17779 examples [00:15, 1120.01 examples/s]17895 examples [00:15, 1129.85 examples/s]18017 examples [00:15, 1153.71 examples/s]18138 examples [00:15, 1169.09 examples/s]18258 examples [00:16, 1176.56 examples/s]18384 examples [00:16, 1198.02 examples/s]18505 examples [00:16, 1187.79 examples/s]18624 examples [00:16, 1145.51 examples/s]18740 examples [00:16, 1122.63 examples/s]18857 examples [00:16, 1135.44 examples/s]18973 examples [00:16, 1140.75 examples/s]19088 examples [00:16, 1142.90 examples/s]19211 examples [00:16, 1165.76 examples/s]19329 examples [00:16, 1167.85 examples/s]19449 examples [00:17, 1176.73 examples/s]19575 examples [00:17, 1199.27 examples/s]19696 examples [00:17, 1177.85 examples/s]19815 examples [00:17, 1145.17 examples/s]19931 examples [00:17, 1147.61 examples/s]20047 examples [00:17, 1106.11 examples/s]20164 examples [00:17, 1123.51 examples/s]20285 examples [00:17, 1146.29 examples/s]20406 examples [00:17, 1163.50 examples/s]20527 examples [00:18, 1176.88 examples/s]20649 examples [00:18, 1187.02 examples/s]20768 examples [00:18, 1183.03 examples/s]20887 examples [00:18, 1136.90 examples/s]21002 examples [00:18, 1097.17 examples/s]21116 examples [00:18, 1109.38 examples/s]21234 examples [00:18, 1127.53 examples/s]21352 examples [00:18, 1140.98 examples/s]21469 examples [00:18, 1147.07 examples/s]21596 examples [00:18, 1179.22 examples/s]21718 examples [00:19, 1189.71 examples/s]21845 examples [00:19, 1210.62 examples/s]21969 examples [00:19, 1216.91 examples/s]22091 examples [00:19, 1147.09 examples/s]22207 examples [00:19, 1139.93 examples/s]22322 examples [00:19, 1139.13 examples/s]22437 examples [00:19, 1137.03 examples/s]22552 examples [00:19, 1140.69 examples/s]22667 examples [00:19, 1122.74 examples/s]22793 examples [00:19, 1158.53 examples/s]22910 examples [00:20, 1143.32 examples/s]23036 examples [00:20, 1174.71 examples/s]23154 examples [00:20, 1156.94 examples/s]23271 examples [00:20, 1135.65 examples/s]23388 examples [00:20, 1144.56 examples/s]23503 examples [00:20, 1119.10 examples/s]23616 examples [00:20, 1070.62 examples/s]23729 examples [00:20, 1084.76 examples/s]23844 examples [00:20, 1102.09 examples/s]23961 examples [00:21, 1121.30 examples/s]24081 examples [00:21, 1141.67 examples/s]24202 examples [00:21, 1159.28 examples/s]24319 examples [00:21, 1100.43 examples/s]24430 examples [00:21, 1063.55 examples/s]24546 examples [00:21, 1088.55 examples/s]24664 examples [00:21, 1112.65 examples/s]24782 examples [00:21, 1131.87 examples/s]24904 examples [00:21, 1154.53 examples/s]25031 examples [00:21, 1184.71 examples/s]25155 examples [00:22, 1199.44 examples/s]25277 examples [00:22, 1202.12 examples/s]25398 examples [00:22, 1192.70 examples/s]25518 examples [00:22, 1116.54 examples/s]25633 examples [00:22, 1126.19 examples/s]25749 examples [00:22, 1132.24 examples/s]25863 examples [00:22, 1129.58 examples/s]25984 examples [00:22, 1150.81 examples/s]26101 examples [00:22, 1156.35 examples/s]26226 examples [00:23, 1180.09 examples/s]26345 examples [00:23, 1181.49 examples/s]26464 examples [00:23, 1182.70 examples/s]26583 examples [00:23, 1154.19 examples/s]26699 examples [00:23, 1125.19 examples/s]26815 examples [00:23, 1134.53 examples/s]26935 examples [00:23, 1152.88 examples/s]27054 examples [00:23, 1161.24 examples/s]27171 examples [00:23, 1151.81 examples/s]27293 examples [00:23, 1171.44 examples/s]27417 examples [00:24, 1190.80 examples/s]27537 examples [00:24, 1193.42 examples/s]27657 examples [00:24, 1186.74 examples/s]27776 examples [00:24, 1104.59 examples/s]27888 examples [00:24, 1092.92 examples/s]27999 examples [00:24, 1071.51 examples/s]28116 examples [00:24, 1097.87 examples/s]28234 examples [00:24, 1118.73 examples/s]28347 examples [00:24, 1116.44 examples/s]28460 examples [00:24, 1118.56 examples/s]28577 examples [00:25, 1132.77 examples/s]28693 examples [00:25, 1139.53 examples/s]28808 examples [00:25, 1142.40 examples/s]28923 examples [00:25, 1104.75 examples/s]29040 examples [00:25, 1123.34 examples/s]29157 examples [00:25, 1136.37 examples/s]29275 examples [00:25, 1146.12 examples/s]29393 examples [00:25, 1153.11 examples/s]29512 examples [00:25, 1161.85 examples/s]29632 examples [00:25, 1171.37 examples/s]29754 examples [00:26, 1183.56 examples/s]29873 examples [00:26, 1185.17 examples/s]29992 examples [00:26, 1180.37 examples/s]30111 examples [00:26, 1084.42 examples/s]30227 examples [00:26, 1104.33 examples/s]30346 examples [00:26, 1127.38 examples/s]30460 examples [00:26, 1116.54 examples/s]30579 examples [00:26, 1137.36 examples/s]30694 examples [00:26, 1135.36 examples/s]30811 examples [00:27, 1144.82 examples/s]30938 examples [00:27, 1177.17 examples/s]31057 examples [00:27, 1158.61 examples/s]31174 examples [00:27, 1158.02 examples/s]31291 examples [00:27, 1113.40 examples/s]31403 examples [00:27, 1111.94 examples/s]31519 examples [00:27, 1123.88 examples/s]31632 examples [00:27, 1108.24 examples/s]31744 examples [00:27, 1110.11 examples/s]31856 examples [00:27, 1111.01 examples/s]31975 examples [00:28, 1133.20 examples/s]32093 examples [00:28, 1146.21 examples/s]32212 examples [00:28, 1156.93 examples/s]32328 examples [00:28, 1098.39 examples/s]32439 examples [00:28, 1085.23 examples/s]32553 examples [00:28, 1100.99 examples/s]32667 examples [00:28, 1110.28 examples/s]32785 examples [00:28, 1129.09 examples/s]32899 examples [00:28, 1098.47 examples/s]33015 examples [00:28, 1114.91 examples/s]33136 examples [00:29, 1139.22 examples/s]33256 examples [00:29, 1156.56 examples/s]33372 examples [00:29, 1154.97 examples/s]33488 examples [00:29, 1119.82 examples/s]33601 examples [00:29, 1098.99 examples/s]33714 examples [00:29, 1107.95 examples/s]33833 examples [00:29, 1129.33 examples/s]33952 examples [00:29, 1146.75 examples/s]34068 examples [00:29, 1148.45 examples/s]34184 examples [00:30, 1133.13 examples/s]34298 examples [00:30, 1131.50 examples/s]34416 examples [00:30, 1143.67 examples/s]34531 examples [00:30, 1130.80 examples/s]34645 examples [00:30, 1100.51 examples/s]34756 examples [00:30, 1103.06 examples/s]34870 examples [00:30, 1113.77 examples/s]34982 examples [00:30, 1091.87 examples/s]35094 examples [00:30, 1098.82 examples/s]35211 examples [00:30, 1116.48 examples/s]35323 examples [00:31, 1116.13 examples/s]35435 examples [00:31, 1089.94 examples/s]35554 examples [00:31, 1115.61 examples/s]35666 examples [00:31, 1090.79 examples/s]35776 examples [00:31, 1087.43 examples/s]35885 examples [00:31, 1079.74 examples/s]36001 examples [00:31, 1100.73 examples/s]36116 examples [00:31, 1114.88 examples/s]36232 examples [00:31, 1125.71 examples/s]36351 examples [00:31, 1142.67 examples/s]36468 examples [00:32, 1149.97 examples/s]36592 examples [00:32, 1174.67 examples/s]36711 examples [00:32, 1178.90 examples/s]36830 examples [00:32, 1141.86 examples/s]36945 examples [00:32, 1125.36 examples/s]37058 examples [00:32, 1099.04 examples/s]37169 examples [00:32, 1049.93 examples/s]37289 examples [00:32, 1088.93 examples/s]37405 examples [00:32, 1107.88 examples/s]37524 examples [00:33, 1128.60 examples/s]37646 examples [00:33, 1152.96 examples/s]37766 examples [00:33, 1164.70 examples/s]37883 examples [00:33, 1146.59 examples/s]37999 examples [00:33, 1103.99 examples/s]38115 examples [00:33, 1119.87 examples/s]38230 examples [00:33, 1127.79 examples/s]38344 examples [00:33, 1126.76 examples/s]38460 examples [00:33, 1134.64 examples/s]38580 examples [00:33, 1151.44 examples/s]38702 examples [00:34, 1169.50 examples/s]38824 examples [00:34, 1182.86 examples/s]38943 examples [00:34, 1183.93 examples/s]39062 examples [00:34, 1134.43 examples/s]39176 examples [00:34, 1118.90 examples/s]39289 examples [00:34, 1119.91 examples/s]39402 examples [00:34, 1116.93 examples/s]39515 examples [00:34, 1119.48 examples/s]39632 examples [00:34, 1132.76 examples/s]39746 examples [00:34, 1129.91 examples/s]39865 examples [00:35, 1145.12 examples/s]39982 examples [00:35, 1150.56 examples/s]40098 examples [00:35, 1087.04 examples/s]40208 examples [00:35, 1049.80 examples/s]40321 examples [00:35, 1072.41 examples/s]40434 examples [00:35, 1087.61 examples/s]40549 examples [00:35, 1105.04 examples/s]40662 examples [00:35, 1111.15 examples/s]40784 examples [00:35, 1141.51 examples/s]40899 examples [00:36, 1141.98 examples/s]41016 examples [00:36, 1148.53 examples/s]41133 examples [00:36, 1153.56 examples/s]41251 examples [00:36, 1158.53 examples/s]41367 examples [00:36, 1110.27 examples/s]41484 examples [00:36, 1126.46 examples/s]41600 examples [00:36, 1134.27 examples/s]41721 examples [00:36, 1152.56 examples/s]41837 examples [00:36, 1153.32 examples/s]41961 examples [00:36, 1176.85 examples/s]42083 examples [00:37, 1187.52 examples/s]42203 examples [00:37, 1189.28 examples/s]42325 examples [00:37, 1197.73 examples/s]42445 examples [00:37, 1175.69 examples/s]42563 examples [00:37, 1138.22 examples/s]42678 examples [00:37, 1139.81 examples/s]42796 examples [00:37, 1149.14 examples/s]42914 examples [00:37, 1155.57 examples/s]43031 examples [00:37, 1159.58 examples/s]43153 examples [00:37, 1175.94 examples/s]43276 examples [00:38, 1189.89 examples/s]43398 examples [00:38, 1196.38 examples/s]43518 examples [00:38, 1195.02 examples/s]43638 examples [00:38, 1163.71 examples/s]43755 examples [00:38, 1125.25 examples/s]43870 examples [00:38, 1131.09 examples/s]43984 examples [00:38, 1130.52 examples/s]44098 examples [00:38, 1129.06 examples/s]44212 examples [00:38, 1127.46 examples/s]44326 examples [00:38, 1127.63 examples/s]44439 examples [00:39, 1108.81 examples/s]44552 examples [00:39, 1114.49 examples/s]44664 examples [00:39, 1105.84 examples/s]44775 examples [00:39, 1085.97 examples/s]44884 examples [00:39, 1071.54 examples/s]44997 examples [00:39, 1085.75 examples/s]45107 examples [00:39, 1089.30 examples/s]45217 examples [00:39, 1090.49 examples/s]45331 examples [00:39, 1104.56 examples/s]45445 examples [00:39, 1112.79 examples/s]45557 examples [00:40, 1103.75 examples/s]45680 examples [00:40, 1138.73 examples/s]45795 examples [00:40, 1130.57 examples/s]45909 examples [00:40, 1087.07 examples/s]46025 examples [00:40, 1106.09 examples/s]46143 examples [00:40, 1126.38 examples/s]46260 examples [00:40, 1138.51 examples/s]46377 examples [00:40, 1147.28 examples/s]46492 examples [00:40, 1141.13 examples/s]46608 examples [00:41, 1146.44 examples/s]46725 examples [00:41, 1153.16 examples/s]46841 examples [00:41, 1150.00 examples/s]46957 examples [00:41, 1110.52 examples/s]47069 examples [00:41, 1092.59 examples/s]47181 examples [00:41, 1100.25 examples/s]47295 examples [00:41, 1110.28 examples/s]47407 examples [00:41, 1095.03 examples/s]47521 examples [00:41, 1106.06 examples/s]47641 examples [00:41, 1130.99 examples/s]47755 examples [00:42, 1121.86 examples/s]47868 examples [00:42, 1117.27 examples/s]47980 examples [00:42, 1116.99 examples/s]48092 examples [00:42, 1082.46 examples/s]48201 examples [00:42, 1079.96 examples/s]48310 examples [00:42, 1072.85 examples/s]48425 examples [00:42, 1094.19 examples/s]48537 examples [00:42, 1100.50 examples/s]48650 examples [00:42, 1108.69 examples/s]48766 examples [00:42, 1120.86 examples/s]48886 examples [00:43, 1141.64 examples/s]49011 examples [00:43, 1169.32 examples/s]49129 examples [00:43, 1171.90 examples/s]49247 examples [00:43, 1133.72 examples/s]49361 examples [00:43, 1115.41 examples/s]49474 examples [00:43, 1116.81 examples/s]49592 examples [00:43, 1134.80 examples/s]49708 examples [00:43, 1141.54 examples/s]49829 examples [00:43, 1158.76 examples/s]49950 examples [00:43, 1172.64 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▋        | 8157/50000 [00:00<00:00, 81567.74 examples/s] 45%|████▍     | 22286/50000 [00:00<00:00, 93412.94 examples/s] 73%|███████▎  | 36688/50000 [00:00<00:00, 104418.75 examples/s]                                                                0 examples [00:00, ? examples/s]99 examples [00:00, 984.92 examples/s]213 examples [00:00, 1025.88 examples/s]325 examples [00:00, 1051.72 examples/s]438 examples [00:00, 1073.61 examples/s]555 examples [00:00, 1098.19 examples/s]676 examples [00:00, 1127.70 examples/s]785 examples [00:00, 1114.95 examples/s]902 examples [00:00, 1130.76 examples/s]1010 examples [00:00, 1106.74 examples/s]1119 examples [00:01, 1099.92 examples/s]1238 examples [00:01, 1124.78 examples/s]1359 examples [00:01, 1146.71 examples/s]1478 examples [00:01, 1158.60 examples/s]1600 examples [00:01, 1175.61 examples/s]1726 examples [00:01, 1197.12 examples/s]1853 examples [00:01, 1217.88 examples/s]1977 examples [00:01, 1223.39 examples/s]2100 examples [00:01, 1216.80 examples/s]2222 examples [00:01, 1172.35 examples/s]2340 examples [00:02, 1167.48 examples/s]2460 examples [00:02, 1174.67 examples/s]2580 examples [00:02, 1181.94 examples/s]2699 examples [00:02, 1165.12 examples/s]2816 examples [00:02, 1165.70 examples/s]2937 examples [00:02, 1177.62 examples/s]3058 examples [00:02, 1185.46 examples/s]3183 examples [00:02, 1202.55 examples/s]3304 examples [00:02, 1202.96 examples/s]3425 examples [00:02, 1147.55 examples/s]3544 examples [00:03, 1157.99 examples/s]3667 examples [00:03, 1176.71 examples/s]3786 examples [00:03, 1179.42 examples/s]3905 examples [00:03, 1182.05 examples/s]4032 examples [00:03, 1206.82 examples/s]4161 examples [00:03, 1228.10 examples/s]4288 examples [00:03, 1239.41 examples/s]4413 examples [00:03, 1195.70 examples/s]4534 examples [00:03, 1132.86 examples/s]4649 examples [00:03, 1118.92 examples/s]4769 examples [00:04, 1140.93 examples/s]4890 examples [00:04, 1160.65 examples/s]5013 examples [00:04, 1180.48 examples/s]5132 examples [00:04, 1172.85 examples/s]5256 examples [00:04, 1191.21 examples/s]5378 examples [00:04, 1199.60 examples/s]5504 examples [00:04, 1216.51 examples/s]5626 examples [00:04, 1216.14 examples/s]5748 examples [00:04, 1178.08 examples/s]5867 examples [00:05, 1137.35 examples/s]5988 examples [00:05, 1157.44 examples/s]6108 examples [00:05, 1167.32 examples/s]6226 examples [00:05, 1144.79 examples/s]6349 examples [00:05, 1167.31 examples/s]6470 examples [00:05, 1179.24 examples/s]6589 examples [00:05, 1171.71 examples/s]6709 examples [00:05, 1178.57 examples/s]6831 examples [00:05, 1188.81 examples/s]6951 examples [00:05, 1109.36 examples/s]7069 examples [00:06, 1127.45 examples/s]7185 examples [00:06, 1136.11 examples/s]7306 examples [00:06, 1155.96 examples/s]7431 examples [00:06, 1180.52 examples/s]7550 examples [00:06, 1131.90 examples/s]7672 examples [00:06, 1156.86 examples/s]7789 examples [00:06, 1148.53 examples/s]7905 examples [00:06, 1141.71 examples/s]8020 examples [00:06, 1120.76 examples/s]8136 examples [00:06, 1132.10 examples/s]8261 examples [00:07, 1163.22 examples/s]8386 examples [00:07, 1186.16 examples/s]8508 examples [00:07, 1195.77 examples/s]8628 examples [00:07, 1188.38 examples/s]8749 examples [00:07, 1193.18 examples/s]8870 examples [00:07, 1197.25 examples/s]8996 examples [00:07, 1213.71 examples/s]9123 examples [00:07, 1227.94 examples/s]9246 examples [00:07, 1224.13 examples/s]9369 examples [00:08, 1191.02 examples/s]9489 examples [00:08, 1192.78 examples/s]9609 examples [00:08, 1181.72 examples/s]9729 examples [00:08, 1185.14 examples/s]9854 examples [00:08, 1201.38 examples/s]9975 examples [00:08, 1202.34 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete58YEZD/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete58YEZD/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe68fd48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe68fd48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe68fd48950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe619216860>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe6191d8be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe68fd48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe68fd48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe68fd48950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe68fd36ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe68d441668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe60780f620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe60780f620> 

  function with postional parmater data_info <function split_train_valid at 0x7fe60780f620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe60780f730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe60780f730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe60780f730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=9e1ba6771183c1839097e78c935425486b0a28dd4595e58bd0f6c4480a2ca70b
  Stored in directory: /tmp/pip-ephem-wheel-cache-thg4y09w/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<29:18:08, 8.17kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:45:51, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<14:35:21, 16.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 655k/862M [00:01<10:13:26, 23.4kB/s].vector_cache/glove.6B.zip:   0%|          | 991k/862M [00:01<7:11:38, 33.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.11M/862M [00:01<5:01:38, 47.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.62M/862M [00:01<3:30:24, 67.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.96M/862M [00:01<2:27:04, 96.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<1:42:42, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.0M/862M [00:02<1:11:57, 196kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:02<50:23, 280kB/s]  .vector_cache/glove.6B.zip:   2%|▏         | 19.1M/862M [00:02<35:21, 397kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.6M/862M [00:02<24:46, 565kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:02<17:43, 789kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:02<12:42, 1.10MB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.1M/862M [00:02<08:59, 1.54MB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.2M/862M [00:02<06:29, 2.14MB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:02<04:44, 2.92MB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:03<03:32, 3.89MB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:03<02:44, 5.02MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:03<02:06, 6.54MB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:03<01:38, 8.33MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:03<01:19, 10.4MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.7M/862M [00:03<01:06, 12.3MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:03<00:58, 14.0MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:03<00:47, 17.1MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:03<01:14, 10.9MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.5M/862M [00:04<01:05, 12.4MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:04<00:59, 13.6MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:05<16:26, 818kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:06<17:18, 776kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:06<15:58, 841kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:06<12:44, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<09:39, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:06<07:08, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:06<05:35, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<04:15, 3.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<1:09:45, 192kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:08<50:19, 266kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:08<35:28, 376kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<24:57, 533kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<56:39, 235kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<42:39, 312kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:10<30:29, 436kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:10<21:36, 614kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<15:18, 865kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:11<1:43:42, 128kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:11<1:15:34, 175kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:12<53:34, 247kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:12<37:46, 349kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<29:30, 446kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:13<23:37, 557kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:14<17:14, 762kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:14<12:10, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<18:05, 723kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:15<15:28, 846kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:16<11:27, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:16<08:15, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<09:32, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:17<09:29, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:18<07:14, 1.80MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:18<05:17, 2.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<08:04, 1.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<08:25, 1.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:20<06:29, 2.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:20<04:43, 2.73MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:21<08:08, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:21<08:29, 1.52MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<06:34, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:22<04:48, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:23<07:46, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:23<08:20, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<06:33, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.4M/862M [00:24<04:43, 2.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:25<11:04, 1.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<10:37, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:25<08:03, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:26<05:56, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:32<15:48, 802kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:32<26:48, 473kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:32<25:24, 499kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:32<19:34, 647kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:32<14:21, 882kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:32<10:14, 1.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<14:08, 891kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<22:18, 565kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<22:45, 554kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:35<20:19, 620kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:36<15:17, 824kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:36<11:06, 1.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:36<07:54, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:36<08:23, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:36<06:28, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:36<04:43, 2.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:38<07:34, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:38<14:27, 863kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:38<15:06, 825kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:38<11:27, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:38<08:32, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<06:05, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:40<5:27:17, 37.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:40<3:51:57, 53.5kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:40<2:44:49, 75.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:40<1:55:34, 107kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:42<1:22:58, 149kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:42<1:03:33, 194kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:42<47:01, 262kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:42<33:53, 364kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:42<24:00, 513kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:44<19:13, 638kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:44<16:17, 753kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:44<12:59, 943kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:44<09:25, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:49<13:26, 907kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<22:40, 538kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<18:48, 648kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<13:49, 881kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:49<10:07, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:52<11:53, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:52<21:10, 572kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:52<17:49, 680kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:53<13:12, 917kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:53<09:25, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:54<10:49, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:55<13:24, 900kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:55<12:50, 939kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:55<10:00, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:55<07:28, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:55<05:32, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:55<04:12, 2.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:56<11:53, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:57<10:10, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:57<07:58, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:57<05:51, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:57<04:14, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:58<1:29:30, 133kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:58<1:06:50, 178kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:59<48:05, 248kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:59<34:11, 348kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:59<24:00, 494kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [01:00<23:11, 511kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [01:00<20:36, 575kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:01<16:14, 729kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:01<12:00, 986kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:01<08:34, 1.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:02<09:55, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:02<10:33, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:03<08:36, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:03<06:20, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:04<07:00, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:04<08:38, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:05<08:28, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:05<06:55, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:05<05:13, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:05<03:56, 2.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:06<06:30, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:06<06:28, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:06<05:35, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:07<04:19, 2.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:07<03:10, 3.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:08<10:32, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:08<13:52, 834kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:08<10:57, 1.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:09<08:15, 1.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:09<05:55, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:10<09:00, 1.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:10<09:32, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:10<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:11<05:51, 1.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:11<04:13, 2.70MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:12<13:27, 850kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:12<13:22, 855kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:12<10:15, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:13<07:33, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:14<07:21, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:14<07:44, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:14<06:02, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:15<04:21, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:16<16:02, 704kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:16<13:29, 837kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:16<09:56, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:17<07:03, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:18<28:58, 388kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:18<22:44, 494kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:18<16:32, 678kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:19<11:41, 956kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:20<13:55, 802kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:20<12:28, 894kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:20<09:15, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:20<06:39, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:22<08:15, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:22<09:16, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:22<08:10, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:22<06:19, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:23<04:41, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:24<05:55, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:24<07:04, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:24<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:25<04:02, 2.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:31<20:37, 531kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:31<25:54, 423kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:31<20:45, 527kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:31<15:47, 693kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:31<11:53, 920kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:31<08:39, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:33<08:00, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:33<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:33<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:39<12:11, 887kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:39<20:18, 533kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:39<16:51, 641kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:40<13:07, 824kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:40<09:27, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:41<08:48, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:41<08:53, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:41<09:54, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:41<08:25, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:42<17:51, 601kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:43<13:37, 783kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:43<10:34, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:43<08:03, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:44<05:50, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:45<06:50, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:45<06:47, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:45<05:23, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:45<03:56, 2.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:50<12:18, 856kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:50<19:23, 543kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:50<16:11, 651kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:51<12:15, 859kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:51<08:49, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:51<06:22, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:51<04:41, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:51<03:28, 3.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:53<11:11, 929kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:54<12:39, 822kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:54<10:01, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:54<07:18, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:56<07:53, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:56<16:26, 628kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:56<19:16, 536kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:56<17:09, 602kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:56<13:30, 765kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:56<09:56, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:57<07:10, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [02:02<13:42, 748kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [02:02<20:09, 509kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [02:02<16:26, 624kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [02:02<12:06, 846kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [02:02<08:50, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [02:02<06:25, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:08<21:53, 466kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:08<32:02, 318kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:08<30:06, 338kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:08<23:24, 435kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:08<17:23, 586kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [02:08<12:40, 803kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [02:09<08:57, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:11<20:42, 489kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:12<25:33, 396kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:12<21:08, 479kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:12<17:08, 590kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:12<13:02, 775kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [02:12<09:36, 1.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [02:12<07:09, 1.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [02:12<05:22, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [02:12<04:04, 2.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:15<35:28, 283kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:15<35:35, 283kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:15<27:21, 367kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:16<20:08, 499kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:16<14:24, 696kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:16<10:19, 970kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [02:19<12:41, 787kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [02:19<19:18, 517kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [02:19<15:47, 632kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:19<11:39, 855kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:19<08:31, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:19<06:22, 1.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:19<04:42, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:21<14:49, 669kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:21<15:41, 632kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:21<13:10, 753kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [02:21<09:57, 994kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [02:22<07:23, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:22<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:24<09:34, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:24<17:01, 579kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:25<14:14, 692kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [02:25<10:32, 933kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:25<07:33, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:31<15:40, 624kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:31<31:06, 314kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:31<26:46, 365kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:31<19:44, 495kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:31<14:28, 674kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:32<10:12, 952kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:37<24:32, 396kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:37<26:05, 372kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:37<23:19, 416kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:37<17:30, 554kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [02:37<12:59, 747kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:37<09:27, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:40<10:04, 958kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:40<24:54, 387kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:40<21:28, 449kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:40<18:37, 518kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:40<13:59, 689kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:41<10:07, 950kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:41<07:11, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:41<06:43, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:41<05:07, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:41<03:45, 2.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:44<07:54, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:44<15:29, 613kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:44<16:55, 561kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:44<13:31, 703kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:44<10:02, 945kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:44<07:13, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:45<05:14, 1.80MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:49<31:15, 302kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:49<34:09, 276kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:49<26:10, 360kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:49<18:51, 499kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:50<13:16, 706kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:53<29:38, 316kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:53<30:49, 304kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:53<23:49, 393kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:54<17:09, 545kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:54<12:09, 767kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:55<10:18, 902kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:55<08:02, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:55<06:20, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:55<04:48, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:59<07:33, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:59<14:00, 658kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:59<14:54, 619kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:59<11:38, 792kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:59<10:03, 918kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:59<07:45, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:59<05:29, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [03:04<59:33, 154kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:04<49:55, 183kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:04<38:08, 240kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:04<29:35, 309kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:04<21:09, 432kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [03:04<14:56, 610kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [03:08<15:51, 573kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [03:08<20:35, 442kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [03:08<17:31, 519kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [03:08<13:23, 678kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [03:08<09:45, 929kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [03:08<06:57, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:11<11:12, 804kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:11<28:52, 313kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:11<25:28, 354kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:11<18:44, 481kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [03:11<13:22, 673kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [03:12<09:30, 943kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:17<18:02, 496kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:17<21:51, 410kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:17<17:58, 498kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [03:17<13:09, 680kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [03:17<09:19, 956kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [03:22<15:13, 583kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [03:22<18:26, 482kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [03:22<15:49, 561kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [03:22<11:44, 756kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [03:22<08:29, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [03:27<11:22, 775kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [03:27<18:18, 482kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [03:27<14:56, 590kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [03:28<10:59, 801kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:28<07:56, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [03:33<11:11, 782kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [03:33<16:56, 517kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [03:33<14:56, 586kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [03:33<11:13, 778kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [03:33<08:06, 1.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [03:40<12:56, 671kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [03:40<17:09, 506kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [03:40<20:33, 422kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:40<19:10, 453kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:40<14:12, 610kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [03:40<10:07, 854kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [03:41<07:10, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [03:42<34:09, 252kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [03:42<24:43, 348kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [03:42<18:06, 475kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [03:42<13:29, 637kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [03:42<09:52, 869kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [03:42<07:29, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [03:42<05:24, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [03:43<12:13, 699kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [03:43<08:35, 989kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [03:43<06:04, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [03:43<04:18, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [03:58<20:29, 408kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [03:58<21:57, 381kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [03:58<17:22, 481kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [03:58<12:52, 648kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [03:58<09:16, 899kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [03:58<06:36, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [03:59<10:53, 761kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [03:59<08:15, 1.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:59<06:10, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [04:00<04:36, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [04:00<03:35, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [04:00<02:48, 2.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [04:04<18:49, 437kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [04:05<21:48, 377kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [04:05<17:15, 476kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [04:05<12:33, 654kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [04:05<08:52, 921kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:05<06:26, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:08<4:58:21, 27.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:08<3:36:35, 37.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:08<2:36:04, 52.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:08<1:52:55, 72.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:08<1:23:01, 98.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<1:02:12, 131kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<47:38, 171kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<36:00, 226kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<29:00, 281kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<24:41, 330kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [04:09<19:18, 422kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [04:09<16:00, 509kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [04:09<12:15, 664kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [04:09<09:13, 882kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [04:10<06:44, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [04:10<04:55, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [04:11<09:34, 844kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [04:12<10:02, 806kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [04:12<08:55, 906kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [04:12<06:39, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [04:12<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [04:15<06:14, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [04:15<12:51, 623kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [04:15<10:51, 739kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [04:15<08:36, 931kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [04:15<06:36, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [04:15<04:56, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [04:15<03:41, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:15<02:46, 2.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:18<13:15:00, 10.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:18<9:26:24, 14.0kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:18<6:38:15, 20.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [04:19<4:38:51, 28.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [04:19<3:14:35, 40.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [04:21<2:19:42, 56.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [04:21<2:09:03, 61.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [04:21<1:34:28, 83.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [04:21<1:09:23, 114kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [04:21<51:13, 154kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [04:22<37:09, 212kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [04:22<26:34, 296kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [04:22<18:47, 418kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [04:22<13:18, 589kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [04:27<21:26, 364kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [04:27<22:02, 354kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [04:27<17:29, 447kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [04:27<13:48, 566kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [04:27<10:01, 778kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [04:27<07:07, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [04:30<10:01, 772kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [04:30<15:19, 505kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [04:30<18:39, 415kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [04:30<15:16, 506kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [04:30<12:01, 644kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [04:30<08:45, 882kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [04:31<06:20, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [04:31<04:33, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [04:32<10:04, 761kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [04:32<08:12, 935kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [04:32<06:25, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [04:32<04:59, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [04:32<03:49, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [04:32<02:48, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [04:38<20:01, 380kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [04:38<21:58, 346kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [04:38<17:14, 441kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [04:38<13:10, 577kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [04:38<09:29, 799kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [04:38<06:42, 1.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [04:43<22:18, 338kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [04:43<22:32, 334kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [04:43<17:36, 428kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [04:43<12:44, 590kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [04:43<09:11, 817kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [04:43<06:31, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [04:43<04:43, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [04:43<03:34, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [04:45<04:27, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [04:45<03:50, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [04:45<03:02, 2.44MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [04:45<02:15, 3.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [04:47<04:01, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [04:47<04:23, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [04:47<04:36, 1.59MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [04:47<04:24, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [04:47<03:39, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [04:47<03:04, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [04:47<02:48, 2.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [04:47<02:25, 3.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [04:48<02:02, 3.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [04:48<01:49, 3.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [04:49<04:13, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [04:49<03:36, 2.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [04:49<02:45, 2.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [04:49<02:01, 3.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [04:58<27:29, 262kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [04:59<25:32, 282kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [04:59<19:38, 367kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [04:59<14:36, 493kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [04:59<10:26, 688kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [04:59<07:22, 969kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [04:59<06:10, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [04:59<04:30, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [04:59<03:19, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [05:03<06:30, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [05:03<12:01, 588kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [05:03<09:59, 707kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [05:03<07:50, 900kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [05:03<05:44, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [05:04<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [05:05<07:37, 918kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [05:05<08:56, 783kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [05:05<08:11, 853kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [05:05<07:13, 968kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [05:05<06:23, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [05:05<05:02, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [05:06<03:44, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [05:07<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [05:07<04:55, 1.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [05:07<05:09, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [05:07<04:14, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [05:07<03:06, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [05:07<02:18, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<05:26, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<09:15, 741kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<09:01, 760kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<08:37, 795kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<08:22, 818kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:09<08:21, 820kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:10<08:26, 812kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:10<07:43, 886kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:10<07:17, 940kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:10<06:46, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [05:10<06:20, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [05:10<06:05, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [05:10<05:21, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [05:10<04:23, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [05:10<03:44, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [05:11<03:30, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [05:11<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [05:11<03:19, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [05:11<03:02, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [05:11<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [05:11<03:46, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [05:11<03:23, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [05:11<03:32, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [05:11<03:27, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [05:11<03:12, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [05:12<02:41, 2.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [05:12<02:10, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [05:12<01:44, 3.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [05:12<02:42, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [05:12<02:33, 2.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [05:13<01:52, 3.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [05:13<01:25, 4.68MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [05:15<13:26, 495kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [05:15<16:04, 415kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [05:15<13:02, 511kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [05:15<10:52, 612kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [05:15<08:15, 806kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [05:15<06:07, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [05:15<04:23, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [05:17<05:06, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [05:17<04:35, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [05:17<03:34, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [05:17<02:46, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [05:17<02:01, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [05:19<07:38, 853kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [05:19<07:43, 844kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [05:19<06:33, 993kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [05:19<04:46, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [05:19<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [05:21<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [05:21<07:42, 838kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [05:21<06:25, 1.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [05:22<05:00, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [05:22<03:36, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [05:22<02:38, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [05:45<2:59:53, 35.5kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [05:46<2:11:17, 48.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [05:46<1:33:28, 68.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [05:46<1:05:40, 97.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [05:46<45:55, 138kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [05:46<32:19, 195kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [05:46<22:36, 278kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [05:46<15:50, 395kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [05:48<27:18, 229kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [05:48<23:16, 268kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [05:48<17:30, 357kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [05:49<12:40, 492kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [05:49<09:04, 686kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:49<06:23, 967kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:53<35:01, 176kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:53<30:44, 201kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [05:54<22:57, 269kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [05:54<16:24, 376kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [05:54<11:31, 531kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [05:55<10:34, 578kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [05:55<10:35, 576kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [05:55<09:15, 660kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [05:56<07:23, 826kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [05:56<05:31, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [05:56<04:00, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [05:56<02:54, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [05:57<06:38, 909kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [05:57<05:42, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [05:58<04:10, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [05:58<03:10, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [05:58<02:20, 2.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [05:58<01:49, 3.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [06:03<6:27:23, 15.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [06:03<4:37:06, 21.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [06:03<3:15:50, 30.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [06:03<2:18:21, 43.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [06:04<1:37:13, 61.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [06:04<1:08:12, 87.2kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [06:04<47:52, 124kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [06:04<33:30, 176kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [06:04<23:48, 248kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [06:04<16:48, 350kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [06:04<12:06, 486kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [06:04<08:44, 671kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [06:04<06:23, 917kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [06:07<07:51, 742kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [06:07<10:25, 560kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [06:07<08:48, 662kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [06:07<07:18, 797kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [06:07<05:14, 1.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [06:07<03:46, 1.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [06:09<05:08, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [06:09<04:52, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [06:09<04:08, 1.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [06:09<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [06:09<02:54, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [06:09<02:35, 2.22MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [06:09<02:13, 2.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [06:09<02:00, 2.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [06:10<01:41, 3.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:10<01:29, 3.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:10<01:21, 4.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:11<13:00, 438kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:11<10:03, 566kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [06:11<07:46, 731kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [06:11<05:51, 969kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [06:11<04:35, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [06:11<03:21, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [06:11<02:28, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [06:13<08:17, 679kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [06:13<06:47, 829kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [06:13<05:09, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [06:13<03:42, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [06:13<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [06:15<04:17, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [06:15<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [06:15<03:43, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [06:15<02:50, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [06:15<02:12, 2.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [06:15<01:44, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [06:17<04:29, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [06:17<04:20, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [06:17<04:04, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [06:17<03:14, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [06:17<02:44, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [06:17<02:07, 2.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [06:17<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [06:17<01:37, 3.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [06:17<01:25, 3.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [06:18<01:19, 4.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [06:19<17:44, 306kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [06:19<13:14, 409kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [06:19<09:57, 543kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [06:19<07:23, 731kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [06:19<05:37, 962kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [06:19<04:27, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [06:19<03:28, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [06:19<02:41, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [06:19<02:10, 2.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [06:21<03:22, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [06:21<03:03, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [06:21<02:21, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [06:21<01:46, 2.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [06:21<01:22, 3.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [06:21<01:24, 3.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [06:23<06:40, 791kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [06:23<05:06, 1.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [06:23<03:47, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [06:23<02:52, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [06:25<03:12, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [06:25<04:23, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [06:25<03:35, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [06:25<02:40, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [06:25<02:00, 2.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [06:25<01:37, 3.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [06:25<01:25, 3.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [06:27<06:12, 828kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [06:27<05:22, 956kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [06:27<03:58, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [06:27<02:53, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [06:27<02:08, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [06:29<05:26, 932kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [06:29<05:39, 896kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [06:29<04:24, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [06:29<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [06:29<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [06:29<02:06, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [06:33<08:23, 597kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [06:33<11:12, 447kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [06:33<08:57, 558kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [06:33<07:07, 702kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [06:33<05:26, 917kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [06:33<03:53, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [06:33<02:56, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [06:34<02:12, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [06:34<02:14, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [06:34<01:43, 2.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [06:34<01:17, 3.76MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [06:36<02:29, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [06:36<03:32, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [06:36<02:53, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [06:37<02:07, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:39<03:30, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:39<11:48, 404kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:39<11:14, 424kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:39<09:39, 494kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:40<08:20, 571kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [06:40<06:15, 761kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [06:40<04:31, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [06:40<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:55<13:55, 338kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:55<14:36, 322kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:55<11:22, 413kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [06:55<08:14, 569kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [06:55<05:56, 787kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [06:55<04:27, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [06:55<03:14, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:55<02:31, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:57<08:20, 555kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:57<08:44, 530kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:57<07:22, 628kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [06:58<05:38, 819kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [06:58<04:09, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [06:58<02:59, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [06:58<02:13, 2.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [06:59<04:17, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [06:59<04:23, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [06:59<03:44, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [07:00<02:58, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [07:00<02:21, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [07:00<01:55, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [07:00<01:34, 2.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [07:00<01:12, 3.73MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [07:05<12:26, 361kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [07:05<15:16, 294kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [07:05<11:56, 376kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [07:05<08:36, 520kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [07:06<06:03, 734kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [07:07<05:55, 747kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [07:07<06:02, 732kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [07:07<06:20, 698kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [07:08<05:20, 828kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [07:08<04:11, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [07:08<03:17, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [07:08<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [07:08<02:07, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [07:08<01:46, 2.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [07:08<01:25, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:09<02:36, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:09<02:53, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:09<03:37, 1.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:09<03:37, 1.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:10<04:02, 1.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:10<04:34, 949kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [07:10<04:53, 889kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<04:57, 876kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<05:52, 740kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<05:07, 846kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<04:19, 1.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<03:28, 1.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [07:10<03:02, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [07:11<02:45, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [07:11<02:33, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [07:11<02:19, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [07:11<02:09, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [07:11<02:01, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [07:11<02:40, 1.61MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [07:11<02:12, 1.95MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [07:11<01:57, 2.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [07:11<01:47, 2.41MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [07:12<01:41, 2.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [07:12<01:40, 2.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [07:12<01:38, 2.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [07:12<01:30, 2.82MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [07:12<01:17, 3.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [07:12<01:10, 3.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [07:12<01:37, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [07:13<01:17, 3.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [07:13<00:58, 4.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [07:13<00:45, 5.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [07:14<32:30, 128kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [07:15<26:11, 159kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [07:15<19:08, 218kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [07:15<13:32, 307kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [07:15<09:31, 434kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [07:15<06:45, 609kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [07:16<07:17, 562kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [07:17<05:28, 747kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [07:17<03:54, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [07:17<02:47, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [07:18<05:44, 703kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [07:19<05:33, 725kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [07:19<04:15, 947kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [07:19<03:03, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [07:19<02:12, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [07:20<10:15, 386kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [07:21<08:57, 442kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [07:21<06:44, 587kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [07:21<04:47, 821kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [07:21<03:27, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [07:22<03:43, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [07:22<04:21, 892kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [07:23<03:25, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [07:23<02:37, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [07:23<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [07:23<01:25, 2.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [07:24<05:53, 649kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [07:24<06:01, 634kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [07:25<05:34, 685kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [07:25<05:00, 763kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [07:25<03:54, 975kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [07:25<02:52, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [07:25<02:04, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [07:25<01:35, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [07:25<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [07:25<01:19, 2.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [07:26<01:00, 3.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [08:06<42:24, 87.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [08:06<33:19, 111kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [08:07<24:11, 152kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [08:07<17:03, 215kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [08:07<11:58, 305kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [08:07<08:27, 430kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<09:39, 375kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<10:05, 359kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<09:16, 390kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<07:27, 485kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<06:03, 597kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [08:11<05:17, 684kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [08:11<04:00, 902kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [08:11<02:59, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [08:11<02:13, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [08:12<01:38, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [08:13<02:29, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [08:13<02:19, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [08:13<02:03, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [08:13<01:44, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [08:13<01:31, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [08:13<01:24, 2.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [08:13<01:08, 3.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [08:13<00:52, 3.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [08:15<03:06, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [08:15<02:58, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [08:15<02:29, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [08:15<01:51, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [08:15<01:25, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [08:15<01:08, 3.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [08:15<00:56, 3.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [08:16<11:23, 300kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [08:17<08:19, 410kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [08:17<06:00, 567kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [08:17<04:16, 792kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [08:17<03:03, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [08:18<03:39, 914kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [08:19<04:05, 816kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [08:19<03:11, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [08:19<02:25, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [08:19<01:47, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [08:19<01:21, 2.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [08:19<01:05, 2.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [08:20<06:50, 479kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [08:21<05:44, 570kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [08:21<04:51, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [08:21<04:12, 778kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [08:21<03:14, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [08:21<02:20, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [08:21<01:48, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [08:21<01:20, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [08:23<02:30, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [08:23<02:27, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [08:23<01:50, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [08:23<01:25, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [08:23<01:04, 2.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [08:23<00:54, 3.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [08:25<05:59, 523kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [08:25<04:30, 693kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [08:25<03:16, 952kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [08:25<02:19, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [08:25<01:41, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [08:29<35:11, 87.0kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [08:29<27:42, 111kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [08:30<20:05, 152kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [08:30<14:11, 215kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [08:30<09:53, 305kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [08:30<06:57, 431kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [08:31<10:17, 291kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [08:31<08:09, 367kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [08:31<06:16, 476kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [08:32<04:43, 631kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [08:32<03:35, 828kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [08:32<02:41, 1.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [08:32<01:55, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [08:33<02:13, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [08:33<02:03, 1.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [08:33<01:44, 1.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [08:34<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [08:34<00:57, 3.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [08:34<00:45, 3.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [08:35<10:40, 267kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [08:35<08:33, 334kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [08:36<06:11, 459kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [08:36<04:30, 630kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [08:36<03:17, 859kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [08:36<02:21, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [08:37<02:32, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [08:37<02:16, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [08:38<01:42, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [08:38<01:15, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [08:38<00:58, 2.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [08:39<01:52, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [08:39<02:28, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [08:39<01:58, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [08:40<01:28, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [08:40<01:05, 2.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [08:40<00:48, 3.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [08:41<08:49, 300kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [08:41<07:14, 365kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [08:41<05:20, 494kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [08:42<03:53, 676kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [08:42<02:55, 899kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [08:42<02:13, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [08:42<01:43, 1.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [08:42<01:19, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [08:42<01:03, 2.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [08:43<02:00, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [08:43<02:10, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [08:43<01:37, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [08:44<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [08:44<01:25, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [08:44<01:14, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [08:44<01:05, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [08:44<00:57, 2.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [08:44<01:20, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [08:44<01:12, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [08:44<01:01, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [08:45<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [08:45<09:18, 270kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [08:45<07:00, 358kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [08:45<05:07, 487kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [08:46<03:45, 664kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [08:46<02:52, 866kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [08:46<02:11, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [08:46<01:43, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [08:46<01:22, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [08:46<01:14, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [08:46<01:32, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [08:46<01:15, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:46<01:02, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:47<13:09, 185kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:47<09:50, 247kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:47<07:15, 335kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:47<05:44, 423kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:48<04:48, 506kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [08:48<04:07, 588kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<03:38, 667kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<03:25, 707kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<03:11, 760kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<03:18, 731kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<02:49, 856kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<02:43, 888kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:48<02:39, 907kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:49<02:36, 928kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:49<02:36, 924kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [08:49<02:34, 939kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:49<02:25, 991kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:49<02:18, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:49<02:57, 811kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:49<02:38, 911kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:49<02:29, 964kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [08:50<06:51, 350kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [08:50<04:55, 486kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [08:50<03:57, 603kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [08:50<03:04, 774kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [08:50<02:45, 863kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [08:50<02:21, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [08:50<01:54, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [08:51<01:36, 1.48MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [08:51<01:22, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [08:51<01:11, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [08:51<01:11, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [08:51<01:07, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [08:51<01:18, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [08:51<01:13, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [08:51<01:10, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [08:51<01:09, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:52<01:05, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:52<01:00, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:52<00:56, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:52<00:53, 2.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [08:52<00:52, 2.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [08:52<01:04, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [08:52<01:10, 1.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [08:52<01:02, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [08:52<00:57, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [08:53<00:54, 2.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [08:53<00:51, 2.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [08:53<00:51, 2.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [08:53<00:52, 2.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [08:53<00:55, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:53<01:10, 1.92MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:53<00:59, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:53<01:03, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:54<00:59, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [08:54<01:01, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [08:54<00:58, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [08:54<00:54, 2.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [08:54<00:51, 2.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [08:54<00:52, 2.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [08:54<01:07, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [08:54<01:00, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [08:54<00:55, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [08:55<00:50, 2.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [08:55<00:50, 2.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [08:55<00:49, 2.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [08:55<00:48, 2.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [08:55<00:49, 2.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [08:55<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:55<01:06, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:55<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:55<00:58, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [08:56<00:52, 2.48MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [08:56<00:50, 2.58MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [08:56<00:48, 2.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [08:56<00:45, 2.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [08:56<00:44, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [08:56<00:48, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [08:57<03:52, 551kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:57<03:01, 703kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:57<02:27, 861kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:57<01:54, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:57<01:41, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [08:57<01:25, 1.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [08:58<01:12, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [08:58<01:04, 1.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [08:58<00:58, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [08:58<00:51, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [08:58<01:01, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [08:58<01:11, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [08:58<01:00, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [08:58<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [08:59<00:52, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [09:03<23:01, 89.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [09:03<18:16, 113kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [09:03<13:10, 156kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [09:03<09:37, 213kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [09:03<07:06, 288kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [09:03<05:09, 395kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [09:03<03:47, 537kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [09:04<03:05, 658kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [09:04<02:23, 849kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [09:04<01:51, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [09:04<01:31, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [09:04<01:18, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [09:04<01:32, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [09:04<01:28, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [09:04<01:13, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:04<01:00, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<01:43, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<01:44, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<01:42, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<01:34, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [09:05<02:07, 929kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [09:05<01:47, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [09:05<01:26, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [09:05<01:12, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [09:06<01:04, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [09:06<00:55, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [09:06<00:49, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [09:06<00:45, 2.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [09:06<00:42, 2.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [09:06<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [09:06<00:57, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [09:06<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [09:07<01:10, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [09:07<01:01, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [09:07<00:53, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [09:07<00:50, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [09:07<00:52, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [09:07<00:59, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [09:07<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [09:07<00:44, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [09:08<00:46, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [09:08<00:47, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [09:08<00:44, 2.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [09:08<00:40, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:08<00:38, 2.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:08<00:45, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:08<00:51, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:09<26:37, 69.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:09<19:47, 93.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:10<14:32, 127kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:10<10:54, 169kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:10<08:32, 216kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [09:10<06:44, 274kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:10<05:28, 337kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:10<05:25, 340kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:10<04:18, 428kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:10<03:42, 495kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:10<03:05, 595kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:11<02:42, 676kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:11<02:25, 759kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:11<02:08, 855kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:11<01:58, 923kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [09:11<01:53, 964kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:11<02:29, 732kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:11<02:07, 857kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:11<02:13, 819kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:11<02:20, 776kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<02:14, 810kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<02:06, 865kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<02:04, 876kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<02:01, 895kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<01:59, 913kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [09:12<02:03, 879kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [09:12<01:46, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [09:12<01:39, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [09:12<01:24, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [09:12<01:09, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [09:13<00:55, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [09:13<00:47, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [09:13<01:32, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [09:13<01:27, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [09:13<02:07, 838kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [09:13<01:47, 995kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [09:13<01:45, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [09:13<01:41, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [09:14<01:26, 1.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [09:14<01:15, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [09:14<01:04, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [09:14<00:55, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [09:14<00:44, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [09:14<00:52, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [09:14<00:48, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [09:14<00:45, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [09:14<00:37, 2.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [09:14<00:31, 3.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [09:15<03:23, 504kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:15<02:39, 643kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:15<02:25, 707kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:15<02:07, 804kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:15<02:04, 822kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:16<01:41, 1.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:16<01:30, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [09:16<01:23, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [09:16<01:14, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [09:16<01:14, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [09:16<00:55, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [09:16<00:45, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [09:17<00:38, 2.58MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [09:17<00:59, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [09:17<01:18, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [09:17<01:12, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [09:17<00:57, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [09:17<00:48, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [09:18<00:39, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [09:18<00:33, 2.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [09:18<00:29, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [09:18<00:25, 3.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [09:18<00:23, 4.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [09:21<04:16, 369kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [09:21<04:38, 339kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [09:21<03:37, 435kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [09:22<02:37, 595kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [09:22<01:53, 817kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [09:22<01:24, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [09:22<01:03, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [09:22<00:52, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [09:22<00:45, 2.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:23<01:15, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:23<01:19, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:23<01:24, 1.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:23<01:31, 987kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:24<01:26, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [09:24<01:11, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [09:24<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [09:24<00:42, 2.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [09:24<00:34, 2.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [09:24<00:31, 2.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [09:24<00:29, 2.91MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [09:24<00:25, 3.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [09:25<05:17, 272kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [09:25<03:57, 363kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [09:25<02:50, 503kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [09:25<02:04, 686kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [09:26<01:32, 919kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [09:26<01:09, 1.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [09:26<00:53, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [09:26<00:41, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [09:26<00:32, 2.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [09:27<03:10, 431kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [09:27<03:18, 413kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [09:27<02:35, 529kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [09:27<01:54, 715kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [09:27<01:24, 959kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [09:28<01:03, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [09:28<00:49, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [09:28<00:37, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [09:28<00:30, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [09:28<00:25, 3.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [09:29<02:43, 477kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [09:29<02:56, 443kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [09:29<02:13, 582kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [09:29<01:41, 765kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [09:29<01:21, 949kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [09:30<01:09, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [09:30<00:53, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [09:30<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [09:30<00:34, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [09:30<00:28, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [09:30<00:25, 2.94MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:30<00:24, 3.05MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:30<00:29, 2.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:30<00:31, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:31<03:41, 334kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:31<03:02, 405kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [09:31<02:29, 492kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:31<02:20, 526kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:31<02:12, 554kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:31<02:00, 608kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:32<01:41, 720kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:32<01:27, 840kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:32<01:17, 938kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:32<01:17, 942kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:32<01:11, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:32<02:22, 510kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [09:32<01:49, 663kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [09:32<01:20, 894kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [09:32<00:59, 1.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [09:33<00:45, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [09:33<00:33, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [09:33<00:46, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:33<00:47, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:33<00:40, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:33<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:33<00:26, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:34<00:24, 2.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [09:34<00:19, 3.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [09:34<00:16, 3.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [09:34<00:15, 4.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:35<02:40, 408kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:35<02:06, 517kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:35<02:18, 473kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:35<01:42, 638kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:35<01:15, 852kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [09:35<00:55, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [09:36<00:41, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [09:36<00:31, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [09:36<00:25, 2.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:37<00:45, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:37<00:50, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:37<00:51, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:37<00:43, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [09:37<00:35, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [09:37<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [09:38<00:21, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [09:38<00:17, 3.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [09:38<00:14, 3.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:39<00:41, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:39<00:50, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:39<00:51, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:39<00:44, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [09:39<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [09:39<00:28, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [09:40<00:26, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [09:40<00:25, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [09:40<00:23, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [09:40<00:20, 2.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [09:40<00:17, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [09:40<00:17, 3.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [09:40<00:16, 3.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [09:41<05:05, 174kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [09:41<03:37, 244kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [09:41<02:32, 343kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [09:41<01:52, 461kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [09:41<01:19, 641kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [09:41<00:58, 872kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [09:41<00:42, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [09:42<00:30, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [09:43<19:18, 42.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [09:43<13:44, 59.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:43<09:36, 84.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:43<06:41, 119kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [09:43<04:40, 169kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [09:43<03:14, 239kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [09:43<02:16, 336kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [09:45<01:52, 398kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [09:45<01:36, 466kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [09:45<01:23, 538kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [09:45<01:22, 544kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:45<01:02, 713kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:45<00:47, 933kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:45<00:36, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [09:45<00:29, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [09:46<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [09:46<00:18, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [09:46<00:14, 2.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [09:47<00:24, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [09:47<00:21, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [09:47<00:15, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [09:47<00:12, 3.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [09:47<00:09, 4.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:49<00:56, 645kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:49<00:57, 633kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:49<00:44, 817kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [09:49<00:31, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [09:49<00:22, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [09:51<00:24, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [09:51<00:23, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [09:51<00:17, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [09:51<00:13, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [09:51<00:09, 3.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:53<00:26, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:53<00:28, 1.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:53<00:22, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [09:53<00:15, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [09:53<00:11, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [09:53<00:08, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [09:55<00:25, 950kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [09:55<00:21, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [09:55<00:15, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [09:55<00:10, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [09:57<00:20, 991kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [09:57<00:20, 983kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [09:57<00:15, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [09:57<00:10, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [09:57<00:06, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [10:01<01:25, 185kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [10:02<01:14, 213kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [10:02<00:56, 279kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [10:02<00:40, 380kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [10:02<00:28, 530kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [10:02<00:17, 747kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [10:03<00:14, 801kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [10:04<00:11, 982kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [10:04<00:08, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [10:04<00:05, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [10:04<00:04, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [10:04<00:02, 2.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [10:04<00:02, 3.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [10:05<00:24, 304kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [10:06<00:17, 418kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [10:06<00:09, 590kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [10:06<00:04, 830kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [10:11<00:14, 238kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [10:11<00:13, 251kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [10:11<00:09, 330kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [10:11<00:05, 459kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [10:12<00:00, 649kB/s].vector_cache/glove.6B.zip: 862MB [10:12, 1.41MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:54:36,  6.98it/s]  0%|          | 983/400000 [00:00<11:06:47,  9.97it/s]  0%|          | 1947/400000 [00:00<7:45:49, 14.24it/s]  1%|          | 2965/400000 [00:00<5:25:26, 20.33it/s]  1%|          | 3920/400000 [00:00<3:47:28, 29.02it/s]  1%|          | 4929/400000 [00:00<2:39:01, 41.41it/s]  1%|▏         | 5861/400000 [00:00<1:51:15, 59.04it/s]  2%|▏         | 6734/400000 [00:00<1:17:56, 84.10it/s]  2%|▏         | 7670/400000 [00:00<54:38, 119.68it/s]   2%|▏         | 8589/400000 [00:01<38:22, 170.02it/s]  2%|▏         | 9545/400000 [00:01<26:59, 241.05it/s]  3%|▎         | 10493/400000 [00:01<19:03, 340.65it/s]  3%|▎         | 11535/400000 [00:01<13:29, 479.91it/s]  3%|▎         | 12545/400000 [00:01<09:36, 671.89it/s]  3%|▎         | 13519/400000 [00:01<06:55, 931.01it/s]  4%|▎         | 14522/400000 [00:01<05:01, 1279.09it/s]  4%|▍         | 15528/400000 [00:01<03:41, 1732.85it/s]  4%|▍         | 16511/400000 [00:01<02:47, 2284.20it/s]  4%|▍         | 17466/400000 [00:01<02:09, 2949.19it/s]  5%|▍         | 18427/400000 [00:02<01:42, 3723.13it/s]  5%|▍         | 19454/400000 [00:02<01:22, 4602.86it/s]  5%|▌         | 20443/400000 [00:02<01:09, 5481.77it/s]  5%|▌         | 21461/400000 [00:02<00:59, 6362.41it/s]  6%|▌         | 22528/400000 [00:02<00:52, 7238.07it/s]  6%|▌         | 23552/400000 [00:02<00:47, 7934.36it/s]  6%|▌         | 24568/400000 [00:02<00:44, 8473.68it/s]  6%|▋         | 25581/400000 [00:02<00:43, 8547.99it/s]  7%|▋         | 26552/400000 [00:02<00:42, 8751.46it/s]  7%|▋         | 27511/400000 [00:02<00:41, 8985.76it/s]  7%|▋         | 28506/400000 [00:03<00:40, 9252.03it/s]  7%|▋         | 29496/400000 [00:03<00:39, 9437.16it/s]  8%|▊         | 30472/400000 [00:03<00:39, 9417.73it/s]  8%|▊         | 31450/400000 [00:03<00:38, 9520.20it/s]  8%|▊         | 32455/400000 [00:03<00:37, 9672.66it/s]  8%|▊         | 33486/400000 [00:03<00:37, 9855.23it/s]  9%|▊         | 34549/400000 [00:03<00:36, 10074.94it/s]  9%|▉         | 35564/400000 [00:03<00:37, 9838.42it/s]   9%|▉         | 36555/400000 [00:03<00:37, 9582.95it/s]  9%|▉         | 37520/400000 [00:04<00:37, 9575.54it/s] 10%|▉         | 38515/400000 [00:04<00:37, 9682.93it/s] 10%|▉         | 39526/400000 [00:04<00:36, 9806.72it/s] 10%|█         | 40536/400000 [00:04<00:36, 9891.65it/s] 10%|█         | 41565/400000 [00:04<00:35, 10007.63it/s] 11%|█         | 42578/400000 [00:04<00:35, 10043.86it/s] 11%|█         | 43622/400000 [00:04<00:35, 10158.36it/s] 11%|█         | 44640/400000 [00:04<00:35, 10135.36it/s] 11%|█▏        | 45655/400000 [00:04<00:36, 9784.27it/s]  12%|█▏        | 46644/400000 [00:04<00:36, 9812.86it/s] 12%|█▏        | 47671/400000 [00:05<00:35, 9944.29it/s] 12%|█▏        | 48687/400000 [00:05<00:35, 10006.72it/s] 12%|█▏        | 49730/400000 [00:05<00:34, 10128.54it/s] 13%|█▎        | 50745/400000 [00:05<00:34, 10104.94it/s] 13%|█▎        | 51768/400000 [00:05<00:34, 10139.47it/s] 13%|█▎        | 52783/400000 [00:05<00:34, 10124.29it/s] 13%|█▎        | 53845/400000 [00:05<00:33, 10266.43it/s] 14%|█▎        | 54914/400000 [00:05<00:33, 10389.08it/s] 14%|█▍        | 55954/400000 [00:05<00:34, 9983.74it/s]  14%|█▍        | 56957/400000 [00:05<00:34, 9958.17it/s] 14%|█▍        | 57972/400000 [00:06<00:34, 10012.30it/s] 15%|█▍        | 58976/400000 [00:06<00:34, 9816.96it/s]  15%|█▍        | 59960/400000 [00:06<00:35, 9634.98it/s] 15%|█▌        | 60931/400000 [00:06<00:35, 9655.14it/s] 15%|█▌        | 61899/400000 [00:06<00:35, 9646.22it/s] 16%|█▌        | 62917/400000 [00:06<00:34, 9799.16it/s] 16%|█▌        | 63950/400000 [00:06<00:33, 9950.91it/s] 16%|█▌        | 64956/400000 [00:06<00:33, 9980.23it/s] 16%|█▋        | 65956/400000 [00:06<00:34, 9753.94it/s] 17%|█▋        | 66959/400000 [00:06<00:33, 9833.09it/s] 17%|█▋        | 67991/400000 [00:07<00:33, 9974.23it/s] 17%|█▋        | 69004/400000 [00:07<00:33, 10015.79it/s] 18%|█▊        | 70069/400000 [00:07<00:32, 10196.50it/s] 18%|█▊        | 71157/400000 [00:07<00:31, 10390.97it/s] 18%|█▊        | 72206/400000 [00:07<00:31, 10418.93it/s] 18%|█▊        | 73250/400000 [00:07<00:31, 10395.00it/s] 19%|█▊        | 74291/400000 [00:07<00:31, 10179.59it/s] 19%|█▉        | 75311/400000 [00:07<00:33, 9716.73it/s]  19%|█▉        | 76289/400000 [00:07<00:33, 9644.10it/s] 19%|█▉        | 77258/400000 [00:07<00:33, 9637.83it/s] 20%|█▉        | 78257/400000 [00:08<00:33, 9740.30it/s] 20%|█▉        | 79312/400000 [00:08<00:32, 9967.91it/s] 20%|██        | 80357/400000 [00:08<00:31, 10105.60it/s] 20%|██        | 81371/400000 [00:08<00:31, 10100.89it/s] 21%|██        | 82383/400000 [00:08<00:31, 10042.19it/s] 21%|██        | 83389/400000 [00:08<00:31, 9984.63it/s]  21%|██        | 84434/400000 [00:08<00:31, 10119.80it/s] 21%|██▏       | 85448/400000 [00:08<00:32, 9805.05it/s]  22%|██▏       | 86432/400000 [00:08<00:32, 9676.73it/s] 22%|██▏       | 87403/400000 [00:09<00:32, 9589.29it/s] 22%|██▏       | 88378/400000 [00:09<00:32, 9635.01it/s] 22%|██▏       | 89379/400000 [00:09<00:31, 9743.80it/s] 23%|██▎       | 90399/400000 [00:09<00:31, 9785.40it/s] 23%|██▎       | 91379/400000 [00:09<00:31, 9663.31it/s] 23%|██▎       | 92364/400000 [00:09<00:31, 9718.55it/s] 23%|██▎       | 93404/400000 [00:09<00:30, 9913.23it/s] 24%|██▎       | 94411/400000 [00:09<00:30, 9958.70it/s] 24%|██▍       | 95408/400000 [00:09<00:31, 9628.69it/s] 24%|██▍       | 96420/400000 [00:09<00:31, 9770.37it/s] 24%|██▍       | 97428/400000 [00:10<00:30, 9856.27it/s] 25%|██▍       | 98416/400000 [00:10<00:30, 9834.82it/s] 25%|██▍       | 99414/400000 [00:10<00:30, 9876.77it/s] 25%|██▌       | 100403/400000 [00:10<00:30, 9839.82it/s] 25%|██▌       | 101470/400000 [00:10<00:29, 10073.95it/s] 26%|██▌       | 102480/400000 [00:10<00:29, 10047.70it/s] 26%|██▌       | 103552/400000 [00:10<00:28, 10239.48it/s] 26%|██▌       | 104578/400000 [00:10<00:28, 10188.69it/s] 26%|██▋       | 105599/400000 [00:10<00:29, 9814.46it/s]  27%|██▋       | 106585/400000 [00:10<00:30, 9754.02it/s] 27%|██▋       | 107580/400000 [00:11<00:29, 9811.66it/s] 27%|██▋       | 108616/400000 [00:11<00:29, 9967.86it/s] 27%|██▋       | 109649/400000 [00:11<00:28, 10072.82it/s] 28%|██▊       | 110658/400000 [00:11<00:29, 9960.68it/s]  28%|██▊       | 111662/400000 [00:11<00:28, 9982.39it/s] 28%|██▊       | 112671/400000 [00:11<00:28, 10013.96it/s] 28%|██▊       | 113718/400000 [00:11<00:28, 10145.33it/s] 29%|██▊       | 114734/400000 [00:11<00:28, 9992.58it/s]  29%|██▉       | 115735/400000 [00:11<00:29, 9764.09it/s] 29%|██▉       | 116714/400000 [00:11<00:29, 9714.29it/s] 29%|██▉       | 117687/400000 [00:12<00:29, 9688.56it/s] 30%|██▉       | 118705/400000 [00:12<00:28, 9830.22it/s] 30%|██▉       | 119741/400000 [00:12<00:28, 9982.16it/s] 30%|███       | 120779/400000 [00:12<00:27, 10094.72it/s] 30%|███       | 121790/400000 [00:12<00:27, 9945.64it/s]  31%|███       | 122786/400000 [00:12<00:28, 9846.48it/s] 31%|███       | 123854/400000 [00:12<00:27, 10079.82it/s] 31%|███       | 124865/400000 [00:12<00:28, 9805.08it/s]  31%|███▏      | 125849/400000 [00:12<00:28, 9747.53it/s] 32%|███▏      | 126846/400000 [00:12<00:27, 9812.35it/s] 32%|███▏      | 127834/400000 [00:13<00:27, 9831.83it/s] 32%|███▏      | 128819/400000 [00:13<00:28, 9683.12it/s] 32%|███▏      | 129796/400000 [00:13<00:27, 9707.27it/s] 33%|███▎      | 130834/400000 [00:13<00:27, 9898.15it/s] 33%|███▎      | 131859/400000 [00:13<00:26, 10001.07it/s] 33%|███▎      | 132868/400000 [00:13<00:26, 10026.13it/s] 33%|███▎      | 133919/400000 [00:13<00:26, 10165.62it/s] 34%|███▎      | 134937/400000 [00:13<00:26, 9832.47it/s]  34%|███▍      | 135924/400000 [00:13<00:27, 9718.01it/s] 34%|███▍      | 136899/400000 [00:14<00:27, 9630.88it/s] 34%|███▍      | 137885/400000 [00:14<00:27, 9696.56it/s] 35%|███▍      | 138911/400000 [00:14<00:26, 9857.86it/s] 35%|███▍      | 139967/400000 [00:14<00:25, 10057.64it/s] 35%|███▌      | 141003/400000 [00:14<00:25, 10145.70it/s] 36%|███▌      | 142032/400000 [00:14<00:25, 10186.96it/s] 36%|███▌      | 143065/400000 [00:14<00:25, 10227.41it/s] 36%|███▌      | 144135/400000 [00:14<00:24, 10364.50it/s] 36%|███▋      | 145173/400000 [00:14<00:25, 9861.10it/s]  37%|███▋      | 146174/400000 [00:14<00:25, 9902.12it/s] 37%|███▋      | 147178/400000 [00:15<00:25, 9943.03it/s] 37%|███▋      | 148179/400000 [00:15<00:25, 9961.34it/s] 37%|███▋      | 149178/400000 [00:15<00:25, 9940.86it/s] 38%|███▊      | 150174/400000 [00:15<00:26, 9546.36it/s] 38%|███▊      | 151134/400000 [00:15<00:26, 9403.14it/s] 38%|███▊      | 152089/400000 [00:15<00:26, 9446.52it/s] 38%|███▊      | 153102/400000 [00:15<00:25, 9641.27it/s] 39%|███▊      | 154089/400000 [00:15<00:25, 9708.57it/s] 39%|███▉      | 155062/400000 [00:15<00:25, 9606.22it/s] 39%|███▉      | 156091/400000 [00:15<00:24, 9799.89it/s] 39%|███▉      | 157112/400000 [00:16<00:24, 9914.02it/s] 40%|███▉      | 158124/400000 [00:16<00:24, 9974.05it/s] 40%|███▉      | 159166/400000 [00:16<00:23, 10101.92it/s] 40%|████      | 160206/400000 [00:16<00:23, 10188.28it/s] 40%|████      | 161294/400000 [00:16<00:22, 10384.42it/s] 41%|████      | 162335/400000 [00:16<00:23, 10067.66it/s] 41%|████      | 163346/400000 [00:16<00:24, 9845.43it/s]  41%|████      | 164334/400000 [00:16<00:24, 9668.88it/s] 41%|████▏     | 165304/400000 [00:16<00:24, 9610.39it/s] 42%|████▏     | 166338/400000 [00:16<00:23, 9816.73it/s] 42%|████▏     | 167325/400000 [00:17<00:23, 9830.70it/s] 42%|████▏     | 168335/400000 [00:17<00:23, 9907.95it/s] 42%|████▏     | 169400/400000 [00:17<00:22, 10116.07it/s] 43%|████▎     | 170472/400000 [00:17<00:22, 10288.61it/s] 43%|████▎     | 171522/400000 [00:17<00:22, 10349.00it/s] 43%|████▎     | 172559/400000 [00:17<00:21, 10346.51it/s] 43%|████▎     | 173595/400000 [00:17<00:22, 10220.91it/s] 44%|████▎     | 174619/400000 [00:17<00:23, 9778.61it/s]  44%|████▍     | 175602/400000 [00:17<00:23, 9638.56it/s] 44%|████▍     | 176588/400000 [00:18<00:23, 9702.10it/s] 44%|████▍     | 177562/400000 [00:18<00:22, 9701.15it/s] 45%|████▍     | 178535/400000 [00:18<00:22, 9663.34it/s] 45%|████▍     | 179537/400000 [00:18<00:22, 9765.82it/s] 45%|████▌     | 180529/400000 [00:18<00:22, 9810.83it/s] 45%|████▌     | 181559/400000 [00:18<00:21, 9952.61it/s] 46%|████▌     | 182556/400000 [00:18<00:21, 9944.91it/s] 46%|████▌     | 183552/400000 [00:18<00:21, 9847.38it/s] 46%|████▌     | 184538/400000 [00:18<00:22, 9641.07it/s] 46%|████▋     | 185529/400000 [00:18<00:22, 9717.43it/s] 47%|████▋     | 186502/400000 [00:19<00:22, 9648.85it/s] 47%|████▋     | 187468/400000 [00:19<00:22, 9509.79it/s] 47%|████▋     | 188493/400000 [00:19<00:21, 9719.16it/s] 47%|████▋     | 189514/400000 [00:19<00:21, 9858.85it/s] 48%|████▊     | 190520/400000 [00:19<00:21, 9915.86it/s] 48%|████▊     | 191513/400000 [00:19<00:21, 9893.56it/s] 48%|████▊     | 192559/400000 [00:19<00:20, 10055.82it/s] 48%|████▊     | 193568/400000 [00:19<00:20, 10063.02it/s] 49%|████▊     | 194576/400000 [00:19<00:20, 9803.61it/s]  49%|████▉     | 195581/400000 [00:19<00:20, 9873.98it/s] 49%|████▉     | 196628/400000 [00:20<00:20, 10044.18it/s] 49%|████▉     | 197635/400000 [00:20<00:20, 9954.75it/s]  50%|████▉     | 198636/400000 [00:20<00:20, 9969.37it/s] 50%|████▉     | 199698/400000 [00:20<00:19, 10154.73it/s] 50%|█████     | 200723/400000 [00:20<00:19, 10182.17it/s] 50%|█████     | 201743/400000 [00:20<00:19, 9976.35it/s]  51%|█████     | 202781/400000 [00:20<00:19, 10091.58it/s] 51%|█████     | 203792/400000 [00:20<00:19, 9960.63it/s]  51%|█████     | 204790/400000 [00:20<00:20, 9633.44it/s] 51%|█████▏    | 205811/400000 [00:20<00:19, 9798.61it/s] 52%|█████▏    | 206807/400000 [00:21<00:19, 9843.87it/s] 52%|█████▏    | 207841/400000 [00:21<00:19, 9987.30it/s] 52%|█████▏    | 208875/400000 [00:21<00:18, 10090.36it/s] 52%|█████▏    | 209892/400000 [00:21<00:18, 10113.26it/s] 53%|█████▎    | 210905/400000 [00:21<00:18, 9988.68it/s]  53%|█████▎    | 211910/400000 [00:21<00:18, 10006.61it/s] 53%|█████▎    | 212916/400000 [00:21<00:18, 10020.82it/s] 53%|█████▎    | 213919/400000 [00:21<00:19, 9764.89it/s]  54%|█████▎    | 214898/400000 [00:21<00:19, 9584.09it/s] 54%|█████▍    | 215870/400000 [00:21<00:19, 9622.32it/s] 54%|█████▍    | 216896/400000 [00:22<00:18, 9803.75it/s] 54%|█████▍    | 217951/400000 [00:22<00:18, 10015.83it/s] 55%|█████▍    | 218983/400000 [00:22<00:17, 10103.79it/s] 55%|█████▌    | 220000/400000 [00:22<00:17, 10122.92it/s] 55%|█████▌    | 221014/400000 [00:22<00:18, 9857.29it/s]  56%|█████▌    | 222003/400000 [00:22<00:18, 9698.50it/s] 56%|█████▌    | 222976/400000 [00:22<00:18, 9542.86it/s] 56%|█████▌    | 223933/400000 [00:22<00:18, 9400.39it/s] 56%|█████▌    | 224896/400000 [00:22<00:18, 9465.71it/s] 56%|█████▋    | 225889/400000 [00:23<00:18, 9599.03it/s] 57%|█████▋    | 226851/400000 [00:23<00:18, 9563.81it/s] 57%|█████▋    | 227809/400000 [00:23<00:18, 9455.50it/s] 57%|█████▋    | 228756/400000 [00:23<00:18, 9273.55it/s] 57%|█████▋    | 229707/400000 [00:23<00:18, 9341.37it/s] 58%|█████▊    | 230680/400000 [00:23<00:17, 9452.04it/s] 58%|█████▊    | 231701/400000 [00:23<00:17, 9666.99it/s] 58%|█████▊    | 232725/400000 [00:23<00:17, 9829.11it/s] 58%|█████▊    | 233710/400000 [00:23<00:17, 9277.05it/s] 59%|█████▊    | 234672/400000 [00:23<00:17, 9374.57it/s] 59%|█████▉    | 235648/400000 [00:24<00:17, 9485.19it/s] 59%|█████▉    | 236659/400000 [00:24<00:16, 9663.73it/s] 59%|█████▉    | 237630/400000 [00:24<00:16, 9676.08it/s] 60%|█████▉    | 238645/400000 [00:24<00:16, 9812.50it/s] 60%|█████▉    | 239666/400000 [00:24<00:16, 9927.00it/s] 60%|██████    | 240681/400000 [00:24<00:15, 9991.42it/s] 60%|██████    | 241689/400000 [00:24<00:15, 10014.34it/s] 61%|██████    | 242692/400000 [00:24<00:16, 9677.22it/s]  61%|██████    | 243663/400000 [00:24<00:16, 9536.48it/s] 61%|██████    | 244620/400000 [00:24<00:16, 9413.90it/s] 61%|██████▏   | 245574/400000 [00:25<00:16, 9449.28it/s] 62%|██████▏   | 246575/400000 [00:25<00:15, 9608.16it/s] 62%|██████▏   | 247538/400000 [00:25<00:16, 9386.55it/s] 62%|██████▏   | 248480/400000 [00:25<00:16, 9299.07it/s] 62%|██████▏   | 249511/400000 [00:25<00:15, 9579.58it/s] 63%|██████▎   | 250556/400000 [00:25<00:15, 9823.89it/s] 63%|██████▎   | 251628/400000 [00:25<00:14, 10074.39it/s] 63%|██████▎   | 252640/400000 [00:25<00:15, 9677.89it/s]  63%|██████▎   | 253615/400000 [00:25<00:15, 9645.70it/s] 64%|██████▎   | 254641/400000 [00:26<00:14, 9821.65it/s] 64%|██████▍   | 255667/400000 [00:26<00:14, 9947.99it/s] 64%|██████▍   | 256665/400000 [00:26<00:14, 9942.70it/s] 64%|██████▍   | 257662/400000 [00:26<00:14, 9937.81it/s] 65%|██████▍   | 258701/400000 [00:26<00:14, 10067.93it/s] 65%|██████▍   | 259730/400000 [00:26<00:13, 10107.92it/s] 65%|██████▌   | 260742/400000 [00:26<00:13, 10043.91it/s] 65%|██████▌   | 261771/400000 [00:26<00:13, 10115.51it/s] 66%|██████▌   | 262784/400000 [00:26<00:14, 9755.40it/s]  66%|██████▌   | 263789/400000 [00:26<00:13, 9840.01it/s] 66%|██████▌   | 264815/400000 [00:27<00:13, 9961.52it/s] 66%|██████▋   | 265814/400000 [00:27<00:13, 9909.37it/s] 67%|██████▋   | 266848/400000 [00:27<00:13, 10034.65it/s] 67%|██████▋   | 267853/400000 [00:27<00:13, 9993.02it/s]  67%|██████▋   | 268899/400000 [00:27<00:12, 10127.19it/s] 67%|██████▋   | 269932/400000 [00:27<00:12, 10186.22it/s] 68%|██████▊   | 270958/400000 [00:27<00:12, 10205.92it/s] 68%|██████▊   | 271987/400000 [00:27<00:12, 10227.86it/s] 68%|██████▊   | 273011/400000 [00:27<00:12, 9807.05it/s]  68%|██████▊   | 273996/400000 [00:27<00:13, 9670.42it/s] 69%|██████▊   | 274980/400000 [00:28<00:12, 9718.77it/s] 69%|██████▉   | 275996/400000 [00:28<00:12, 9846.20it/s] 69%|██████▉   | 277027/400000 [00:28<00:12, 9980.53it/s] 70%|██████▉   | 278050/400000 [00:28<00:12, 10053.12it/s] 70%|██████▉   | 279117/400000 [00:28<00:11, 10229.69it/s] 70%|███████   | 280142/400000 [00:28<00:11, 10027.83it/s] 70%|███████   | 281185/400000 [00:28<00:11, 10143.07it/s] 71%|███████   | 282204/400000 [00:28<00:11, 10152.37it/s] 71%|███████   | 283221/400000 [00:28<00:12, 9729.35it/s]  71%|███████   | 284238/400000 [00:28<00:11, 9855.83it/s] 71%|███████▏  | 285228/400000 [00:29<00:11, 9782.91it/s] 72%|███████▏  | 286209/400000 [00:29<00:11, 9575.61it/s] 72%|███████▏  | 287253/400000 [00:29<00:11, 9817.12it/s] 72%|███████▏  | 288299/400000 [00:29<00:11, 10000.55it/s] 72%|███████▏  | 289320/400000 [00:29<00:11, 10061.30it/s] 73%|███████▎  | 290329/400000 [00:29<00:11, 9955.36it/s]  73%|███████▎  | 291354/400000 [00:29<00:10, 10041.23it/s] 73%|███████▎  | 292360/400000 [00:29<00:10, 9885.29it/s]  73%|███████▎  | 293351/400000 [00:29<00:10, 9757.61it/s] 74%|███████▎  | 294355/400000 [00:29<00:10, 9840.33it/s] 74%|███████▍  | 295390/400000 [00:30<00:10, 9986.32it/s] 74%|███████▍  | 296391/400000 [00:30<00:10, 9977.75it/s] 74%|███████▍  | 297482/400000 [00:30<00:10, 10239.14it/s] 75%|███████▍  | 298509/400000 [00:30<00:09, 10243.17it/s] 75%|███████▍  | 299535/400000 [00:30<00:10, 9978.91it/s]  75%|███████▌  | 300553/400000 [00:30<00:09, 10037.58it/s] 75%|███████▌  | 301622/400000 [00:30<00:09, 10224.12it/s] 76%|███████▌  | 302647/400000 [00:30<00:09, 9764.05it/s]  76%|███████▌  | 303630/400000 [00:30<00:09, 9723.20it/s] 76%|███████▌  | 304607/400000 [00:31<00:09, 9667.14it/s] 76%|███████▋  | 305602/400000 [00:31<00:09, 9749.51it/s] 77%|███████▋  | 306581/400000 [00:31<00:09, 9759.36it/s] 77%|███████▋  | 307559/400000 [00:31<00:09, 9590.29it/s] 77%|███████▋  | 308520/400000 [00:31<00:09, 9403.05it/s] 77%|███████▋  | 309495/400000 [00:31<00:09, 9502.79it/s] 78%|███████▊  | 310458/400000 [00:31<00:09, 9537.65it/s] 78%|███████▊  | 311413/400000 [00:31<00:09, 9246.41it/s] 78%|███████▊  | 312341/400000 [00:31<00:09, 9167.58it/s] 78%|███████▊  | 313283/400000 [00:31<00:09, 9239.86it/s] 79%|███████▊  | 314269/400000 [00:32<00:09, 9417.53it/s] 79%|███████▉  | 315279/400000 [00:32<00:08, 9611.55it/s] 79%|███████▉  | 316268/400000 [00:32<00:08, 9690.61it/s] 79%|███████▉  | 317251/400000 [00:32<00:08, 9730.81it/s] 80%|███████▉  | 318265/400000 [00:32<00:08, 9846.84it/s] 80%|███████▉  | 319322/400000 [00:32<00:08, 10050.52it/s] 80%|████████  | 320356/400000 [00:32<00:07, 10135.31it/s] 80%|████████  | 321372/400000 [00:32<00:07, 9831.81it/s]  81%|████████  | 322359/400000 [00:32<00:08, 9331.72it/s] 81%|████████  | 323300/400000 [00:32<00:08, 9352.31it/s] 81%|████████  | 324289/400000 [00:33<00:07, 9506.19it/s] 81%|████████▏ | 325297/400000 [00:33<00:07, 9669.36it/s] 82%|████████▏ | 326299/400000 [00:33<00:07, 9770.17it/s] 82%|████████▏ | 327327/400000 [00:33<00:07, 9917.08it/s] 82%|████████▏ | 328363/400000 [00:33<00:07, 10044.07it/s] 82%|████████▏ | 329370/400000 [00:33<00:07, 9992.96it/s]  83%|████████▎ | 330381/400000 [00:33<00:06, 10025.20it/s] 83%|████████▎ | 331385/400000 [00:33<00:06, 9809.49it/s]  83%|████████▎ | 332368/400000 [00:33<00:07, 9590.52it/s] 83%|████████▎ | 333330/400000 [00:33<00:06, 9541.81it/s] 84%|████████▎ | 334306/400000 [00:34<00:06, 9605.43it/s] 84%|████████▍ | 335308/400000 [00:34<00:06, 9724.26it/s] 84%|████████▍ | 336282/400000 [00:34<00:06, 9645.39it/s] 84%|████████▍ | 337248/400000 [00:34<00:06, 9636.36it/s] 85%|████████▍ | 338217/400000 [00:34<00:06, 9651.15it/s] 85%|████████▍ | 339183/400000 [00:34<00:06, 9608.30it/s] 85%|████████▌ | 340171/400000 [00:34<00:06, 9686.64it/s] 85%|████████▌ | 341141/400000 [00:34<00:06, 9385.85it/s] 86%|████████▌ | 342082/400000 [00:34<00:06, 9153.67it/s] 86%|████████▌ | 343001/400000 [00:35<00:06, 9114.19it/s] 86%|████████▌ | 343936/400000 [00:35<00:06, 9181.43it/s] 86%|████████▌ | 344929/400000 [00:35<00:05, 9393.17it/s] 86%|████████▋ | 345910/400000 [00:35<00:05, 9514.02it/s] 87%|████████▋ | 346864/400000 [00:35<00:05, 9482.06it/s] 87%|████████▋ | 347827/400000 [00:35<00:05, 9524.54it/s] 87%|████████▋ | 348795/400000 [00:35<00:05, 9569.90it/s] 87%|████████▋ | 349755/400000 [00:35<00:05, 9576.51it/s] 88%|████████▊ | 350714/400000 [00:35<00:05, 9343.69it/s] 88%|████████▊ | 351650/400000 [00:35<00:05, 9294.61it/s] 88%|████████▊ | 352581/400000 [00:36<00:05, 9161.45it/s] 88%|████████▊ | 353523/400000 [00:36<00:05, 9234.49it/s] 89%|████████▊ | 354450/400000 [00:36<00:04, 9244.16it/s] 89%|████████▉ | 355516/400000 [00:36<00:04, 9626.82it/s] 89%|████████▉ | 356590/400000 [00:36<00:04, 9935.73it/s] 89%|████████▉ | 357591/400000 [00:36<00:04, 9955.89it/s] 90%|████████▉ | 358591/400000 [00:36<00:04, 9871.43it/s] 90%|████████▉ | 359582/400000 [00:36<00:04, 9473.86it/s] 90%|█████████ | 360535/400000 [00:36<00:04, 9284.99it/s] 90%|█████████ | 361469/400000 [00:36<00:04, 9172.91it/s] 91%|█████████ | 362422/400000 [00:37<00:04, 9274.91it/s] 91%|█████████ | 363418/400000 [00:37<00:03, 9467.80it/s] 91%|█████████ | 364428/400000 [00:37<00:03, 9647.44it/s] 91%|█████████▏| 365456/400000 [00:37<00:03, 9827.56it/s] 92%|█████████▏| 366442/400000 [00:37<00:03, 9692.47it/s] 92%|█████████▏| 367414/400000 [00:37<00:03, 9458.17it/s] 92%|█████████▏| 368363/400000 [00:37<00:03, 9422.29it/s] 92%|█████████▏| 369308/400000 [00:37<00:03, 9287.42it/s] 93%|█████████▎| 370239/400000 [00:37<00:03, 9214.12it/s] 93%|█████████▎| 371162/400000 [00:37<00:03, 9201.59it/s] 93%|█████████▎| 372171/400000 [00:38<00:02, 9449.70it/s] 93%|█████████▎| 373185/400000 [00:38<00:02, 9645.39it/s] 94%|█████████▎| 374169/400000 [00:38<00:02, 9701.17it/s] 94%|█████████▍| 375178/400000 [00:38<00:02, 9814.63it/s] 94%|█████████▍| 376191/400000 [00:38<00:02, 9904.93it/s] 94%|█████████▍| 377220/400000 [00:38<00:02, 10017.26it/s] 95%|█████████▍| 378283/400000 [00:38<00:02, 10193.52it/s] 95%|█████████▍| 379304/400000 [00:38<00:02, 9832.98it/s]  95%|█████████▌| 380292/400000 [00:38<00:02, 9789.77it/s] 95%|█████████▌| 381303/400000 [00:39<00:01, 9880.61it/s] 96%|█████████▌| 382299/400000 [00:39<00:01, 9901.31it/s] 96%|█████████▌| 383291/400000 [00:39<00:01, 9707.63it/s] 96%|█████████▌| 384329/400000 [00:39<00:01, 9898.97it/s] 96%|█████████▋| 385397/400000 [00:39<00:01, 10118.60it/s] 97%|█████████▋| 386412/400000 [00:39<00:01, 10047.77it/s] 97%|█████████▋| 387419/400000 [00:39<00:01, 10032.44it/s] 97%|█████████▋| 388436/400000 [00:39<00:01, 10070.00it/s] 97%|█████████▋| 389445/400000 [00:39<00:01, 9692.51it/s]  98%|█████████▊| 390419/400000 [00:39<00:00, 9594.54it/s] 98%|█████████▊| 391382/400000 [00:40<00:00, 9537.35it/s] 98%|█████████▊| 392343/400000 [00:40<00:00, 9557.38it/s] 98%|█████████▊| 393318/400000 [00:40<00:00, 9613.96it/s] 99%|█████████▊| 394293/400000 [00:40<00:00, 9653.48it/s] 99%|█████████▉| 395286/400000 [00:40<00:00, 9734.40it/s] 99%|█████████▉| 396261/400000 [00:40<00:00, 9704.32it/s] 99%|█████████▉| 397250/400000 [00:40<00:00, 9757.08it/s]100%|█████████▉| 398227/400000 [00:40<00:00, 9596.09it/s]100%|█████████▉| 399188/400000 [00:40<00:00, 9381.21it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9770.21it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe60d8890f0>, <torchtext.data.dataset.TabularDataset object at 0x7fe60d889208>, <torchtext.vocab.Vocab object at 0x7fe6fbf7aba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 30.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 56.67 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.82 file/s]2020-06-17 12:21:47.130309: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 12:21:47.134775: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095210000 Hz
2020-06-17 12:21:47.134951: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ec362ad6f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 12:21:47.134965: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3727360/9912422 [00:00<00:00, 30511530.29it/s]9920512it [00:00, 27583693.57it/s]                             
0it [00:00, ?it/s]32768it [00:00, 592192.32it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472538.06it/s]1654784it [00:00, 11570709.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 181988.21it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
