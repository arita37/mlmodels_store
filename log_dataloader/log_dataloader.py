
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f706d1e5598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f706d1e5598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f70d7e95488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f70d7e95488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f70f70b3ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f70f70b3ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70851cf158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70851cf158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70851cf158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3866624/9912422 [00:00<00:00, 38408225.76it/s]9920512it [00:00, 33910280.15it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1030640.13it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457158.64it/s]1654784it [00:00, 11490105.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 175047.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f706c41f6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f708337b978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f70851c6d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f70851c6d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f70851c6d90> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:53,  2.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:53,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  4.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:25,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:18,  7.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:18,  7.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13, 10.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:13, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:13, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13, 10.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:10, 12.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:10, 12.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:10, 12.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:10, 12.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:09, 12.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:09, 12.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 16.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 16.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 16.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 16.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 16.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:07, 16.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 20.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 24.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 24.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 24.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 24.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 24.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 24.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 27.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 27.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 27.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 27.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 27.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 27.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 30.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 32.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 32.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 32.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 32.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 32.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 37.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 37.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 37.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 37.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 37.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 37.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 40.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 40.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 40.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 40.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 42.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 41.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 42.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 42.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 42.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 42.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 42.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 42.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 43.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 43.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 43.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 43.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 43.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 43.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 43.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 44.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 44.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 44.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 44.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 44.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 44.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 43.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 43.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 43.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 43.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 43.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 43.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 44.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 44.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 44.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 44.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 44.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 44.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 44.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 44.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 44.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 44.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 44.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 44.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 44.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 44.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 44.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 44.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 44.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 44.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 44.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 44.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 44.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 44.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 44.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 44.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 44.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 44.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 44.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 44.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 44.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 44.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 43.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 43.74 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 46.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.12s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 46.89 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.07s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 46.89 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.07s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.07s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 26.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.07s/ url]
0 examples [00:00, ? examples/s]2020-08-12 12:06:35.889631: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-12 12:06:35.934776: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-12 12:06:35.935010: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5606881ebb10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-12 12:06:35.935031: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.27 examples/s]112 examples [00:00,  3.24 examples/s]225 examples [00:00,  4.62 examples/s]336 examples [00:00,  6.59 examples/s]453 examples [00:00,  9.40 examples/s]572 examples [00:00, 13.38 examples/s]687 examples [00:01, 19.02 examples/s]801 examples [00:01, 26.97 examples/s]907 examples [00:01, 38.11 examples/s]1020 examples [00:01, 53.66 examples/s]1133 examples [00:01, 75.12 examples/s]1256 examples [00:01, 104.56 examples/s]1375 examples [00:01, 143.94 examples/s]1490 examples [00:01, 194.58 examples/s]1607 examples [00:01, 259.36 examples/s]1725 examples [00:01, 338.53 examples/s]1844 examples [00:02, 431.05 examples/s]1962 examples [00:02, 532.25 examples/s]2079 examples [00:02, 631.63 examples/s]2198 examples [00:02, 735.09 examples/s]2315 examples [00:02, 807.59 examples/s]2428 examples [00:02, 861.07 examples/s]2543 examples [00:02, 930.43 examples/s]2658 examples [00:02, 985.12 examples/s]2771 examples [00:02, 1013.14 examples/s]2882 examples [00:02, 1030.47 examples/s]2999 examples [00:03, 1067.15 examples/s]3118 examples [00:03, 1098.63 examples/s]3232 examples [00:03, 1106.74 examples/s]3358 examples [00:03, 1146.64 examples/s]3478 examples [00:03, 1160.26 examples/s]3597 examples [00:03, 1166.41 examples/s]3717 examples [00:03, 1173.39 examples/s]3839 examples [00:03, 1185.89 examples/s]3959 examples [00:03, 1184.24 examples/s]4078 examples [00:04, 1167.55 examples/s]4198 examples [00:04, 1177.00 examples/s]4321 examples [00:04, 1191.13 examples/s]4445 examples [00:04, 1204.17 examples/s]4566 examples [00:04, 1198.30 examples/s]4686 examples [00:04, 1186.57 examples/s]4805 examples [00:04, 1179.50 examples/s]4926 examples [00:04, 1187.00 examples/s]5045 examples [00:04, 1182.02 examples/s]5166 examples [00:04, 1189.77 examples/s]5286 examples [00:05, 1183.60 examples/s]5405 examples [00:05, 1174.74 examples/s]5531 examples [00:05, 1196.52 examples/s]5652 examples [00:05, 1199.20 examples/s]5773 examples [00:05, 1192.65 examples/s]5893 examples [00:05, 1173.29 examples/s]6013 examples [00:05, 1180.83 examples/s]6132 examples [00:05, 1171.10 examples/s]6250 examples [00:05, 1165.70 examples/s]6373 examples [00:05, 1183.42 examples/s]6492 examples [00:06, 1171.12 examples/s]6613 examples [00:06, 1180.39 examples/s]6732 examples [00:06, 1179.57 examples/s]6858 examples [00:06, 1201.54 examples/s]6981 examples [00:06, 1208.24 examples/s]7102 examples [00:06, 1194.11 examples/s]7222 examples [00:06, 1146.12 examples/s]7338 examples [00:06, 1139.13 examples/s]7453 examples [00:06, 1140.19 examples/s]7573 examples [00:06, 1156.44 examples/s]7690 examples [00:07, 1159.59 examples/s]7809 examples [00:07, 1167.47 examples/s]7930 examples [00:07, 1178.68 examples/s]8048 examples [00:07, 1161.27 examples/s]8165 examples [00:07, 1154.48 examples/s]8286 examples [00:07, 1168.76 examples/s]8405 examples [00:07, 1173.23 examples/s]8523 examples [00:07, 1169.34 examples/s]8640 examples [00:07, 1165.81 examples/s]8757 examples [00:07, 1150.13 examples/s]8873 examples [00:08, 1127.96 examples/s]8988 examples [00:08, 1132.91 examples/s]9111 examples [00:08, 1158.94 examples/s]9231 examples [00:08, 1170.44 examples/s]9349 examples [00:08, 1169.27 examples/s]9467 examples [00:08, 1142.27 examples/s]9584 examples [00:08, 1149.57 examples/s]9700 examples [00:08, 1140.82 examples/s]9815 examples [00:08, 1134.95 examples/s]9929 examples [00:09, 1136.02 examples/s]10043 examples [00:09, 1080.48 examples/s]10164 examples [00:09, 1114.65 examples/s]10286 examples [00:09, 1141.96 examples/s]10405 examples [00:09, 1153.54 examples/s]10527 examples [00:09, 1169.79 examples/s]10647 examples [00:09, 1177.76 examples/s]10766 examples [00:09, 1175.32 examples/s]10884 examples [00:09, 1161.49 examples/s]11002 examples [00:09, 1165.88 examples/s]11119 examples [00:10, 1164.95 examples/s]11236 examples [00:10, 1163.79 examples/s]11356 examples [00:10, 1172.62 examples/s]11474 examples [00:10, 1168.76 examples/s]11594 examples [00:10, 1175.58 examples/s]11712 examples [00:10, 1168.44 examples/s]11829 examples [00:10, 1142.31 examples/s]11948 examples [00:10, 1153.59 examples/s]12064 examples [00:10, 1138.36 examples/s]12183 examples [00:10, 1152.75 examples/s]12299 examples [00:11, 1146.28 examples/s]12414 examples [00:11, 1141.80 examples/s]12529 examples [00:11, 1131.20 examples/s]12646 examples [00:11, 1140.57 examples/s]12766 examples [00:11, 1155.09 examples/s]12886 examples [00:11, 1165.94 examples/s]13003 examples [00:11, 1155.35 examples/s]13119 examples [00:11, 1134.26 examples/s]13233 examples [00:11, 1129.73 examples/s]13350 examples [00:11, 1139.10 examples/s]13469 examples [00:12, 1153.30 examples/s]13585 examples [00:12, 1153.38 examples/s]13701 examples [00:12, 1149.53 examples/s]13817 examples [00:12, 1145.26 examples/s]13937 examples [00:12, 1160.81 examples/s]14054 examples [00:12, 1163.37 examples/s]14172 examples [00:12, 1167.72 examples/s]14290 examples [00:12, 1168.78 examples/s]14412 examples [00:12, 1181.30 examples/s]14531 examples [00:12, 1180.69 examples/s]14650 examples [00:13, 1172.83 examples/s]14768 examples [00:13, 1168.91 examples/s]14885 examples [00:13, 1143.56 examples/s]15003 examples [00:13, 1151.43 examples/s]15123 examples [00:13, 1162.85 examples/s]15240 examples [00:13, 1162.24 examples/s]15360 examples [00:13, 1170.26 examples/s]15478 examples [00:13, 1167.35 examples/s]15595 examples [00:13, 1150.28 examples/s]15711 examples [00:14, 1138.72 examples/s]15825 examples [00:14, 1127.71 examples/s]15938 examples [00:14, 1117.65 examples/s]16054 examples [00:14, 1128.48 examples/s]16170 examples [00:14, 1135.76 examples/s]16284 examples [00:14, 1130.18 examples/s]16400 examples [00:14, 1138.52 examples/s]16514 examples [00:14, 1137.87 examples/s]16633 examples [00:14, 1152.03 examples/s]16749 examples [00:14, 1153.50 examples/s]16866 examples [00:15, 1157.35 examples/s]16982 examples [00:15, 1153.33 examples/s]17098 examples [00:15, 1146.16 examples/s]17221 examples [00:15, 1166.98 examples/s]17339 examples [00:15, 1168.56 examples/s]17456 examples [00:15, 1165.15 examples/s]17573 examples [00:15, 1160.08 examples/s]17690 examples [00:15, 1160.67 examples/s]17807 examples [00:15, 1147.30 examples/s]17922 examples [00:15, 1136.70 examples/s]18040 examples [00:16, 1147.29 examples/s]18156 examples [00:16, 1149.71 examples/s]18274 examples [00:16, 1158.38 examples/s]18392 examples [00:16, 1161.99 examples/s]18509 examples [00:16, 1157.74 examples/s]18626 examples [00:16, 1159.13 examples/s]18742 examples [00:16, 1157.08 examples/s]18858 examples [00:16, 1127.45 examples/s]18971 examples [00:16, 1115.67 examples/s]19083 examples [00:16, 1111.22 examples/s]19199 examples [00:17, 1122.61 examples/s]19314 examples [00:17, 1128.80 examples/s]19431 examples [00:17, 1138.47 examples/s]19550 examples [00:17, 1152.83 examples/s]19667 examples [00:17, 1155.34 examples/s]19784 examples [00:17, 1159.48 examples/s]19901 examples [00:17, 1153.00 examples/s]20017 examples [00:17, 1090.09 examples/s]20134 examples [00:17, 1112.77 examples/s]20250 examples [00:17, 1124.50 examples/s]20365 examples [00:18, 1129.38 examples/s]20482 examples [00:18, 1138.50 examples/s]20597 examples [00:18, 1137.40 examples/s]20711 examples [00:18, 1128.43 examples/s]20825 examples [00:18, 1130.93 examples/s]20943 examples [00:18, 1144.26 examples/s]21060 examples [00:18, 1149.40 examples/s]21176 examples [00:18, 1150.24 examples/s]21294 examples [00:18, 1157.83 examples/s]21410 examples [00:18, 1151.95 examples/s]21533 examples [00:19, 1173.46 examples/s]21651 examples [00:19, 1171.01 examples/s]21769 examples [00:19, 1163.61 examples/s]21886 examples [00:19, 1158.40 examples/s]22002 examples [00:19, 1148.10 examples/s]22124 examples [00:19, 1167.31 examples/s]22241 examples [00:19, 1154.94 examples/s]22357 examples [00:19, 1151.75 examples/s]22473 examples [00:19, 1137.46 examples/s]22587 examples [00:20, 1131.70 examples/s]22703 examples [00:20, 1139.62 examples/s]22821 examples [00:20, 1150.13 examples/s]22940 examples [00:20, 1161.71 examples/s]23062 examples [00:20, 1177.79 examples/s]23180 examples [00:20, 1168.96 examples/s]23297 examples [00:20, 1164.23 examples/s]23414 examples [00:20, 1162.46 examples/s]23531 examples [00:20, 1154.56 examples/s]23647 examples [00:20, 1154.34 examples/s]23764 examples [00:21, 1155.96 examples/s]23884 examples [00:21, 1166.93 examples/s]24002 examples [00:21, 1169.69 examples/s]24123 examples [00:21, 1179.02 examples/s]24241 examples [00:21, 1173.09 examples/s]24359 examples [00:21, 1164.95 examples/s]24477 examples [00:21, 1167.19 examples/s]24594 examples [00:21, 1163.27 examples/s]24711 examples [00:21, 1142.80 examples/s]24827 examples [00:21, 1145.52 examples/s]24942 examples [00:22, 1143.02 examples/s]25060 examples [00:22, 1152.98 examples/s]25176 examples [00:22, 1138.00 examples/s]25290 examples [00:22, 1123.72 examples/s]25413 examples [00:22, 1152.80 examples/s]25530 examples [00:22, 1155.92 examples/s]25646 examples [00:22, 1156.92 examples/s]25762 examples [00:22, 1130.49 examples/s]25876 examples [00:22, 1129.02 examples/s]25995 examples [00:22, 1145.58 examples/s]26113 examples [00:23, 1154.10 examples/s]26232 examples [00:23, 1162.45 examples/s]26350 examples [00:23, 1165.06 examples/s]26472 examples [00:23, 1180.18 examples/s]26595 examples [00:23, 1193.52 examples/s]26715 examples [00:23, 1190.82 examples/s]26835 examples [00:23, 1169.90 examples/s]26953 examples [00:23, 1162.13 examples/s]27073 examples [00:23, 1170.89 examples/s]27191 examples [00:23, 1172.34 examples/s]27309 examples [00:24, 1169.15 examples/s]27429 examples [00:24, 1176.86 examples/s]27548 examples [00:24, 1180.65 examples/s]27669 examples [00:24, 1187.76 examples/s]27788 examples [00:24, 1179.21 examples/s]27907 examples [00:24, 1181.66 examples/s]28029 examples [00:24, 1190.46 examples/s]28149 examples [00:24, 1183.81 examples/s]28268 examples [00:24, 1176.80 examples/s]28386 examples [00:24, 1167.54 examples/s]28506 examples [00:25, 1176.57 examples/s]28624 examples [00:25, 1152.23 examples/s]28744 examples [00:25, 1165.91 examples/s]28867 examples [00:25, 1181.65 examples/s]28987 examples [00:25, 1184.95 examples/s]29106 examples [00:25, 1171.84 examples/s]29224 examples [00:25, 1167.96 examples/s]29341 examples [00:25, 1165.54 examples/s]29458 examples [00:25, 1139.57 examples/s]29574 examples [00:26, 1145.33 examples/s]29691 examples [00:26, 1149.98 examples/s]29807 examples [00:26, 1145.07 examples/s]29922 examples [00:26, 1125.69 examples/s]30035 examples [00:26, 1081.48 examples/s]30148 examples [00:26, 1093.54 examples/s]30264 examples [00:26, 1112.45 examples/s]30385 examples [00:26, 1138.81 examples/s]30500 examples [00:26, 1140.01 examples/s]30616 examples [00:26, 1143.12 examples/s]30731 examples [00:27, 1139.57 examples/s]30847 examples [00:27, 1145.06 examples/s]30965 examples [00:27, 1154.72 examples/s]31082 examples [00:27, 1159.11 examples/s]31202 examples [00:27, 1169.91 examples/s]31322 examples [00:27, 1176.28 examples/s]31440 examples [00:27, 1164.55 examples/s]31557 examples [00:27, 1156.32 examples/s]31673 examples [00:27, 1155.63 examples/s]31791 examples [00:27, 1160.46 examples/s]31909 examples [00:28, 1164.92 examples/s]32026 examples [00:28, 1165.14 examples/s]32143 examples [00:28, 1161.50 examples/s]32266 examples [00:28, 1181.21 examples/s]32385 examples [00:28, 1183.03 examples/s]32505 examples [00:28, 1185.48 examples/s]32627 examples [00:28, 1194.00 examples/s]32747 examples [00:28, 1183.74 examples/s]32866 examples [00:28, 1150.22 examples/s]32982 examples [00:28, 1150.13 examples/s]33098 examples [00:29, 1152.02 examples/s]33214 examples [00:29, 1144.64 examples/s]33329 examples [00:29, 1135.62 examples/s]33443 examples [00:29, 1128.64 examples/s]33556 examples [00:29, 1121.43 examples/s]33671 examples [00:29, 1129.61 examples/s]33791 examples [00:29, 1147.72 examples/s]33906 examples [00:29, 1140.39 examples/s]34021 examples [00:29, 1117.31 examples/s]34133 examples [00:29, 1115.65 examples/s]34246 examples [00:30, 1118.96 examples/s]34363 examples [00:30, 1133.17 examples/s]34477 examples [00:30, 1135.09 examples/s]34591 examples [00:30, 1129.25 examples/s]34708 examples [00:30, 1138.84 examples/s]34823 examples [00:30, 1139.20 examples/s]34941 examples [00:30, 1149.40 examples/s]35061 examples [00:30, 1163.57 examples/s]35178 examples [00:30, 1152.23 examples/s]35295 examples [00:31, 1154.22 examples/s]35411 examples [00:31, 1153.99 examples/s]35528 examples [00:31, 1157.64 examples/s]35648 examples [00:31, 1167.26 examples/s]35765 examples [00:31, 1166.14 examples/s]35888 examples [00:31, 1181.99 examples/s]36009 examples [00:31, 1189.64 examples/s]36131 examples [00:31, 1195.63 examples/s]36251 examples [00:31, 1191.03 examples/s]36371 examples [00:31, 1181.12 examples/s]36490 examples [00:32, 1179.86 examples/s]36609 examples [00:32, 1182.76 examples/s]36732 examples [00:32, 1194.01 examples/s]36852 examples [00:32, 1184.50 examples/s]36971 examples [00:32, 1177.25 examples/s]37089 examples [00:32, 1167.90 examples/s]37207 examples [00:32, 1170.98 examples/s]37325 examples [00:32, 1168.19 examples/s]37443 examples [00:32, 1169.58 examples/s]37560 examples [00:32, 1162.80 examples/s]37677 examples [00:33, 1160.97 examples/s]37795 examples [00:33, 1164.15 examples/s]37912 examples [00:33, 1156.28 examples/s]38028 examples [00:33, 1155.37 examples/s]38148 examples [00:33, 1166.73 examples/s]38265 examples [00:33, 1164.28 examples/s]38382 examples [00:33, 1148.35 examples/s]38497 examples [00:33, 1141.95 examples/s]38612 examples [00:33, 1133.07 examples/s]38726 examples [00:33, 1116.65 examples/s]38838 examples [00:34, 1111.26 examples/s]38950 examples [00:34, 1106.44 examples/s]39065 examples [00:34, 1117.99 examples/s]39185 examples [00:34, 1141.30 examples/s]39302 examples [00:34, 1148.89 examples/s]39420 examples [00:34, 1157.23 examples/s]39540 examples [00:34, 1166.99 examples/s]39657 examples [00:34, 1161.27 examples/s]39774 examples [00:34, 1140.27 examples/s]39893 examples [00:34, 1154.67 examples/s]40009 examples [00:35, 1095.34 examples/s]40130 examples [00:35, 1125.77 examples/s]40248 examples [00:35, 1140.48 examples/s]40363 examples [00:35, 1133.38 examples/s]40477 examples [00:35, 1132.48 examples/s]40595 examples [00:35, 1145.29 examples/s]40713 examples [00:35, 1155.05 examples/s]40830 examples [00:35, 1159.10 examples/s]40950 examples [00:35, 1170.91 examples/s]41069 examples [00:35, 1173.49 examples/s]41192 examples [00:36, 1187.88 examples/s]41314 examples [00:36, 1194.56 examples/s]41434 examples [00:36, 1191.10 examples/s]41554 examples [00:36, 1191.37 examples/s]41674 examples [00:36, 1164.97 examples/s]41796 examples [00:36, 1180.32 examples/s]41916 examples [00:36, 1183.01 examples/s]42035 examples [00:36, 1180.54 examples/s]42154 examples [00:36, 1160.09 examples/s]42271 examples [00:37, 1151.81 examples/s]42387 examples [00:37, 1153.39 examples/s]42504 examples [00:37, 1155.39 examples/s]42620 examples [00:37, 1155.16 examples/s]42737 examples [00:37, 1159.11 examples/s]42854 examples [00:37, 1159.72 examples/s]42973 examples [00:37, 1166.57 examples/s]43090 examples [00:37, 1133.06 examples/s]43204 examples [00:37, 1135.08 examples/s]43318 examples [00:37, 1133.08 examples/s]43432 examples [00:38, 1131.06 examples/s]43546 examples [00:38, 1132.55 examples/s]43660 examples [00:38, 1132.80 examples/s]43777 examples [00:38, 1141.81 examples/s]43897 examples [00:38, 1157.60 examples/s]44013 examples [00:38, 1153.11 examples/s]44134 examples [00:38, 1167.90 examples/s]44251 examples [00:38, 1140.59 examples/s]44366 examples [00:38, 1122.38 examples/s]44479 examples [00:38, 1119.20 examples/s]44592 examples [00:39, 1109.96 examples/s]44704 examples [00:39, 1104.48 examples/s]44817 examples [00:39, 1109.42 examples/s]44934 examples [00:39, 1124.45 examples/s]45048 examples [00:39, 1127.21 examples/s]45161 examples [00:39, 1121.73 examples/s]45277 examples [00:39, 1131.51 examples/s]45395 examples [00:39, 1144.69 examples/s]45514 examples [00:39, 1155.44 examples/s]45635 examples [00:39, 1170.04 examples/s]45753 examples [00:40, 1163.28 examples/s]45870 examples [00:40, 1156.65 examples/s]45987 examples [00:40, 1158.90 examples/s]46107 examples [00:40, 1169.76 examples/s]46230 examples [00:40, 1184.52 examples/s]46349 examples [00:40, 1185.27 examples/s]46468 examples [00:40, 1174.56 examples/s]46586 examples [00:40, 1173.71 examples/s]46704 examples [00:40, 1148.34 examples/s]46819 examples [00:40, 1142.93 examples/s]46934 examples [00:41, 1140.17 examples/s]47050 examples [00:41, 1145.89 examples/s]47169 examples [00:41, 1158.68 examples/s]47285 examples [00:41, 1156.17 examples/s]47401 examples [00:41, 1106.80 examples/s]47513 examples [00:41, 1108.03 examples/s]47625 examples [00:41, 1078.01 examples/s]47734 examples [00:41, 1061.13 examples/s]47852 examples [00:41, 1092.79 examples/s]47966 examples [00:42, 1104.31 examples/s]48077 examples [00:42, 1094.90 examples/s]48192 examples [00:42, 1108.51 examples/s]48306 examples [00:42, 1115.51 examples/s]48419 examples [00:42, 1119.57 examples/s]48535 examples [00:42, 1131.35 examples/s]48649 examples [00:42, 1127.84 examples/s]48763 examples [00:42, 1129.06 examples/s]48882 examples [00:42, 1145.43 examples/s]49000 examples [00:42, 1153.46 examples/s]49118 examples [00:43, 1159.84 examples/s]49235 examples [00:43, 1146.11 examples/s]49351 examples [00:43, 1150.17 examples/s]49470 examples [00:43, 1160.44 examples/s]49593 examples [00:43, 1178.84 examples/s]49715 examples [00:43, 1190.05 examples/s]49837 examples [00:43, 1196.76 examples/s]49957 examples [00:43, 1182.72 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8956/50000 [00:00<00:00, 89557.31 examples/s] 48%|████▊     | 24181/50000 [00:00<00:00, 102179.60 examples/s] 78%|███████▊  | 38910/50000 [00:00<00:00, 112516.93 examples/s]                                                                0 examples [00:00, ? examples/s]93 examples [00:00, 928.60 examples/s]215 examples [00:00, 998.85 examples/s]339 examples [00:00, 1060.56 examples/s]462 examples [00:00, 1105.18 examples/s]582 examples [00:00, 1130.88 examples/s]703 examples [00:00, 1153.50 examples/s]825 examples [00:00, 1170.49 examples/s]948 examples [00:00, 1186.15 examples/s]1070 examples [00:00, 1194.05 examples/s]1187 examples [00:01, 1174.79 examples/s]1303 examples [00:01, 1160.52 examples/s]1419 examples [00:01, 1159.19 examples/s]1542 examples [00:01, 1176.92 examples/s]1663 examples [00:01, 1185.93 examples/s]1782 examples [00:01, 1175.71 examples/s]1904 examples [00:01, 1185.82 examples/s]2023 examples [00:01, 1167.87 examples/s]2144 examples [00:01, 1178.36 examples/s]2262 examples [00:01, 1162.70 examples/s]2379 examples [00:02, 1154.14 examples/s]2495 examples [00:02, 1151.83 examples/s]2611 examples [00:02, 1144.88 examples/s]2726 examples [00:02, 1136.04 examples/s]2840 examples [00:02, 1135.49 examples/s]2954 examples [00:02, 1110.49 examples/s]3068 examples [00:02, 1117.36 examples/s]3180 examples [00:02, 1111.72 examples/s]3292 examples [00:02, 1111.79 examples/s]3404 examples [00:02, 1108.67 examples/s]3518 examples [00:03, 1115.35 examples/s]3632 examples [00:03, 1119.99 examples/s]3746 examples [00:03, 1123.92 examples/s]3860 examples [00:03, 1127.32 examples/s]3974 examples [00:03, 1130.94 examples/s]4088 examples [00:03, 1130.62 examples/s]4202 examples [00:03, 1130.88 examples/s]4316 examples [00:03, 1133.51 examples/s]4430 examples [00:03, 1129.86 examples/s]4544 examples [00:03, 1130.70 examples/s]4658 examples [00:04, 1124.29 examples/s]4772 examples [00:04, 1102.71 examples/s]4888 examples [00:04, 1118.80 examples/s]5005 examples [00:04, 1131.45 examples/s]5123 examples [00:04, 1143.37 examples/s]5238 examples [00:04, 1139.06 examples/s]5352 examples [00:04, 1132.78 examples/s]5466 examples [00:04, 1132.26 examples/s]5581 examples [00:04, 1134.55 examples/s]5695 examples [00:04, 1121.92 examples/s]5810 examples [00:05, 1128.00 examples/s]5923 examples [00:05, 1119.06 examples/s]6036 examples [00:05, 1120.33 examples/s]6153 examples [00:05, 1133.16 examples/s]6273 examples [00:05, 1150.32 examples/s]6389 examples [00:05, 1145.59 examples/s]6505 examples [00:05, 1147.90 examples/s]6626 examples [00:05, 1164.08 examples/s]6745 examples [00:05, 1169.99 examples/s]6863 examples [00:05, 1171.17 examples/s]6982 examples [00:06, 1175.74 examples/s]7100 examples [00:06, 1164.90 examples/s]7217 examples [00:06, 1164.00 examples/s]7340 examples [00:06, 1182.50 examples/s]7459 examples [00:06, 1181.48 examples/s]7578 examples [00:06, 1171.63 examples/s]7697 examples [00:06, 1176.64 examples/s]7815 examples [00:06, 1162.41 examples/s]7932 examples [00:06, 1162.13 examples/s]8049 examples [00:07, 1157.35 examples/s]8167 examples [00:07, 1161.53 examples/s]8284 examples [00:07, 1157.61 examples/s]8406 examples [00:07, 1175.32 examples/s]8528 examples [00:07, 1186.41 examples/s]8654 examples [00:07, 1206.04 examples/s]8775 examples [00:07, 1202.63 examples/s]8896 examples [00:07, 1187.46 examples/s]9015 examples [00:07, 1183.19 examples/s]9134 examples [00:07, 1166.16 examples/s]9256 examples [00:08, 1180.74 examples/s]9376 examples [00:08, 1184.51 examples/s]9495 examples [00:08, 1176.22 examples/s]9615 examples [00:08, 1182.09 examples/s]9734 examples [00:08, 1182.30 examples/s]9853 examples [00:08, 1173.53 examples/s]9971 examples [00:08, 1167.99 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete427FSR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete427FSR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70851cf158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70851cf158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70851cf158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f70d0bae0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f704414e828>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f70851cf158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f70851cf158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f70851cf158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f708337b2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f708337b048>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.11 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.90 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.98 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.98 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.67 file/s]2020-08-12 12:07:35.427336: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-12 12:07:35.431839: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-12 12:07:35.432006: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5558e10b6310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-12 12:07:35.432020: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157814.00it/s] 81%|████████  | 8003584/9912422 [00:00<00:08, 225257.66it/s]9920512it [00:00, 44846869.60it/s]                           
0it [00:00, ?it/s]32768it [00:00, 566992.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 471347.48it/s]1654784it [00:00, 11961264.31it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186864.65it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
