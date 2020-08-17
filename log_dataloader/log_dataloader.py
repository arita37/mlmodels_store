
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0cba498598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0cba498598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0d2514a400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0d2514a400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0d44366ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0d44366ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0cd24850d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0cd24850d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0cd24850d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|███▉      | 3932160/9912422 [00:00<00:00, 39315263.52it/s]9920512it [00:00, 38738825.06it/s]                             
0it [00:00, ?it/s]32768it [00:00, 556915.52it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 415312.93it/s]1654784it [00:00, 10875089.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171366.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cd0657710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cd0972978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0cd247dd08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0cd247dd08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0cd247dd08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:45,  1.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:45,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:45,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:44,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:01<00:36,  4.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:36,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:01<00:26,  5.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:26,  5.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:26,  5.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:26,  5.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:25,  5.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:19,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:19,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:14,  9.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:14,  9.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13,  9.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:13,  9.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:13,  9.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:10, 12.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:10, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:10, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:10, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:10, 12.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:08, 15.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:08, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:08, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:08, 15.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 18.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 18.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 25.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 25.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 25.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 25.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 25.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 27.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 27.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 27.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 27.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 27.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:02<00:03, 29.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:03, 29.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 29.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 29.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 29.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 30.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 30.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 30.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 30.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 30.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 31.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 31.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 31.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 31.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 31.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 32.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 32.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 32.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 33.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 33.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 33.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 33.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 33.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 33.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 36.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 36.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 36.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 36.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 36.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 36.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 39.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 39.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 39.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 39.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 39.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 39.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 41.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 43.62 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 43.62 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 43.62 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 43.62 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 45.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 46.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 46.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 46.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 46.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 46.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 46.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 46.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 46.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 46.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 46.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 46.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 46.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 46.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 46.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 46.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 46.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 46.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 46.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 46.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 46.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 46.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 46.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 46.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 46.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 46.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 46.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 47.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 47.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 47.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 47.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 47.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 47.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 47.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 47.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 47.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 47.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 47.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 47.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 47.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 48.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 48.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 48.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 48.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 48.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 48.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 48.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 48.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 48.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 48.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 48.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.34 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.71s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 48.34 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.71s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.71s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 24.14 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.71s/ url]
0 examples [00:00, ? examples/s]2020-08-17 00:07:31.706978: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-17 00:07:31.721566: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-08-17 00:07:31.721819: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eba779f490 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-17 00:07:31.721840: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
3 examples [00:00, 29.72 examples/s]112 examples [00:00, 41.97 examples/s]217 examples [00:00, 58.95 examples/s]317 examples [00:00, 82.13 examples/s]423 examples [00:00, 113.56 examples/s]518 examples [00:00, 154.28 examples/s]624 examples [00:00, 207.42 examples/s]728 examples [00:00, 272.93 examples/s]837 examples [00:00, 351.85 examples/s]944 examples [00:01, 440.24 examples/s]1054 examples [00:01, 536.30 examples/s]1159 examples [00:01, 627.65 examples/s]1269 examples [00:01, 719.55 examples/s]1381 examples [00:01, 804.81 examples/s]1493 examples [00:01, 878.74 examples/s]1604 examples [00:01, 935.45 examples/s]1714 examples [00:01, 978.25 examples/s]1824 examples [00:01, 1001.14 examples/s]1938 examples [00:01, 1037.13 examples/s]2054 examples [00:02, 1070.18 examples/s]2166 examples [00:02, 1066.50 examples/s]2276 examples [00:02, 1064.19 examples/s]2385 examples [00:02, 1044.94 examples/s]2492 examples [00:02, 1030.67 examples/s]2603 examples [00:02, 1050.96 examples/s]2714 examples [00:02, 1065.48 examples/s]2825 examples [00:02, 1077.74 examples/s]2934 examples [00:02, 1070.33 examples/s]3043 examples [00:02, 1073.80 examples/s]3153 examples [00:03, 1081.25 examples/s]3262 examples [00:03, 1077.45 examples/s]3370 examples [00:03, 1054.47 examples/s]3476 examples [00:03, 1016.60 examples/s]3579 examples [00:03, 989.92 examples/s] 3687 examples [00:03, 1013.34 examples/s]3793 examples [00:03, 1026.51 examples/s]3897 examples [00:03, 1023.77 examples/s]4003 examples [00:03, 1032.64 examples/s]4107 examples [00:03, 1003.23 examples/s]4214 examples [00:04, 1020.02 examples/s]4317 examples [00:04, 1007.96 examples/s]4419 examples [00:04, 982.50 examples/s] 4519 examples [00:04, 986.61 examples/s]4618 examples [00:04, 926.47 examples/s]4712 examples [00:04, 911.95 examples/s]4814 examples [00:04, 941.57 examples/s]4924 examples [00:04, 981.72 examples/s]5032 examples [00:04, 1007.93 examples/s]5142 examples [00:05, 1031.92 examples/s]5251 examples [00:05, 1047.43 examples/s]5360 examples [00:05, 1058.88 examples/s]5469 examples [00:05, 1064.84 examples/s]5576 examples [00:05, 1062.55 examples/s]5684 examples [00:05, 1066.12 examples/s]5792 examples [00:05, 1069.68 examples/s]5900 examples [00:05, 1041.85 examples/s]6005 examples [00:05, 1007.90 examples/s]6107 examples [00:05, 1009.17 examples/s]6213 examples [00:06, 1021.99 examples/s]6317 examples [00:06, 1026.89 examples/s]6423 examples [00:06, 1035.51 examples/s]6533 examples [00:06, 1051.54 examples/s]6639 examples [00:06, 1051.84 examples/s]6747 examples [00:06, 1059.03 examples/s]6856 examples [00:06, 1065.72 examples/s]6966 examples [00:06, 1072.84 examples/s]7074 examples [00:06, 1070.85 examples/s]7182 examples [00:06, 1065.84 examples/s]7291 examples [00:07, 1072.58 examples/s]7401 examples [00:07, 1078.79 examples/s]7509 examples [00:07, 1066.96 examples/s]7616 examples [00:07, 1063.97 examples/s]7723 examples [00:07, 1052.05 examples/s]7831 examples [00:07, 1058.88 examples/s]7937 examples [00:07, 1049.57 examples/s]8044 examples [00:07, 1055.57 examples/s]8153 examples [00:07, 1065.10 examples/s]8260 examples [00:07, 1065.02 examples/s]8369 examples [00:08, 1071.04 examples/s]8478 examples [00:08, 1075.43 examples/s]8586 examples [00:08, 1075.54 examples/s]8694 examples [00:08, 1064.15 examples/s]8801 examples [00:08, 1059.55 examples/s]8911 examples [00:08, 1069.28 examples/s]9018 examples [00:08, 1043.33 examples/s]9125 examples [00:08, 1048.92 examples/s]9232 examples [00:08, 1055.11 examples/s]9338 examples [00:09, 1049.42 examples/s]9445 examples [00:09, 1054.55 examples/s]9553 examples [00:09, 1061.71 examples/s]9660 examples [00:09, 1063.44 examples/s]9767 examples [00:09, 1060.53 examples/s]9876 examples [00:09, 1068.71 examples/s]9983 examples [00:09, 1062.05 examples/s]10090 examples [00:09, 1012.31 examples/s]10198 examples [00:09, 1031.34 examples/s]10308 examples [00:09, 1049.35 examples/s]10414 examples [00:10, 1049.30 examples/s]10523 examples [00:10, 1059.72 examples/s]10633 examples [00:10, 1070.25 examples/s]10743 examples [00:10, 1077.62 examples/s]10853 examples [00:10, 1082.12 examples/s]10962 examples [00:10, 1082.93 examples/s]11071 examples [00:10, 1068.60 examples/s]11178 examples [00:10, 1066.64 examples/s]11285 examples [00:10, 1059.41 examples/s]11391 examples [00:10, 1052.01 examples/s]11497 examples [00:11, 1052.50 examples/s]11604 examples [00:11, 1057.28 examples/s]11710 examples [00:11, 1055.43 examples/s]11817 examples [00:11, 1058.31 examples/s]11923 examples [00:11, 1020.56 examples/s]12031 examples [00:11, 1035.44 examples/s]12141 examples [00:11, 1053.72 examples/s]12247 examples [00:11, 1038.25 examples/s]12352 examples [00:11, 1025.05 examples/s]12458 examples [00:11, 1034.97 examples/s]12562 examples [00:12, 1026.29 examples/s]12672 examples [00:12, 1046.51 examples/s]12777 examples [00:12, 1022.12 examples/s]12885 examples [00:12, 1036.35 examples/s]12989 examples [00:12, 1028.06 examples/s]13092 examples [00:12, 1018.88 examples/s]13200 examples [00:12, 1034.42 examples/s]13311 examples [00:12, 1054.26 examples/s]13420 examples [00:12, 1063.89 examples/s]13531 examples [00:12, 1074.40 examples/s]13639 examples [00:13, 1072.28 examples/s]13748 examples [00:13, 1076.64 examples/s]13857 examples [00:13, 1078.96 examples/s]13965 examples [00:13, 1068.98 examples/s]14072 examples [00:13, 1063.31 examples/s]14179 examples [00:13, 1046.34 examples/s]14284 examples [00:13, 1041.43 examples/s]14392 examples [00:13, 1052.31 examples/s]14498 examples [00:13, 1054.00 examples/s]14609 examples [00:14, 1068.30 examples/s]14718 examples [00:14, 1074.02 examples/s]14827 examples [00:14, 1078.71 examples/s]14937 examples [00:14, 1082.38 examples/s]15046 examples [00:14, 1068.98 examples/s]15155 examples [00:14, 1074.10 examples/s]15264 examples [00:14, 1077.94 examples/s]15372 examples [00:14, 1069.46 examples/s]15479 examples [00:14, 1061.02 examples/s]15586 examples [00:14, 1062.85 examples/s]15694 examples [00:15, 1067.77 examples/s]15803 examples [00:15, 1072.76 examples/s]15912 examples [00:15, 1077.05 examples/s]16022 examples [00:15, 1083.21 examples/s]16131 examples [00:15, 1083.97 examples/s]16241 examples [00:15, 1086.18 examples/s]16350 examples [00:15, 1084.72 examples/s]16459 examples [00:15, 1071.35 examples/s]16567 examples [00:15, 1054.32 examples/s]16673 examples [00:15, 1040.30 examples/s]16778 examples [00:16, 1034.83 examples/s]16882 examples [00:16, 1031.46 examples/s]16986 examples [00:16, 1020.59 examples/s]17095 examples [00:16, 1039.44 examples/s]17202 examples [00:16, 1048.26 examples/s]17310 examples [00:16, 1055.23 examples/s]17416 examples [00:16, 1041.01 examples/s]17526 examples [00:16, 1056.06 examples/s]17636 examples [00:16, 1068.74 examples/s]17744 examples [00:16, 1071.32 examples/s]17853 examples [00:17, 1076.20 examples/s]17961 examples [00:17, 1076.52 examples/s]18069 examples [00:17, 1073.72 examples/s]18177 examples [00:17, 1069.22 examples/s]18284 examples [00:17, 1066.10 examples/s]18394 examples [00:17, 1075.96 examples/s]18502 examples [00:17, 1067.20 examples/s]18609 examples [00:17, 1038.49 examples/s]18714 examples [00:17, 1018.14 examples/s]18819 examples [00:17, 1025.32 examples/s]18922 examples [00:18, 984.48 examples/s] 19031 examples [00:18, 1011.82 examples/s]19140 examples [00:18, 1033.54 examples/s]19249 examples [00:18, 1048.67 examples/s]19355 examples [00:18, 1041.53 examples/s]19466 examples [00:18, 1058.74 examples/s]19573 examples [00:18, 1060.17 examples/s]19683 examples [00:18, 1070.17 examples/s]19791 examples [00:18, 1046.18 examples/s]19896 examples [00:19, 1045.48 examples/s]20001 examples [00:19, 1005.35 examples/s]20107 examples [00:19, 1019.32 examples/s]20218 examples [00:19, 1043.82 examples/s]20327 examples [00:19, 1055.70 examples/s]20436 examples [00:19, 1062.97 examples/s]20543 examples [00:19, 1060.03 examples/s]20651 examples [00:19, 1064.72 examples/s]20758 examples [00:19, 1033.64 examples/s]20864 examples [00:19, 1039.11 examples/s]20973 examples [00:20, 1052.77 examples/s]21083 examples [00:20, 1064.01 examples/s]21190 examples [00:20, 1064.57 examples/s]21298 examples [00:20, 1068.55 examples/s]21407 examples [00:20, 1074.35 examples/s]21515 examples [00:20, 1072.92 examples/s]21624 examples [00:20, 1077.09 examples/s]21732 examples [00:20, 1073.32 examples/s]21840 examples [00:20, 1072.71 examples/s]21948 examples [00:20, 1067.62 examples/s]22055 examples [00:21, 1055.07 examples/s]22165 examples [00:21, 1067.20 examples/s]22273 examples [00:21, 1068.14 examples/s]22384 examples [00:21, 1077.77 examples/s]22492 examples [00:21, 1067.31 examples/s]22599 examples [00:21, 1057.64 examples/s]22708 examples [00:21, 1065.15 examples/s]22815 examples [00:21, 1056.24 examples/s]22923 examples [00:21, 1050.41 examples/s]23031 examples [00:21, 1057.08 examples/s]23139 examples [00:22, 1062.83 examples/s]23246 examples [00:22, 1054.85 examples/s]23352 examples [00:22, 1033.06 examples/s]23459 examples [00:22, 1042.47 examples/s]23568 examples [00:22, 1054.77 examples/s]23678 examples [00:22, 1066.65 examples/s]23788 examples [00:22, 1076.15 examples/s]23896 examples [00:22, 1072.00 examples/s]24008 examples [00:22, 1085.23 examples/s]24117 examples [00:22, 1073.27 examples/s]24226 examples [00:23, 1077.29 examples/s]24337 examples [00:23, 1084.90 examples/s]24446 examples [00:23, 1080.17 examples/s]24555 examples [00:23, 1064.03 examples/s]24667 examples [00:23, 1079.83 examples/s]24778 examples [00:23, 1086.31 examples/s]24888 examples [00:23, 1088.62 examples/s]24997 examples [00:23, 1076.68 examples/s]25105 examples [00:23, 1069.19 examples/s]25215 examples [00:24, 1071.65 examples/s]25323 examples [00:24, 1054.00 examples/s]25430 examples [00:24, 1057.09 examples/s]25536 examples [00:24, 1047.37 examples/s]25642 examples [00:24, 1049.93 examples/s]25748 examples [00:24, 1050.56 examples/s]25855 examples [00:24, 1054.11 examples/s]25963 examples [00:24, 1061.33 examples/s]26070 examples [00:24, 1054.63 examples/s]26176 examples [00:24, 1054.86 examples/s]26282 examples [00:25, 1022.32 examples/s]26388 examples [00:25, 1031.51 examples/s]26495 examples [00:25, 1040.40 examples/s]26600 examples [00:25, 1042.64 examples/s]26706 examples [00:25, 1045.04 examples/s]26811 examples [00:25, 1038.05 examples/s]26915 examples [00:25, 1027.75 examples/s]27020 examples [00:25, 1033.99 examples/s]27124 examples [00:25, 1030.54 examples/s]27228 examples [00:25, 1020.01 examples/s]27331 examples [00:26, 1021.64 examples/s]27439 examples [00:26, 1036.14 examples/s]27547 examples [00:26, 1046.61 examples/s]27652 examples [00:26, 1044.20 examples/s]27757 examples [00:26, 1034.67 examples/s]27861 examples [00:26, 1033.37 examples/s]27965 examples [00:26, 1032.25 examples/s]28072 examples [00:26, 1042.72 examples/s]28177 examples [00:26, 1043.00 examples/s]28284 examples [00:26, 1048.90 examples/s]28389 examples [00:27, 1031.83 examples/s]28499 examples [00:27, 1050.76 examples/s]28607 examples [00:27, 1059.02 examples/s]28714 examples [00:27, 1056.19 examples/s]28823 examples [00:27, 1063.44 examples/s]28932 examples [00:27, 1070.20 examples/s]29040 examples [00:27, 1060.64 examples/s]29147 examples [00:27, 991.59 examples/s] 29248 examples [00:27, 974.89 examples/s]29356 examples [00:28, 1003.07 examples/s]29458 examples [00:28, 943.95 examples/s] 29554 examples [00:28, 935.13 examples/s]29650 examples [00:28, 941.47 examples/s]29754 examples [00:28, 968.78 examples/s]29865 examples [00:28, 1004.84 examples/s]29975 examples [00:28, 1031.17 examples/s]30079 examples [00:28, 995.53 examples/s] 30183 examples [00:28, 1003.56 examples/s]30292 examples [00:28, 1025.71 examples/s]30401 examples [00:29, 1042.97 examples/s]30507 examples [00:29, 1046.91 examples/s]30613 examples [00:29, 1048.40 examples/s]30721 examples [00:29, 1056.88 examples/s]30827 examples [00:29, 1048.90 examples/s]30936 examples [00:29, 1058.85 examples/s]31045 examples [00:29, 1066.91 examples/s]31152 examples [00:29, 1026.56 examples/s]31257 examples [00:29, 1030.58 examples/s]31366 examples [00:29, 1045.79 examples/s]31479 examples [00:30, 1069.03 examples/s]31587 examples [00:30, 1062.65 examples/s]31697 examples [00:30, 1071.07 examples/s]31805 examples [00:30, 1060.79 examples/s]31912 examples [00:30, 1029.34 examples/s]32017 examples [00:30, 1035.17 examples/s]32123 examples [00:30, 1039.54 examples/s]32229 examples [00:30, 1044.08 examples/s]32334 examples [00:30, 1038.52 examples/s]32438 examples [00:31, 1018.40 examples/s]32547 examples [00:31, 1037.97 examples/s]32658 examples [00:31, 1058.48 examples/s]32771 examples [00:31, 1076.21 examples/s]32879 examples [00:31, 1075.13 examples/s]32987 examples [00:31, 1075.97 examples/s]33095 examples [00:31, 1044.87 examples/s]33201 examples [00:31, 1048.16 examples/s]33307 examples [00:31, 1040.52 examples/s]33412 examples [00:31, 1030.35 examples/s]33517 examples [00:32, 1032.41 examples/s]33621 examples [00:32, 1034.52 examples/s]33728 examples [00:32, 1044.48 examples/s]33833 examples [00:32, 1038.83 examples/s]33940 examples [00:32, 1045.72 examples/s]34049 examples [00:32, 1056.83 examples/s]34157 examples [00:32, 1062.85 examples/s]34267 examples [00:32, 1071.75 examples/s]34379 examples [00:32, 1084.59 examples/s]34489 examples [00:32, 1089.06 examples/s]34598 examples [00:33, 1082.17 examples/s]34707 examples [00:33, 1079.65 examples/s]34816 examples [00:33, 1075.71 examples/s]34924 examples [00:33, 1075.12 examples/s]35032 examples [00:33, 1060.67 examples/s]35139 examples [00:33, 1060.02 examples/s]35246 examples [00:33, 1040.79 examples/s]35351 examples [00:33, 1027.83 examples/s]35457 examples [00:33, 1036.02 examples/s]35561 examples [00:33, 1025.47 examples/s]35664 examples [00:34, 1012.86 examples/s]35772 examples [00:34, 1031.22 examples/s]35877 examples [00:34, 1035.02 examples/s]35981 examples [00:34, 1028.06 examples/s]36084 examples [00:34, 1027.08 examples/s]36189 examples [00:34, 1033.49 examples/s]36294 examples [00:34, 1036.14 examples/s]36398 examples [00:34, 1034.76 examples/s]36504 examples [00:34, 1039.63 examples/s]36608 examples [00:34, 1036.09 examples/s]36713 examples [00:35, 1039.03 examples/s]36818 examples [00:35, 1042.12 examples/s]36923 examples [00:35, 1016.13 examples/s]37028 examples [00:35, 1025.53 examples/s]37131 examples [00:35, 1023.93 examples/s]37234 examples [00:35, 1020.73 examples/s]37337 examples [00:35, 1002.81 examples/s]37440 examples [00:35, 1010.50 examples/s]37542 examples [00:35, 968.79 examples/s] 37640 examples [00:36, 963.70 examples/s]37741 examples [00:36, 976.92 examples/s]37846 examples [00:36, 995.53 examples/s]37947 examples [00:36, 999.31 examples/s]38051 examples [00:36, 1010.12 examples/s]38155 examples [00:36, 1016.05 examples/s]38260 examples [00:36, 1025.10 examples/s]38366 examples [00:36, 1033.09 examples/s]38471 examples [00:36, 1037.32 examples/s]38577 examples [00:36, 1040.41 examples/s]38682 examples [00:37, 1033.02 examples/s]38786 examples [00:37, 1031.88 examples/s]38892 examples [00:37, 1039.43 examples/s]38996 examples [00:37, 1009.77 examples/s]39098 examples [00:37, 1009.00 examples/s]39203 examples [00:37, 1019.17 examples/s]39312 examples [00:37, 1036.86 examples/s]39421 examples [00:37, 1051.57 examples/s]39527 examples [00:37, 1042.19 examples/s]39634 examples [00:37, 1047.41 examples/s]39739 examples [00:38, 1031.11 examples/s]39848 examples [00:38, 1045.70 examples/s]39956 examples [00:38, 1055.15 examples/s]40062 examples [00:38, 1006.99 examples/s]40170 examples [00:38, 1027.29 examples/s]40275 examples [00:38, 1032.02 examples/s]40381 examples [00:38, 1038.87 examples/s]40486 examples [00:38, 1013.16 examples/s]40594 examples [00:38, 1032.02 examples/s]40703 examples [00:38, 1046.51 examples/s]40808 examples [00:39, 1043.60 examples/s]40918 examples [00:39, 1057.55 examples/s]41026 examples [00:39, 1060.83 examples/s]41134 examples [00:39, 1066.31 examples/s]41241 examples [00:39, 1054.05 examples/s]41347 examples [00:39, 1048.46 examples/s]41452 examples [00:39, 1039.83 examples/s]41557 examples [00:39, 1032.75 examples/s]41661 examples [00:39, 1026.71 examples/s]41768 examples [00:39, 1036.52 examples/s]41873 examples [00:40, 1040.29 examples/s]41978 examples [00:40, 1042.76 examples/s]42085 examples [00:40, 1048.25 examples/s]42190 examples [00:40, 1034.53 examples/s]42294 examples [00:40, 1018.69 examples/s]42397 examples [00:40, 1017.17 examples/s]42501 examples [00:40, 1021.59 examples/s]42605 examples [00:40, 1024.41 examples/s]42709 examples [00:40, 1028.19 examples/s]42817 examples [00:41, 1042.04 examples/s]42922 examples [00:41, 1042.59 examples/s]43028 examples [00:41, 1047.57 examples/s]43134 examples [00:41, 1050.15 examples/s]43241 examples [00:41, 1053.41 examples/s]43348 examples [00:41, 1055.75 examples/s]43454 examples [00:41, 1045.99 examples/s]43562 examples [00:41, 1055.20 examples/s]43668 examples [00:41, 1050.84 examples/s]43774 examples [00:41, 1044.01 examples/s]43879 examples [00:42, 1035.66 examples/s]43983 examples [00:42, 1030.85 examples/s]44087 examples [00:42, 1030.77 examples/s]44191 examples [00:42, 1001.20 examples/s]44296 examples [00:42, 1014.41 examples/s]44406 examples [00:42, 1037.86 examples/s]44511 examples [00:42, 1041.13 examples/s]44616 examples [00:42, 1039.33 examples/s]44721 examples [00:42, 1040.08 examples/s]44826 examples [00:42, 1037.84 examples/s]44931 examples [00:43, 1038.40 examples/s]45036 examples [00:43, 1039.94 examples/s]45142 examples [00:43, 1043.37 examples/s]45248 examples [00:43, 1046.23 examples/s]45353 examples [00:43, 1040.58 examples/s]45458 examples [00:43, 1041.72 examples/s]45563 examples [00:43, 1029.54 examples/s]45666 examples [00:43, 994.33 examples/s] 45766 examples [00:43, 991.91 examples/s]45870 examples [00:43, 1003.71 examples/s]45978 examples [00:44, 1023.14 examples/s]46086 examples [00:44, 1038.68 examples/s]46194 examples [00:44, 1050.48 examples/s]46303 examples [00:44, 1059.75 examples/s]46411 examples [00:44, 1065.26 examples/s]46518 examples [00:44, 1044.52 examples/s]46623 examples [00:44, 1023.00 examples/s]46728 examples [00:44, 1028.35 examples/s]46831 examples [00:44, 1024.61 examples/s]46935 examples [00:44, 1026.50 examples/s]47040 examples [00:45, 1031.68 examples/s]47144 examples [00:45, 1032.94 examples/s]47250 examples [00:45, 1038.05 examples/s]47354 examples [00:45, 1032.22 examples/s]47459 examples [00:45, 1035.57 examples/s]47565 examples [00:45, 1042.32 examples/s]47670 examples [00:45, 1042.54 examples/s]47775 examples [00:45, 1041.24 examples/s]47880 examples [00:45, 1030.33 examples/s]47984 examples [00:45, 1032.42 examples/s]48088 examples [00:46, 1025.78 examples/s]48193 examples [00:46, 1031.38 examples/s]48298 examples [00:46, 1034.42 examples/s]48404 examples [00:46, 1039.89 examples/s]48509 examples [00:46, 1010.12 examples/s]48611 examples [00:46, 1002.03 examples/s]48715 examples [00:46, 1011.72 examples/s]48821 examples [00:46, 1023.34 examples/s]48926 examples [00:46, 1029.17 examples/s]49030 examples [00:47, 1028.32 examples/s]49135 examples [00:47, 1031.79 examples/s]49239 examples [00:47, 1032.80 examples/s]49343 examples [00:47, 1028.73 examples/s]49446 examples [00:47, 1007.61 examples/s]49550 examples [00:47, 1016.32 examples/s]49654 examples [00:47, 1021.73 examples/s]49759 examples [00:47, 1029.34 examples/s]49863 examples [00:47, 1027.89 examples/s]49968 examples [00:47, 1031.89 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5920/50000 [00:00<00:00, 59196.39 examples/s] 38%|███▊      | 19105/50000 [00:00<00:00, 70919.51 examples/s] 64%|██████▍   | 32041/50000 [00:00<00:00, 82031.85 examples/s] 90%|█████████ | 45186/50000 [00:00<00:00, 92458.18 examples/s]                                                               0 examples [00:00, ? examples/s]88 examples [00:00, 878.61 examples/s]193 examples [00:00, 922.74 examples/s]292 examples [00:00, 940.24 examples/s]396 examples [00:00, 968.03 examples/s]502 examples [00:00, 992.07 examples/s]606 examples [00:00, 1003.72 examples/s]712 examples [00:00, 1019.08 examples/s]818 examples [00:00, 1029.80 examples/s]925 examples [00:00, 1039.62 examples/s]1035 examples [00:01, 1056.20 examples/s]1143 examples [00:01, 1062.63 examples/s]1249 examples [00:01, 1058.96 examples/s]1354 examples [00:01, 1052.58 examples/s]1459 examples [00:01, 1050.72 examples/s]1564 examples [00:01, 1042.84 examples/s]1669 examples [00:01, 1042.45 examples/s]1773 examples [00:01, 1035.08 examples/s]1878 examples [00:01, 1038.14 examples/s]1984 examples [00:01, 1044.09 examples/s]2090 examples [00:02, 1048.40 examples/s]2197 examples [00:02, 1052.19 examples/s]2304 examples [00:02, 1056.99 examples/s]2410 examples [00:02, 1048.99 examples/s]2515 examples [00:02, 1042.90 examples/s]2620 examples [00:02, 1040.55 examples/s]2725 examples [00:02, 1041.62 examples/s]2833 examples [00:02, 1050.75 examples/s]2941 examples [00:02, 1056.55 examples/s]3047 examples [00:02, 1028.36 examples/s]3153 examples [00:03, 1037.63 examples/s]3265 examples [00:03, 1059.08 examples/s]3376 examples [00:03, 1073.80 examples/s]3486 examples [00:03, 1081.21 examples/s]3596 examples [00:03, 1086.69 examples/s]3707 examples [00:03, 1092.65 examples/s]3817 examples [00:03, 1089.32 examples/s]3926 examples [00:03, 1063.62 examples/s]4033 examples [00:03, 1060.18 examples/s]4140 examples [00:03, 1048.34 examples/s]4245 examples [00:04, 1046.81 examples/s]4352 examples [00:04, 1050.90 examples/s]4459 examples [00:04, 1055.39 examples/s]4566 examples [00:04, 1058.87 examples/s]4673 examples [00:04, 1059.82 examples/s]4780 examples [00:04, 1062.50 examples/s]4887 examples [00:04, 1054.83 examples/s]4994 examples [00:04, 1059.10 examples/s]5100 examples [00:04, 1057.71 examples/s]5206 examples [00:04, 1055.82 examples/s]5312 examples [00:05, 1029.67 examples/s]5419 examples [00:05, 1041.06 examples/s]5524 examples [00:05, 1034.79 examples/s]5628 examples [00:05, 1022.54 examples/s]5737 examples [00:05, 1040.46 examples/s]5847 examples [00:05, 1055.10 examples/s]5955 examples [00:05, 1061.77 examples/s]6063 examples [00:05, 1066.97 examples/s]6170 examples [00:05, 1046.13 examples/s]6275 examples [00:05, 1011.00 examples/s]6381 examples [00:06, 1022.95 examples/s]6490 examples [00:06, 1041.05 examples/s]6600 examples [00:06, 1056.40 examples/s]6706 examples [00:06, 1043.08 examples/s]6812 examples [00:06, 1046.95 examples/s]6920 examples [00:06, 1054.51 examples/s]7026 examples [00:06, 1045.38 examples/s]7131 examples [00:06, 1044.64 examples/s]7236 examples [00:06, 1043.97 examples/s]7341 examples [00:07, 1028.27 examples/s]7445 examples [00:07, 1031.16 examples/s]7551 examples [00:07, 1038.67 examples/s]7655 examples [00:07, 1028.03 examples/s]7760 examples [00:07, 1033.66 examples/s]7864 examples [00:07, 1032.07 examples/s]7972 examples [00:07, 1043.62 examples/s]8077 examples [00:07, 1035.07 examples/s]8181 examples [00:07, 1036.50 examples/s]8285 examples [00:07, 1036.80 examples/s]8389 examples [00:08, 1019.56 examples/s]8496 examples [00:08, 1033.60 examples/s]8603 examples [00:08, 1043.43 examples/s]8708 examples [00:08, 1038.07 examples/s]8814 examples [00:08, 1042.35 examples/s]8919 examples [00:08, 995.62 examples/s] 9024 examples [00:08, 1010.73 examples/s]9127 examples [00:08, 1014.78 examples/s]9229 examples [00:08, 997.33 examples/s] 9336 examples [00:08, 1017.09 examples/s]9438 examples [00:09, 1009.88 examples/s]9543 examples [00:09, 1020.81 examples/s]9652 examples [00:09, 1040.53 examples/s]9757 examples [00:09, 1035.09 examples/s]9861 examples [00:09, 1024.93 examples/s]9964 examples [00:09, 1015.95 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSZ8CEP/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSZ8CEP/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0cd24850d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0cd24850d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0cd24850d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d1de647b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0c58753e80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0cd24850d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0cd24850d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0cd24850d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cd0972358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0cd0972588>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.94 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.63 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.85 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]2020-08-17 00:08:37.505616: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-17 00:08:37.510153: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-08-17 00:08:37.510338: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5621d11e07c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-17 00:08:37.510354: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 156731.68it/s] 82%|████████▏ | 8151040/9912422 [00:00<00:07, 223714.84it/s]9920512it [00:00, 45408420.83it/s]                           
0it [00:00, ?it/s]32768it [00:00, 448601.55it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157508.71it/s]1654784it [00:00, 11369829.85it/s]                         
0it [00:00, ?it/s]8192it [00:00, 163097.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
