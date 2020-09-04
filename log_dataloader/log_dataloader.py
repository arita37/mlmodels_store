
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3fc653a7999aa2761b089e07ab0db303a5051c0b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f8f6cab26a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f8f6cab26a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8fd77db488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8fd77db488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f8ff69fae18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f8ff69fae18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8f84b14510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8f84b14510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8f84b14510> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|██        | 2023424/9912422 [00:00<00:00, 18837910.36it/s] 88%|████████▊ | 8740864/9912422 [00:00<00:00, 24012827.46it/s]9920512it [00:00, 25867731.62it/s]                             
0it [00:00, ?it/s]32768it [00:00, 506661.24it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 138668.95it/s]1654784it [00:00, 10315019.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 135721.27it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f6c9e6f98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f6bd2ecf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f8f84b141e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f8f84b141e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f8f84b141e0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:18,  7.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12,  9.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
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

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 43.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 55.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 63.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 64.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 65.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 65.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 65.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.13s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 65.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.03 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.40s/ url]
0 examples [00:00, ? examples/s]2020-09-04 00:08:09.020573: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-04 00:08:09.033685: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-09-04 00:08:09.033923: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5600a4c50be0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-04 00:08:09.033945: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
22 examples [00:00, 219.04 examples/s]118 examples [00:00, 284.97 examples/s]216 examples [00:00, 361.97 examples/s]319 examples [00:00, 448.95 examples/s]420 examples [00:00, 538.40 examples/s]524 examples [00:00, 628.42 examples/s]623 examples [00:00, 704.40 examples/s]726 examples [00:00, 777.97 examples/s]827 examples [00:00, 834.17 examples/s]926 examples [00:01, 874.84 examples/s]1034 examples [00:01, 926.79 examples/s]1134 examples [00:01, 929.06 examples/s]1238 examples [00:01, 957.80 examples/s]1338 examples [00:01, 969.50 examples/s]1438 examples [00:01, 970.29 examples/s]1537 examples [00:01, 974.53 examples/s]1636 examples [00:01, 975.43 examples/s]1735 examples [00:01, 948.11 examples/s]1837 examples [00:01, 968.42 examples/s]1946 examples [00:02, 1001.17 examples/s]2053 examples [00:02, 1020.03 examples/s]2156 examples [00:02, 1017.29 examples/s]2259 examples [00:02, 1019.33 examples/s]2362 examples [00:02, 997.02 examples/s] 2466 examples [00:02, 1008.29 examples/s]2568 examples [00:02, 1008.05 examples/s]2672 examples [00:02, 1015.66 examples/s]2775 examples [00:02, 1017.84 examples/s]2877 examples [00:02, 1017.21 examples/s]2983 examples [00:03, 1027.23 examples/s]3090 examples [00:03, 1039.12 examples/s]3194 examples [00:03, 1022.31 examples/s]3297 examples [00:03, 1019.42 examples/s]3400 examples [00:03, 996.23 examples/s] 3501 examples [00:03, 999.94 examples/s]3602 examples [00:03, 989.47 examples/s]3707 examples [00:03, 1004.51 examples/s]3813 examples [00:03, 1018.68 examples/s]3918 examples [00:03, 1027.74 examples/s]4029 examples [00:04, 1050.44 examples/s]4139 examples [00:04, 1062.55 examples/s]4249 examples [00:04, 1070.90 examples/s]4357 examples [00:04, 1070.99 examples/s]4465 examples [00:04, 1050.53 examples/s]4571 examples [00:04, 1033.95 examples/s]4675 examples [00:04, 1017.32 examples/s]4777 examples [00:04, 1012.75 examples/s]4883 examples [00:04, 1024.20 examples/s]4986 examples [00:04, 1019.03 examples/s]5088 examples [00:05, 1017.05 examples/s]5190 examples [00:05, 1003.70 examples/s]5291 examples [00:05, 1002.56 examples/s]5392 examples [00:05, 995.66 examples/s] 5496 examples [00:05, 1006.07 examples/s]5599 examples [00:05, 1011.75 examples/s]5701 examples [00:05, 1006.39 examples/s]5805 examples [00:05, 1014.68 examples/s]5907 examples [00:05, 1015.57 examples/s]6012 examples [00:06, 1024.16 examples/s]6121 examples [00:06, 1040.89 examples/s]6230 examples [00:06, 1053.78 examples/s]6336 examples [00:06, 1027.65 examples/s]6439 examples [00:06, 1027.99 examples/s]6542 examples [00:06, 980.63 examples/s] 6641 examples [00:06, 936.64 examples/s]6747 examples [00:06, 970.29 examples/s]6848 examples [00:06, 981.53 examples/s]6953 examples [00:06, 998.76 examples/s]7055 examples [00:07, 1004.67 examples/s]7160 examples [00:07, 1016.42 examples/s]7262 examples [00:07, 981.01 examples/s] 7361 examples [00:07, 969.16 examples/s]7461 examples [00:07, 975.56 examples/s]7567 examples [00:07, 998.92 examples/s]7673 examples [00:07, 1015.14 examples/s]7775 examples [00:07, 987.84 examples/s] 7878 examples [00:07, 997.51 examples/s]7987 examples [00:07, 1022.96 examples/s]8097 examples [00:08, 1044.78 examples/s]8202 examples [00:08, 1031.57 examples/s]8306 examples [00:08, 1031.21 examples/s]8415 examples [00:08, 1046.82 examples/s]8524 examples [00:08, 1057.73 examples/s]8635 examples [00:08, 1070.66 examples/s]8743 examples [00:08, 1063.79 examples/s]8850 examples [00:08, 1049.05 examples/s]8956 examples [00:08, 1033.26 examples/s]9060 examples [00:09, 984.78 examples/s] 9163 examples [00:09, 995.49 examples/s]9263 examples [00:09, 991.36 examples/s]9363 examples [00:09, 991.18 examples/s]9463 examples [00:09, 981.96 examples/s]9562 examples [00:09, 967.42 examples/s]9659 examples [00:09, 967.86 examples/s]9769 examples [00:09, 1002.89 examples/s]9878 examples [00:09, 1026.02 examples/s]9985 examples [00:09, 1036.23 examples/s]10089 examples [00:10, 983.65 examples/s]10192 examples [00:10, 995.60 examples/s]10294 examples [00:10, 1001.61 examples/s]10396 examples [00:10, 1004.97 examples/s]10499 examples [00:10, 1011.71 examples/s]10602 examples [00:10, 1016.49 examples/s]10704 examples [00:10, 1010.18 examples/s]10808 examples [00:10, 1017.44 examples/s]10910 examples [00:10, 986.98 examples/s] 11013 examples [00:10, 996.85 examples/s]11115 examples [00:11, 1003.46 examples/s]11219 examples [00:11, 1011.49 examples/s]11321 examples [00:11, 1003.11 examples/s]11424 examples [00:11, 1009.55 examples/s]11526 examples [00:11, 1005.64 examples/s]11630 examples [00:11, 1013.35 examples/s]11736 examples [00:11, 1026.81 examples/s]11842 examples [00:11, 1035.74 examples/s]11948 examples [00:11, 1041.20 examples/s]12053 examples [00:11, 1034.77 examples/s]12157 examples [00:12, 1035.38 examples/s]12261 examples [00:12, 1024.12 examples/s]12364 examples [00:12, 990.98 examples/s] 12467 examples [00:12, 1000.99 examples/s]12568 examples [00:12, 990.88 examples/s] 12672 examples [00:12, 1004.81 examples/s]12773 examples [00:12, 1000.11 examples/s]12874 examples [00:12, 1001.20 examples/s]12978 examples [00:12, 1010.98 examples/s]13087 examples [00:13, 1030.81 examples/s]13191 examples [00:13, 1014.39 examples/s]13293 examples [00:13, 1012.57 examples/s]13395 examples [00:13, 1002.45 examples/s]13497 examples [00:13, 1007.02 examples/s]13604 examples [00:13, 1024.17 examples/s]13707 examples [00:13, 996.60 examples/s] 13810 examples [00:13, 1006.08 examples/s]13919 examples [00:13, 1027.98 examples/s]14027 examples [00:13, 1042.49 examples/s]14132 examples [00:14, 1044.73 examples/s]14237 examples [00:14, 1041.41 examples/s]14342 examples [00:14, 1028.41 examples/s]14445 examples [00:14, 1025.07 examples/s]14549 examples [00:14, 1026.61 examples/s]14652 examples [00:14, 1010.42 examples/s]14754 examples [00:14, 1003.17 examples/s]14858 examples [00:14, 1013.10 examples/s]14960 examples [00:14, 996.73 examples/s] 15062 examples [00:14, 1001.45 examples/s]15167 examples [00:15, 1013.98 examples/s]15273 examples [00:15, 1025.84 examples/s]15376 examples [00:15, 1022.82 examples/s]15479 examples [00:15, 1023.63 examples/s]15582 examples [00:15, 1021.25 examples/s]15686 examples [00:15, 1025.37 examples/s]15789 examples [00:15, 1015.50 examples/s]15892 examples [00:15, 1018.79 examples/s]15995 examples [00:15, 1020.23 examples/s]16098 examples [00:15, 1021.70 examples/s]16201 examples [00:16, 1010.44 examples/s]16306 examples [00:16, 1020.68 examples/s]16413 examples [00:16, 1032.90 examples/s]16517 examples [00:16, 1031.59 examples/s]16621 examples [00:16, 1028.96 examples/s]16724 examples [00:16, 1029.04 examples/s]16827 examples [00:16, 970.50 examples/s] 16931 examples [00:16, 989.32 examples/s]17040 examples [00:16, 1015.71 examples/s]17146 examples [00:16, 1027.41 examples/s]17250 examples [00:17, 1023.86 examples/s]17359 examples [00:17, 1042.55 examples/s]17466 examples [00:17, 1049.34 examples/s]17572 examples [00:17, 1047.97 examples/s]17682 examples [00:17, 1060.19 examples/s]17789 examples [00:17, 1046.04 examples/s]17896 examples [00:17, 1052.82 examples/s]18004 examples [00:17, 1059.78 examples/s]18113 examples [00:17, 1066.71 examples/s]18221 examples [00:18, 1069.82 examples/s]18329 examples [00:18, 1049.40 examples/s]18435 examples [00:18, 1046.75 examples/s]18540 examples [00:18, 1044.27 examples/s]18645 examples [00:18, 1030.07 examples/s]18749 examples [00:18, 1021.41 examples/s]18852 examples [00:18, 1021.75 examples/s]18955 examples [00:18, 1012.40 examples/s]19057 examples [00:18, 998.42 examples/s] 19163 examples [00:18, 1015.95 examples/s]19268 examples [00:19, 1025.43 examples/s]19371 examples [00:19, 1006.00 examples/s]19477 examples [00:19, 1020.67 examples/s]19585 examples [00:19, 1035.98 examples/s]19695 examples [00:19, 1052.17 examples/s]19802 examples [00:19, 1056.52 examples/s]19908 examples [00:19, 1036.17 examples/s]20012 examples [00:19, 985.90 examples/s] 20117 examples [00:19, 1003.00 examples/s]20225 examples [00:19, 1022.36 examples/s]20328 examples [00:20, 1015.29 examples/s]20436 examples [00:20, 1032.47 examples/s]20540 examples [00:20, 1007.16 examples/s]20645 examples [00:20, 1016.99 examples/s]20748 examples [00:20, 1018.81 examples/s]20851 examples [00:20, 985.19 examples/s] 20953 examples [00:20, 993.03 examples/s]21055 examples [00:20, 997.09 examples/s]21163 examples [00:20, 1019.26 examples/s]21271 examples [00:21, 1035.30 examples/s]21377 examples [00:21, 1042.46 examples/s]21487 examples [00:21, 1057.30 examples/s]21597 examples [00:21, 1068.45 examples/s]21704 examples [00:21, 1054.77 examples/s]21812 examples [00:21, 1061.66 examples/s]21919 examples [00:21, 1051.99 examples/s]22027 examples [00:21, 1057.67 examples/s]22135 examples [00:21, 1061.58 examples/s]22242 examples [00:21, 1063.29 examples/s]22350 examples [00:22, 1065.57 examples/s]22457 examples [00:22, 1044.79 examples/s]22567 examples [00:22, 1058.18 examples/s]22676 examples [00:22, 1064.88 examples/s]22783 examples [00:22, 1036.87 examples/s]22887 examples [00:22, 1019.40 examples/s]22990 examples [00:22, 997.83 examples/s] 23094 examples [00:22, 1008.76 examples/s]23203 examples [00:22, 1030.95 examples/s]23307 examples [00:22, 1025.28 examples/s]23411 examples [00:23, 1028.74 examples/s]23515 examples [00:23, 976.45 examples/s] 23623 examples [00:23, 1004.19 examples/s]23731 examples [00:23, 1023.57 examples/s]23837 examples [00:23, 1033.31 examples/s]23941 examples [00:23, 1026.87 examples/s]24050 examples [00:23, 1044.86 examples/s]24155 examples [00:23, 1035.15 examples/s]24263 examples [00:23, 1046.66 examples/s]24371 examples [00:23, 1054.30 examples/s]24477 examples [00:24, 1054.27 examples/s]24583 examples [00:24, 1055.63 examples/s]24693 examples [00:24, 1066.10 examples/s]24800 examples [00:24, 1050.68 examples/s]24906 examples [00:24, 1048.07 examples/s]25014 examples [00:24, 1056.70 examples/s]25120 examples [00:24, 1055.56 examples/s]25227 examples [00:24, 1058.95 examples/s]25333 examples [00:24, 1036.07 examples/s]25443 examples [00:25, 1052.22 examples/s]25549 examples [00:25, 1030.57 examples/s]25653 examples [00:25, 1031.15 examples/s]25757 examples [00:25, 1027.14 examples/s]25860 examples [00:25, 1016.51 examples/s]25962 examples [00:25, 1007.27 examples/s]26063 examples [00:25, 992.45 examples/s] 26163 examples [00:25, 980.15 examples/s]26269 examples [00:25, 1001.64 examples/s]26370 examples [00:25, 996.50 examples/s] 26478 examples [00:26, 1017.95 examples/s]26581 examples [00:26, 1017.29 examples/s]26688 examples [00:26, 1031.75 examples/s]26793 examples [00:26, 1036.45 examples/s]26897 examples [00:26, 1033.48 examples/s]27001 examples [00:26, 1016.70 examples/s]27105 examples [00:26, 1021.82 examples/s]27208 examples [00:26, 1021.06 examples/s]27316 examples [00:26, 1035.46 examples/s]27422 examples [00:26, 1040.27 examples/s]27527 examples [00:27, 1023.31 examples/s]27630 examples [00:27, 1013.00 examples/s]27732 examples [00:27, 1011.05 examples/s]27839 examples [00:27, 1026.64 examples/s]27942 examples [00:27, 1022.73 examples/s]28045 examples [00:27, 1015.08 examples/s]28148 examples [00:27, 1019.16 examples/s]28251 examples [00:27, 1021.32 examples/s]28354 examples [00:27, 1016.89 examples/s]28458 examples [00:27, 1023.57 examples/s]28561 examples [00:28, 1020.92 examples/s]28664 examples [00:28, 1020.21 examples/s]28767 examples [00:28, 1008.88 examples/s]28873 examples [00:28, 1020.80 examples/s]28976 examples [00:28, 1013.13 examples/s]29079 examples [00:28, 1016.93 examples/s]29181 examples [00:28, 1015.00 examples/s]29284 examples [00:28, 1018.22 examples/s]29387 examples [00:28, 1015.82 examples/s]29489 examples [00:28, 1000.56 examples/s]29591 examples [00:29, 1005.92 examples/s]29700 examples [00:29, 1027.85 examples/s]29808 examples [00:29, 1041.02 examples/s]29918 examples [00:29, 1056.29 examples/s]30024 examples [00:29, 1012.04 examples/s]30128 examples [00:29, 1020.19 examples/s]30231 examples [00:29, 1019.98 examples/s]30339 examples [00:29, 1029.70 examples/s]30447 examples [00:29, 1043.16 examples/s]30557 examples [00:30, 1059.38 examples/s]30664 examples [00:30, 1059.43 examples/s]30773 examples [00:30, 1067.05 examples/s]30880 examples [00:30, 1039.78 examples/s]30985 examples [00:30, 1038.15 examples/s]31089 examples [00:30, 1035.54 examples/s]31193 examples [00:30, 1028.91 examples/s]31296 examples [00:30, 997.85 examples/s] 31397 examples [00:30, 986.26 examples/s]31496 examples [00:30, 982.67 examples/s]31596 examples [00:31, 987.36 examples/s]31698 examples [00:31, 995.68 examples/s]31802 examples [00:31, 1007.06 examples/s]31907 examples [00:31, 1018.32 examples/s]32010 examples [00:31, 1021.25 examples/s]32120 examples [00:31, 1041.62 examples/s]32227 examples [00:31, 1048.98 examples/s]32333 examples [00:31, 1036.88 examples/s]32437 examples [00:31, 1025.02 examples/s]32540 examples [00:31, 1012.12 examples/s]32642 examples [00:32, 1014.14 examples/s]32744 examples [00:32, 1015.04 examples/s]32846 examples [00:32, 1012.70 examples/s]32948 examples [00:32, 1005.60 examples/s]33049 examples [00:32, 990.33 examples/s] 33156 examples [00:32, 1012.52 examples/s]33258 examples [00:32, 1009.88 examples/s]33360 examples [00:32, 1000.57 examples/s]33468 examples [00:32, 1020.62 examples/s]33576 examples [00:32, 1032.81 examples/s]33680 examples [00:33, 1012.64 examples/s]33782 examples [00:33, 997.49 examples/s] 33882 examples [00:33, 989.12 examples/s]33984 examples [00:33, 996.02 examples/s]34084 examples [00:33, 976.08 examples/s]34188 examples [00:33, 992.43 examples/s]34291 examples [00:33, 1001.25 examples/s]34395 examples [00:33, 1009.93 examples/s]34497 examples [00:33, 998.67 examples/s] 34599 examples [00:34, 1004.83 examples/s]34700 examples [00:34, 999.74 examples/s] 34801 examples [00:34, 998.72 examples/s]34902 examples [00:34, 1001.94 examples/s]35003 examples [00:34, 1001.38 examples/s]35107 examples [00:34, 1010.46 examples/s]35209 examples [00:34, 1002.12 examples/s]35310 examples [00:34, 1001.44 examples/s]35411 examples [00:34, 1001.12 examples/s]35521 examples [00:34, 1027.54 examples/s]35631 examples [00:35, 1047.92 examples/s]35737 examples [00:35, 1049.33 examples/s]35844 examples [00:35, 1054.94 examples/s]35950 examples [00:35, 1046.48 examples/s]36060 examples [00:35, 1059.34 examples/s]36167 examples [00:35, 1043.04 examples/s]36272 examples [00:35, 1011.95 examples/s]36377 examples [00:35, 1022.16 examples/s]36485 examples [00:35, 1038.61 examples/s]36591 examples [00:35, 1044.06 examples/s]36699 examples [00:36, 1053.71 examples/s]36805 examples [00:36, 1034.92 examples/s]36914 examples [00:36, 1049.68 examples/s]37024 examples [00:36, 1064.24 examples/s]37131 examples [00:36, 1063.08 examples/s]37238 examples [00:36, 1056.51 examples/s]37344 examples [00:36, 1046.36 examples/s]37449 examples [00:36, 1039.05 examples/s]37553 examples [00:36, 993.92 examples/s] 37661 examples [00:36, 1017.56 examples/s]37764 examples [00:37, 1019.21 examples/s]37873 examples [00:37, 1036.13 examples/s]37980 examples [00:37, 1045.05 examples/s]38085 examples [00:37, 1003.07 examples/s]38191 examples [00:37, 1017.26 examples/s]38299 examples [00:37, 1033.49 examples/s]38405 examples [00:37, 1038.23 examples/s]38510 examples [00:37, 1028.69 examples/s]38614 examples [00:37, 1024.56 examples/s]38718 examples [00:38, 1028.16 examples/s]38821 examples [00:38, 1027.93 examples/s]38925 examples [00:38, 1030.82 examples/s]39029 examples [00:38, 1022.70 examples/s]39134 examples [00:38, 1027.67 examples/s]39241 examples [00:38, 1039.59 examples/s]39349 examples [00:38, 1050.77 examples/s]39455 examples [00:38, 1036.69 examples/s]39565 examples [00:38, 1053.60 examples/s]39674 examples [00:38, 1061.96 examples/s]39781 examples [00:39, 1034.05 examples/s]39885 examples [00:39, 1015.24 examples/s]39988 examples [00:39, 1017.81 examples/s]40090 examples [00:39, 950.82 examples/s] 40193 examples [00:39, 972.27 examples/s]40298 examples [00:39, 991.49 examples/s]40402 examples [00:39, 1002.99 examples/s]40503 examples [00:39, 1000.15 examples/s]40604 examples [00:39, 992.84 examples/s] 40707 examples [00:39, 1001.26 examples/s]40814 examples [00:40, 1018.72 examples/s]40917 examples [00:40, 992.04 examples/s] 41023 examples [00:40, 1011.26 examples/s]41125 examples [00:40, 1006.45 examples/s]41231 examples [00:40, 1021.40 examples/s]41339 examples [00:40, 1035.85 examples/s]41443 examples [00:40, 1028.66 examples/s]41547 examples [00:40, 1011.87 examples/s]41651 examples [00:40, 1018.30 examples/s]41753 examples [00:40, 1012.27 examples/s]41855 examples [00:41, 980.30 examples/s] 41962 examples [00:41, 1003.14 examples/s]42067 examples [00:41, 1016.03 examples/s]42169 examples [00:41, 1000.62 examples/s]42270 examples [00:41, 978.01 examples/s] 42369 examples [00:41, 967.93 examples/s]42467 examples [00:41, 918.44 examples/s]42562 examples [00:41, 926.83 examples/s]42663 examples [00:41, 950.01 examples/s]42764 examples [00:42, 964.73 examples/s]42871 examples [00:42, 991.84 examples/s]42978 examples [00:42, 1013.74 examples/s]43085 examples [00:42, 1028.21 examples/s]43194 examples [00:42, 1044.50 examples/s]43304 examples [00:42, 1059.64 examples/s]43413 examples [00:42, 1068.42 examples/s]43521 examples [00:42, 1053.27 examples/s]43628 examples [00:42, 1056.18 examples/s]43735 examples [00:42, 1057.48 examples/s]43841 examples [00:43, 1043.76 examples/s]43946 examples [00:43, 1038.43 examples/s]44050 examples [00:43, 1033.06 examples/s]44154 examples [00:43, 1034.09 examples/s]44258 examples [00:43, 1005.87 examples/s]44361 examples [00:43, 1011.86 examples/s]44466 examples [00:43, 1022.72 examples/s]44574 examples [00:43, 1038.22 examples/s]44678 examples [00:43, 1034.49 examples/s]44787 examples [00:43, 1049.46 examples/s]44895 examples [00:44, 1057.80 examples/s]45001 examples [00:44, 1035.74 examples/s]45109 examples [00:44, 1047.28 examples/s]45214 examples [00:44, 1031.87 examples/s]45318 examples [00:44, 1026.11 examples/s]45421 examples [00:44, 1003.32 examples/s]45525 examples [00:44, 1011.63 examples/s]45632 examples [00:44, 1028.25 examples/s]45735 examples [00:44, 1025.54 examples/s]45839 examples [00:44, 1027.48 examples/s]45942 examples [00:45, 1011.76 examples/s]46047 examples [00:45, 1020.45 examples/s]46150 examples [00:45, 1013.81 examples/s]46252 examples [00:45, 1015.27 examples/s]46354 examples [00:45, 1014.46 examples/s]46456 examples [00:45, 1009.76 examples/s]46558 examples [00:45, 988.70 examples/s] 46661 examples [00:45, 1000.03 examples/s]46764 examples [00:45, 1007.10 examples/s]46865 examples [00:46, 1004.82 examples/s]46966 examples [00:46, 999.91 examples/s] 47069 examples [00:46, 1006.71 examples/s]47173 examples [00:46, 1015.46 examples/s]47275 examples [00:46, 1005.28 examples/s]47383 examples [00:46, 1024.17 examples/s]47490 examples [00:46, 1037.45 examples/s]47598 examples [00:46, 1049.67 examples/s]47704 examples [00:46, 1027.63 examples/s]47807 examples [00:46, 1005.80 examples/s]47909 examples [00:47, 1009.93 examples/s]48015 examples [00:47, 1018.23 examples/s]48121 examples [00:47, 1028.78 examples/s]48224 examples [00:47, 1023.74 examples/s]48333 examples [00:47, 1042.42 examples/s]48441 examples [00:47, 1052.17 examples/s]48548 examples [00:47, 1057.02 examples/s]48656 examples [00:47, 1062.31 examples/s]48765 examples [00:47, 1070.03 examples/s]48873 examples [00:47, 1066.91 examples/s]48982 examples [00:48, 1070.22 examples/s]49090 examples [00:48, 1054.66 examples/s]49196 examples [00:48, 1053.90 examples/s]49304 examples [00:48, 1059.74 examples/s]49411 examples [00:48, 1033.66 examples/s]49518 examples [00:48, 1042.27 examples/s]49623 examples [00:48, 1042.38 examples/s]49732 examples [00:48, 1053.94 examples/s]49840 examples [00:48, 1059.85 examples/s]49947 examples [00:48, 1050.63 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5793/50000 [00:00<00:00, 57924.81 examples/s] 37%|███▋      | 18602/50000 [00:00<00:00, 69315.36 examples/s] 62%|██████▏   | 31001/50000 [00:00<00:00, 79882.82 examples/s] 88%|████████▊ | 43948/50000 [00:00<00:00, 90252.30 examples/s]                                                               0 examples [00:00, ? examples/s]84 examples [00:00, 838.13 examples/s]192 examples [00:00, 897.14 examples/s]302 examples [00:00, 948.47 examples/s]410 examples [00:00, 982.66 examples/s]520 examples [00:00, 1014.10 examples/s]628 examples [00:00, 1031.95 examples/s]734 examples [00:00, 1039.46 examples/s]845 examples [00:00, 1058.34 examples/s]952 examples [00:00, 1060.54 examples/s]1056 examples [00:01, 1044.96 examples/s]1166 examples [00:01, 1060.79 examples/s]1276 examples [00:01, 1071.10 examples/s]1383 examples [00:01, 1049.56 examples/s]1488 examples [00:01, 1047.59 examples/s]1594 examples [00:01, 1050.19 examples/s]1699 examples [00:01, 1047.63 examples/s]1804 examples [00:01, 1021.96 examples/s]1907 examples [00:01, 1019.65 examples/s]2011 examples [00:01, 1022.97 examples/s]2114 examples [00:02, 1022.33 examples/s]2217 examples [00:02, 1021.12 examples/s]2320 examples [00:02, 1001.28 examples/s]2426 examples [00:02, 1015.75 examples/s]2532 examples [00:02, 1027.21 examples/s]2638 examples [00:02, 1036.21 examples/s]2742 examples [00:02, 1035.67 examples/s]2846 examples [00:02, 1024.15 examples/s]2949 examples [00:02, 1011.08 examples/s]3054 examples [00:02, 1021.57 examples/s]3163 examples [00:03, 1039.16 examples/s]3273 examples [00:03, 1055.66 examples/s]3381 examples [00:03, 1061.55 examples/s]3488 examples [00:03, 1020.01 examples/s]3598 examples [00:03, 1040.26 examples/s]3705 examples [00:03, 1048.85 examples/s]3815 examples [00:03, 1063.12 examples/s]3925 examples [00:03, 1072.79 examples/s]4036 examples [00:03, 1082.32 examples/s]4145 examples [00:03, 1043.70 examples/s]4250 examples [00:04, 1028.97 examples/s]4354 examples [00:04, 1027.93 examples/s]4458 examples [00:04, 1024.94 examples/s]4564 examples [00:04, 1035.10 examples/s]4669 examples [00:04, 1037.89 examples/s]4773 examples [00:04, 783.39 examples/s] 4876 examples [00:04, 842.25 examples/s]4974 examples [00:04, 878.98 examples/s]5078 examples [00:04, 919.67 examples/s]5181 examples [00:05, 949.19 examples/s]5282 examples [00:05, 964.38 examples/s]5386 examples [00:05, 983.38 examples/s]5488 examples [00:05, 993.14 examples/s]5591 examples [00:05, 1003.39 examples/s]5694 examples [00:05, 1009.75 examples/s]5796 examples [00:05, 980.17 examples/s] 5900 examples [00:05, 995.15 examples/s]6001 examples [00:05, 992.93 examples/s]6106 examples [00:06, 1007.67 examples/s]6208 examples [00:06, 971.43 examples/s] 6316 examples [00:06, 1001.14 examples/s]6426 examples [00:06, 1026.67 examples/s]6534 examples [00:06, 1040.20 examples/s]6642 examples [00:06, 1050.25 examples/s]6748 examples [00:06, 1050.50 examples/s]6854 examples [00:06, 1046.74 examples/s]6963 examples [00:06, 1056.76 examples/s]7069 examples [00:06, 1048.36 examples/s]7174 examples [00:07, 1043.31 examples/s]7279 examples [00:07, 1019.40 examples/s]7382 examples [00:07, 1017.62 examples/s]7484 examples [00:07, 1011.81 examples/s]7586 examples [00:07, 977.56 examples/s] 7685 examples [00:07, 981.00 examples/s]7790 examples [00:07, 1000.10 examples/s]7898 examples [00:07, 1021.58 examples/s]8008 examples [00:07, 1041.60 examples/s]8117 examples [00:07, 1053.66 examples/s]8225 examples [00:08, 1060.48 examples/s]8332 examples [00:08, 1062.95 examples/s]8440 examples [00:08, 1066.49 examples/s]8549 examples [00:08, 1071.16 examples/s]8657 examples [00:08, 1061.28 examples/s]8766 examples [00:08, 1067.76 examples/s]8873 examples [00:08, 1052.60 examples/s]8979 examples [00:08, 1044.14 examples/s]9085 examples [00:08, 1046.93 examples/s]9190 examples [00:08, 1033.73 examples/s]9294 examples [00:09, 1034.46 examples/s]9398 examples [00:09, 1033.46 examples/s]9502 examples [00:09, 1023.21 examples/s]9605 examples [00:09, 985.35 examples/s] 9704 examples [00:09, 951.02 examples/s]9800 examples [00:09, 945.47 examples/s]9895 examples [00:09, 894.39 examples/s]9986 examples [00:09, 855.53 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete47EUP0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete47EUP0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8f84b14510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8f84b14510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8f84b14510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f82b752b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f180252b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8f84b14510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8f84b14510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8f84b14510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f18025160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f6bd60240>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.16 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.16 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.05 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.05 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.30 file/s]2020-09-04 00:09:16.283407: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-04 00:09:16.287628: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-09-04 00:09:16.287794: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55afe1275070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-04 00:09:16.287810: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 81%|████████▏ | 8077312/9912422 [00:00<00:00, 80767805.19it/s]9920512it [00:00, 45652190.26it/s]                             
0it [00:00, ?it/s] 57%|█████▋    | 16384/28881 [00:00<00:00, 156895.93it/s]32768it [00:00, 310262.05it/s]                           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456738.32it/s]1654784it [00:00, 11565850.94it/s]                         
0it [00:00, ?it/s]8192it [00:00, 141718.86it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
