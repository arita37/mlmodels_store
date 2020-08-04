
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdc0a18dae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdc0a18dae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fdc74dfd510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fdc74dfd510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdc9402aea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdc9402aea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc221a5a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc221a5a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc221a5a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 120270.36it/s] 77%|███████▋  | 7651328/9912422 [00:00<00:13, 171698.21it/s]9920512it [00:00, 39135313.00it/s]                           
0it [00:00, ?it/s]32768it [00:00, 588570.89it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162149.96it/s]1654784it [00:00, 11253779.38it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194636.35it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc0a0486d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc0a058978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fdc221a56a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fdc221a56a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fdc221a56a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:07,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:07,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:05,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:45,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:45,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:43,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:43,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:30,  4.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:29,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:29,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:29,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:29,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:28,  4.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:20,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:20,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:19,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:19,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:19,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:19,  6.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  8.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 15.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 21.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 27.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 34.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 42.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 50.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.05 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.05 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.08 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ url]
0 examples [00:00, ? examples/s]2020-08-04 06:09:40.895906: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-04 06:09:40.912815: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-04 06:09:40.913126: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dae258cf60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 06:09:40.913147: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.37 examples/s]91 examples [00:00, 13.33 examples/s]181 examples [00:00, 18.92 examples/s]260 examples [00:00, 26.75 examples/s]350 examples [00:00, 37.73 examples/s]439 examples [00:00, 52.94 examples/s]525 examples [00:00, 73.68 examples/s]613 examples [00:00, 101.59 examples/s]703 examples [00:00, 138.35 examples/s]791 examples [00:01, 185.15 examples/s]883 examples [00:01, 243.38 examples/s]972 examples [00:01, 311.10 examples/s]1062 examples [00:01, 386.72 examples/s]1155 examples [00:01, 468.87 examples/s]1250 examples [00:01, 552.30 examples/s]1346 examples [00:01, 631.77 examples/s]1439 examples [00:01, 693.92 examples/s]1534 examples [00:01, 753.56 examples/s]1627 examples [00:01, 777.32 examples/s]1719 examples [00:02, 814.85 examples/s]1811 examples [00:02, 842.09 examples/s]1902 examples [00:02, 859.50 examples/s]1995 examples [00:02, 877.92 examples/s]2087 examples [00:02, 882.98 examples/s]2178 examples [00:02, 869.70 examples/s]2267 examples [00:02, 868.63 examples/s]2356 examples [00:02, 874.42 examples/s]2445 examples [00:02, 871.55 examples/s]2533 examples [00:02, 859.45 examples/s]2620 examples [00:03, 859.35 examples/s]2713 examples [00:03, 877.78 examples/s]2804 examples [00:03, 886.31 examples/s]2895 examples [00:03, 892.64 examples/s]2985 examples [00:03, 891.97 examples/s]3081 examples [00:03, 910.56 examples/s]3176 examples [00:03, 921.23 examples/s]3269 examples [00:03, 922.52 examples/s]3364 examples [00:03, 927.95 examples/s]3458 examples [00:03, 931.09 examples/s]3555 examples [00:04, 939.89 examples/s]3650 examples [00:04, 937.11 examples/s]3744 examples [00:04, 931.49 examples/s]3838 examples [00:04, 930.36 examples/s]3932 examples [00:04, 916.32 examples/s]4026 examples [00:04, 920.89 examples/s]4120 examples [00:04, 926.36 examples/s]4213 examples [00:04, 926.10 examples/s]4306 examples [00:04, 926.93 examples/s]4400 examples [00:04, 928.20 examples/s]4495 examples [00:05, 932.47 examples/s]4589 examples [00:05, 932.06 examples/s]4683 examples [00:05, 930.62 examples/s]4777 examples [00:05, 921.46 examples/s]4870 examples [00:05, 915.74 examples/s]4966 examples [00:05, 927.71 examples/s]5059 examples [00:05, 925.79 examples/s]5152 examples [00:05, 920.87 examples/s]5245 examples [00:05, 920.44 examples/s]5339 examples [00:05, 925.22 examples/s]5435 examples [00:06, 933.53 examples/s]5529 examples [00:06, 932.81 examples/s]5624 examples [00:06, 935.22 examples/s]5718 examples [00:06, 913.17 examples/s]5810 examples [00:06, 900.01 examples/s]5905 examples [00:06, 912.81 examples/s]5997 examples [00:06, 907.89 examples/s]6091 examples [00:06, 914.68 examples/s]6184 examples [00:06, 916.83 examples/s]6277 examples [00:07, 918.86 examples/s]6372 examples [00:07, 926.90 examples/s]6465 examples [00:07, 911.91 examples/s]6557 examples [00:07, 907.41 examples/s]6648 examples [00:07, 891.09 examples/s]6738 examples [00:07, 856.82 examples/s]6831 examples [00:07, 877.40 examples/s]6922 examples [00:07, 886.12 examples/s]7011 examples [00:07, 884.32 examples/s]7102 examples [00:07, 889.45 examples/s]7192 examples [00:08, 887.99 examples/s]7285 examples [00:08, 898.63 examples/s]7375 examples [00:08, 860.54 examples/s]7462 examples [00:08, 832.98 examples/s]7546 examples [00:08, 834.29 examples/s]7630 examples [00:08, 817.72 examples/s]7718 examples [00:08, 834.83 examples/s]7813 examples [00:08, 865.03 examples/s]7905 examples [00:08, 880.21 examples/s]7994 examples [00:08, 861.73 examples/s]8082 examples [00:09, 865.31 examples/s]8174 examples [00:09, 879.29 examples/s]8265 examples [00:09, 886.55 examples/s]8356 examples [00:09, 891.13 examples/s]8448 examples [00:09, 899.07 examples/s]8541 examples [00:09, 907.86 examples/s]8636 examples [00:09, 917.86 examples/s]8730 examples [00:09, 924.08 examples/s]8826 examples [00:09, 932.95 examples/s]8922 examples [00:09, 940.90 examples/s]9019 examples [00:10, 946.68 examples/s]9115 examples [00:10, 950.33 examples/s]9211 examples [00:10, 951.97 examples/s]9307 examples [00:10, 943.88 examples/s]9402 examples [00:10, 920.23 examples/s]9495 examples [00:10, 917.06 examples/s]9589 examples [00:10, 921.44 examples/s]9682 examples [00:10, 919.45 examples/s]9776 examples [00:10, 923.83 examples/s]9869 examples [00:11, 921.02 examples/s]9962 examples [00:11, 917.75 examples/s]10054 examples [00:11, 847.56 examples/s]10145 examples [00:11, 862.00 examples/s]10237 examples [00:11, 876.98 examples/s]10326 examples [00:11, 878.57 examples/s]10418 examples [00:11, 889.12 examples/s]10510 examples [00:11, 896.95 examples/s]10601 examples [00:11, 899.86 examples/s]10692 examples [00:11, 897.15 examples/s]10783 examples [00:12, 898.58 examples/s]10873 examples [00:12, 895.84 examples/s]10963 examples [00:12, 875.58 examples/s]11058 examples [00:12, 894.62 examples/s]11148 examples [00:12, 867.70 examples/s]11236 examples [00:12, 867.49 examples/s]11323 examples [00:12, 855.27 examples/s]11415 examples [00:12, 873.51 examples/s]11508 examples [00:12, 889.13 examples/s]11602 examples [00:12, 899.69 examples/s]11693 examples [00:13, 867.35 examples/s]11787 examples [00:13, 887.04 examples/s]11877 examples [00:13, 888.82 examples/s]11969 examples [00:13, 895.90 examples/s]12059 examples [00:13, 894.62 examples/s]12149 examples [00:13, 895.00 examples/s]12239 examples [00:13, 870.72 examples/s]12327 examples [00:13, 848.40 examples/s]12414 examples [00:13, 854.20 examples/s]12509 examples [00:14, 880.74 examples/s]12605 examples [00:14, 900.59 examples/s]12701 examples [00:14, 917.32 examples/s]12794 examples [00:14, 896.79 examples/s]12885 examples [00:14, 895.59 examples/s]12979 examples [00:14, 906.52 examples/s]13071 examples [00:14, 908.66 examples/s]13163 examples [00:14, 911.60 examples/s]13256 examples [00:14, 916.31 examples/s]13349 examples [00:14, 917.57 examples/s]13443 examples [00:15, 923.98 examples/s]13536 examples [00:15, 923.66 examples/s]13630 examples [00:15, 927.63 examples/s]13724 examples [00:15, 930.15 examples/s]13818 examples [00:15, 924.49 examples/s]13911 examples [00:15, 920.96 examples/s]14007 examples [00:15, 929.12 examples/s]14100 examples [00:15, 929.15 examples/s]14197 examples [00:15, 938.34 examples/s]14297 examples [00:15, 953.62 examples/s]14393 examples [00:16, 942.25 examples/s]14488 examples [00:16, 943.63 examples/s]14583 examples [00:16, 939.11 examples/s]14678 examples [00:16, 939.65 examples/s]14772 examples [00:16, 935.56 examples/s]14866 examples [00:16, 924.94 examples/s]14959 examples [00:16, 925.34 examples/s]15052 examples [00:16, 902.92 examples/s]15143 examples [00:16, 903.62 examples/s]15234 examples [00:16, 875.76 examples/s]15322 examples [00:17, 876.75 examples/s]15414 examples [00:17, 866.34 examples/s]15507 examples [00:17, 882.06 examples/s]15601 examples [00:17, 897.31 examples/s]15694 examples [00:17, 906.36 examples/s]15785 examples [00:17, 891.68 examples/s]15887 examples [00:17, 925.06 examples/s]15980 examples [00:17, 918.44 examples/s]16073 examples [00:17, 917.85 examples/s]16167 examples [00:18, 922.05 examples/s]16260 examples [00:18, 914.17 examples/s]16352 examples [00:18, 914.00 examples/s]16449 examples [00:18, 927.76 examples/s]16543 examples [00:18, 927.81 examples/s]16636 examples [00:18, 921.26 examples/s]16729 examples [00:18, 885.87 examples/s]16825 examples [00:18, 904.60 examples/s]16921 examples [00:18, 917.02 examples/s]17015 examples [00:18, 921.45 examples/s]17109 examples [00:19, 925.33 examples/s]17204 examples [00:19, 930.00 examples/s]17301 examples [00:19, 941.13 examples/s]17396 examples [00:19, 939.77 examples/s]17491 examples [00:19, 934.62 examples/s]17585 examples [00:19, 925.17 examples/s]17678 examples [00:19, 911.03 examples/s]17770 examples [00:19, 907.94 examples/s]17862 examples [00:19, 908.73 examples/s]17953 examples [00:19, 901.03 examples/s]18044 examples [00:20, 898.36 examples/s]18134 examples [00:20, 876.17 examples/s]18224 examples [00:20, 881.02 examples/s]18316 examples [00:20, 891.03 examples/s]18408 examples [00:20, 899.45 examples/s]18502 examples [00:20, 910.22 examples/s]18594 examples [00:20, 909.13 examples/s]18685 examples [00:20, 908.23 examples/s]18780 examples [00:20, 917.84 examples/s]18872 examples [00:20, 917.43 examples/s]18966 examples [00:21, 923.18 examples/s]19059 examples [00:21, 915.39 examples/s]19151 examples [00:21, 913.06 examples/s]19243 examples [00:21, 911.10 examples/s]19335 examples [00:21, 907.88 examples/s]19426 examples [00:21, 906.26 examples/s]19517 examples [00:21, 901.60 examples/s]19608 examples [00:21, 889.63 examples/s]19698 examples [00:21, 888.98 examples/s]19792 examples [00:21, 902.15 examples/s]19885 examples [00:22, 908.81 examples/s]19976 examples [00:22, 907.23 examples/s]20067 examples [00:22, 848.46 examples/s]20159 examples [00:22, 868.51 examples/s]20248 examples [00:22, 873.01 examples/s]20339 examples [00:22, 883.23 examples/s]20428 examples [00:22, 883.59 examples/s]20517 examples [00:22, 876.66 examples/s]20610 examples [00:22, 890.81 examples/s]20701 examples [00:23, 895.16 examples/s]20791 examples [00:23, 841.27 examples/s]20876 examples [00:23, 838.67 examples/s]20968 examples [00:23, 859.53 examples/s]21060 examples [00:23, 875.06 examples/s]21157 examples [00:23, 899.45 examples/s]21251 examples [00:23, 910.87 examples/s]21346 examples [00:23, 919.67 examples/s]21442 examples [00:23, 929.29 examples/s]21539 examples [00:23, 938.72 examples/s]21634 examples [00:24, 901.20 examples/s]21726 examples [00:24, 906.52 examples/s]21817 examples [00:24, 898.50 examples/s]21912 examples [00:24, 910.91 examples/s]22006 examples [00:24, 918.19 examples/s]22098 examples [00:24, 910.18 examples/s]22190 examples [00:24, 908.98 examples/s]22281 examples [00:24, 900.55 examples/s]22372 examples [00:24, 897.15 examples/s]22462 examples [00:24, 880.09 examples/s]22552 examples [00:25, 885.39 examples/s]22645 examples [00:25, 891.82 examples/s]22735 examples [00:25, 890.63 examples/s]22825 examples [00:25, 882.26 examples/s]22914 examples [00:25, 876.02 examples/s]23006 examples [00:25, 886.91 examples/s]23097 examples [00:25, 893.16 examples/s]23187 examples [00:25, 893.02 examples/s]23285 examples [00:25, 914.90 examples/s]23377 examples [00:26, 911.23 examples/s]23469 examples [00:26, 903.57 examples/s]23561 examples [00:26, 907.78 examples/s]23654 examples [00:26, 912.88 examples/s]23746 examples [00:26, 914.57 examples/s]23838 examples [00:26, 889.82 examples/s]23931 examples [00:26, 898.82 examples/s]24023 examples [00:26, 905.03 examples/s]24114 examples [00:26, 888.34 examples/s]24207 examples [00:26, 899.95 examples/s]24301 examples [00:27, 908.95 examples/s]24394 examples [00:27, 913.51 examples/s]24487 examples [00:27, 917.16 examples/s]24579 examples [00:27, 905.36 examples/s]24674 examples [00:27, 917.24 examples/s]24766 examples [00:27, 912.43 examples/s]24858 examples [00:27, 890.33 examples/s]24948 examples [00:27, 889.26 examples/s]25038 examples [00:27, 880.65 examples/s]25130 examples [00:27, 890.37 examples/s]25220 examples [00:28, 891.69 examples/s]25310 examples [00:28, 890.86 examples/s]25400 examples [00:28, 889.48 examples/s]25490 examples [00:28, 891.12 examples/s]25580 examples [00:28, 889.80 examples/s]25670 examples [00:28, 891.14 examples/s]25760 examples [00:28, 891.59 examples/s]25850 examples [00:28, 883.12 examples/s]25940 examples [00:28, 886.90 examples/s]26031 examples [00:28, 893.49 examples/s]26121 examples [00:29, 891.74 examples/s]26213 examples [00:29, 898.25 examples/s]26303 examples [00:29, 890.62 examples/s]26393 examples [00:29, 887.77 examples/s]26485 examples [00:29, 895.44 examples/s]26576 examples [00:29, 899.46 examples/s]26666 examples [00:29, 884.33 examples/s]26756 examples [00:29, 886.59 examples/s]26845 examples [00:29, 879.76 examples/s]26936 examples [00:29, 887.84 examples/s]27027 examples [00:30, 894.16 examples/s]27117 examples [00:30, 893.12 examples/s]27207 examples [00:30, 894.71 examples/s]27297 examples [00:30, 888.59 examples/s]27386 examples [00:30, 886.84 examples/s]27475 examples [00:30, 872.87 examples/s]27563 examples [00:30, 870.28 examples/s]27651 examples [00:30, 860.49 examples/s]27738 examples [00:30, 854.25 examples/s]27827 examples [00:31, 864.26 examples/s]27919 examples [00:31, 877.83 examples/s]28007 examples [00:31, 869.40 examples/s]28097 examples [00:31, 875.79 examples/s]28194 examples [00:31, 899.88 examples/s]28289 examples [00:31, 913.24 examples/s]28381 examples [00:31, 913.66 examples/s]28478 examples [00:31, 927.53 examples/s]28572 examples [00:31, 930.08 examples/s]28667 examples [00:31, 935.42 examples/s]28761 examples [00:32, 935.56 examples/s]28858 examples [00:32, 944.46 examples/s]28955 examples [00:32, 950.44 examples/s]29052 examples [00:32, 954.03 examples/s]29148 examples [00:32, 927.90 examples/s]29241 examples [00:32, 923.35 examples/s]29334 examples [00:32, 922.58 examples/s]29427 examples [00:32, 923.87 examples/s]29520 examples [00:32, 924.22 examples/s]29613 examples [00:32, 915.16 examples/s]29708 examples [00:33, 922.12 examples/s]29801 examples [00:33, 920.69 examples/s]29894 examples [00:33, 904.48 examples/s]29985 examples [00:33, 884.75 examples/s]30074 examples [00:33, 812.01 examples/s]30165 examples [00:33, 839.09 examples/s]30254 examples [00:33, 851.25 examples/s]30346 examples [00:33, 869.07 examples/s]30435 examples [00:33, 875.08 examples/s]30526 examples [00:33, 882.91 examples/s]30617 examples [00:34, 890.03 examples/s]30709 examples [00:34, 894.77 examples/s]30799 examples [00:34, 890.44 examples/s]30889 examples [00:34, 877.27 examples/s]30977 examples [00:34, 858.70 examples/s]31064 examples [00:34, 854.15 examples/s]31152 examples [00:34, 860.54 examples/s]31242 examples [00:34, 870.02 examples/s]31330 examples [00:34, 870.01 examples/s]31425 examples [00:35, 890.65 examples/s]31521 examples [00:35, 909.91 examples/s]31613 examples [00:35, 888.76 examples/s]31706 examples [00:35, 898.69 examples/s]31799 examples [00:35, 906.83 examples/s]31892 examples [00:35, 912.25 examples/s]31987 examples [00:35, 921.15 examples/s]32080 examples [00:35, 917.15 examples/s]32175 examples [00:35, 925.88 examples/s]32270 examples [00:35, 932.92 examples/s]32364 examples [00:36, 928.76 examples/s]32459 examples [00:36, 932.13 examples/s]32555 examples [00:36, 939.24 examples/s]32652 examples [00:36, 945.23 examples/s]32747 examples [00:36, 929.71 examples/s]32841 examples [00:36, 922.87 examples/s]32934 examples [00:36, 900.28 examples/s]33025 examples [00:36, 866.30 examples/s]33117 examples [00:36, 879.75 examples/s]33206 examples [00:36, 882.12 examples/s]33296 examples [00:37, 886.86 examples/s]33389 examples [00:37, 899.36 examples/s]33480 examples [00:37, 897.02 examples/s]33570 examples [00:37, 897.77 examples/s]33661 examples [00:37, 900.45 examples/s]33753 examples [00:37, 905.77 examples/s]33844 examples [00:37, 893.47 examples/s]33934 examples [00:37, 847.90 examples/s]34020 examples [00:37, 843.01 examples/s]34112 examples [00:37, 862.61 examples/s]34204 examples [00:38, 878.36 examples/s]34295 examples [00:38, 887.10 examples/s]34390 examples [00:38, 903.25 examples/s]34486 examples [00:38, 917.12 examples/s]34580 examples [00:38, 921.46 examples/s]34676 examples [00:38, 930.99 examples/s]34770 examples [00:38, 931.56 examples/s]34864 examples [00:38, 931.65 examples/s]34960 examples [00:38, 938.62 examples/s]35054 examples [00:38, 937.06 examples/s]35149 examples [00:39, 939.09 examples/s]35243 examples [00:39, 925.83 examples/s]35336 examples [00:39, 858.55 examples/s]35426 examples [00:39, 870.46 examples/s]35515 examples [00:39, 874.16 examples/s]35603 examples [00:39, 864.32 examples/s]35693 examples [00:39, 874.19 examples/s]35784 examples [00:39, 884.45 examples/s]35875 examples [00:39, 891.56 examples/s]35965 examples [00:40, 892.20 examples/s]36058 examples [00:40, 902.25 examples/s]36149 examples [00:40, 895.00 examples/s]36239 examples [00:40, 896.03 examples/s]36329 examples [00:40, 879.33 examples/s]36419 examples [00:40, 881.06 examples/s]36511 examples [00:40, 890.94 examples/s]36604 examples [00:40, 900.78 examples/s]36695 examples [00:40, 900.19 examples/s]36786 examples [00:40, 902.49 examples/s]36877 examples [00:41, 901.16 examples/s]36970 examples [00:41, 907.81 examples/s]37061 examples [00:41, 896.37 examples/s]37153 examples [00:41, 903.30 examples/s]37244 examples [00:41, 883.96 examples/s]37334 examples [00:41, 888.35 examples/s]37423 examples [00:41, 887.06 examples/s]37512 examples [00:41, 857.84 examples/s]37599 examples [00:41, 846.74 examples/s]37691 examples [00:41, 865.61 examples/s]37784 examples [00:42, 882.96 examples/s]37875 examples [00:42, 890.52 examples/s]37965 examples [00:42, 866.93 examples/s]38056 examples [00:42, 878.39 examples/s]38145 examples [00:42, 858.83 examples/s]38240 examples [00:42, 883.10 examples/s]38329 examples [00:42, 883.38 examples/s]38420 examples [00:42, 889.23 examples/s]38514 examples [00:42, 903.75 examples/s]38605 examples [00:43, 900.25 examples/s]38698 examples [00:43, 907.45 examples/s]38789 examples [00:43, 907.00 examples/s]38880 examples [00:43, 902.19 examples/s]38971 examples [00:43, 895.69 examples/s]39061 examples [00:43, 892.59 examples/s]39153 examples [00:43, 898.16 examples/s]39243 examples [00:43, 897.38 examples/s]39333 examples [00:43, 894.84 examples/s]39426 examples [00:43, 902.96 examples/s]39517 examples [00:44, 888.25 examples/s]39606 examples [00:44, 884.60 examples/s]39695 examples [00:44, 879.58 examples/s]39784 examples [00:44, 874.52 examples/s]39873 examples [00:44, 879.06 examples/s]39961 examples [00:44, 871.38 examples/s]40049 examples [00:44, 798.61 examples/s]40139 examples [00:44, 824.37 examples/s]40231 examples [00:44, 848.56 examples/s]40317 examples [00:44, 851.45 examples/s]40406 examples [00:45, 861.44 examples/s]40497 examples [00:45, 874.53 examples/s]40585 examples [00:45, 858.67 examples/s]40672 examples [00:45, 856.69 examples/s]40758 examples [00:45, 856.73 examples/s]40845 examples [00:45, 859.34 examples/s]40932 examples [00:45, 855.92 examples/s]41023 examples [00:45, 869.72 examples/s]41117 examples [00:45, 887.20 examples/s]41208 examples [00:45, 891.61 examples/s]41300 examples [00:46, 897.88 examples/s]41394 examples [00:46, 909.43 examples/s]41486 examples [00:46, 889.61 examples/s]41579 examples [00:46, 900.12 examples/s]41670 examples [00:46, 894.27 examples/s]41761 examples [00:46, 898.35 examples/s]41853 examples [00:46, 901.60 examples/s]41944 examples [00:46, 899.25 examples/s]42037 examples [00:46, 907.67 examples/s]42128 examples [00:47, 907.88 examples/s]42221 examples [00:47, 912.15 examples/s]42313 examples [00:47, 896.88 examples/s]42403 examples [00:47, 876.94 examples/s]42495 examples [00:47, 888.91 examples/s]42585 examples [00:47, 882.51 examples/s]42677 examples [00:47, 892.19 examples/s]42768 examples [00:47, 896.87 examples/s]42858 examples [00:47, 892.01 examples/s]42948 examples [00:47, 890.83 examples/s]43038 examples [00:48, 887.78 examples/s]43132 examples [00:48, 901.68 examples/s]43225 examples [00:48, 907.23 examples/s]43317 examples [00:48, 908.93 examples/s]43409 examples [00:48, 912.17 examples/s]43503 examples [00:48, 919.73 examples/s]43600 examples [00:48, 931.91 examples/s]43694 examples [00:48, 904.63 examples/s]43785 examples [00:48, 884.46 examples/s]43878 examples [00:48, 896.98 examples/s]43972 examples [00:49, 907.41 examples/s]44064 examples [00:49, 910.15 examples/s]44157 examples [00:49, 913.18 examples/s]44251 examples [00:49, 919.13 examples/s]44347 examples [00:49, 928.95 examples/s]44440 examples [00:49, 912.18 examples/s]44533 examples [00:49, 916.67 examples/s]44625 examples [00:49, 917.35 examples/s]44717 examples [00:49, 917.79 examples/s]44809 examples [00:49, 891.74 examples/s]44899 examples [00:50, 887.76 examples/s]44988 examples [00:50, 860.12 examples/s]45077 examples [00:50, 867.06 examples/s]45168 examples [00:50, 878.54 examples/s]45263 examples [00:50, 897.07 examples/s]45354 examples [00:50, 898.98 examples/s]45446 examples [00:50, 902.01 examples/s]45540 examples [00:50, 911.14 examples/s]45632 examples [00:50, 908.55 examples/s]45723 examples [00:50, 907.29 examples/s]45816 examples [00:51, 912.16 examples/s]45908 examples [00:51, 910.15 examples/s]46000 examples [00:51, 907.25 examples/s]46092 examples [00:51, 910.98 examples/s]46185 examples [00:51, 915.67 examples/s]46279 examples [00:51, 919.79 examples/s]46371 examples [00:51, 917.07 examples/s]46463 examples [00:51, 917.78 examples/s]46555 examples [00:51, 910.94 examples/s]46647 examples [00:52, 907.00 examples/s]46738 examples [00:52, 891.53 examples/s]46828 examples [00:52, 893.62 examples/s]46920 examples [00:52, 900.51 examples/s]47012 examples [00:52, 905.41 examples/s]47105 examples [00:52, 910.38 examples/s]47197 examples [00:52, 904.62 examples/s]47289 examples [00:52, 908.10 examples/s]47380 examples [00:52, 900.65 examples/s]47476 examples [00:52, 915.79 examples/s]47573 examples [00:53, 929.81 examples/s]47667 examples [00:53, 925.32 examples/s]47762 examples [00:53, 931.91 examples/s]47860 examples [00:53, 944.45 examples/s]47956 examples [00:53, 948.74 examples/s]48054 examples [00:53, 955.93 examples/s]48150 examples [00:53, 937.20 examples/s]48250 examples [00:53, 953.03 examples/s]48348 examples [00:53, 960.68 examples/s]48445 examples [00:53, 959.85 examples/s]48543 examples [00:54, 963.46 examples/s]48640 examples [00:54, 944.21 examples/s]48735 examples [00:54, 911.98 examples/s]48827 examples [00:54, 910.69 examples/s]48919 examples [00:54, 909.93 examples/s]49014 examples [00:54, 920.42 examples/s]49107 examples [00:54, 917.06 examples/s]49201 examples [00:54, 921.58 examples/s]49294 examples [00:54, 920.25 examples/s]49387 examples [00:54, 916.00 examples/s]49479 examples [00:55, 910.78 examples/s]49571 examples [00:55, 876.62 examples/s]49661 examples [00:55, 883.02 examples/s]49752 examples [00:55, 888.93 examples/s]49842 examples [00:55, 891.53 examples/s]49935 examples [00:55, 902.15 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6958/50000 [00:00<00:00, 69575.42 examples/s] 39%|███▉      | 19379/50000 [00:00<00:00, 80150.72 examples/s] 64%|██████▎   | 31813/50000 [00:00<00:00, 89715.36 examples/s] 89%|████████▊ | 44292/50000 [00:00<00:00, 97976.37 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 715.21 examples/s]164 examples [00:00, 766.12 examples/s]256 examples [00:00, 805.91 examples/s]349 examples [00:00, 838.86 examples/s]443 examples [00:00, 864.77 examples/s]530 examples [00:00, 863.92 examples/s]614 examples [00:00, 854.99 examples/s]709 examples [00:00, 879.90 examples/s]803 examples [00:00, 896.54 examples/s]893 examples [00:01, 895.62 examples/s]986 examples [00:01, 903.89 examples/s]1080 examples [00:01, 913.19 examples/s]1173 examples [00:01, 918.17 examples/s]1265 examples [00:01, 908.02 examples/s]1356 examples [00:01, 883.58 examples/s]1446 examples [00:01, 886.64 examples/s]1538 examples [00:01, 895.36 examples/s]1635 examples [00:01, 914.81 examples/s]1733 examples [00:01, 931.75 examples/s]1828 examples [00:02, 934.34 examples/s]1922 examples [00:02, 930.03 examples/s]2017 examples [00:02, 935.75 examples/s]2111 examples [00:02, 927.91 examples/s]2206 examples [00:02, 933.72 examples/s]2300 examples [00:02, 931.65 examples/s]2394 examples [00:02, 930.04 examples/s]2488 examples [00:02, 930.17 examples/s]2582 examples [00:02, 914.28 examples/s]2674 examples [00:02, 909.01 examples/s]2765 examples [00:03, 901.38 examples/s]2856 examples [00:03, 895.95 examples/s]2951 examples [00:03, 909.02 examples/s]3045 examples [00:03, 917.64 examples/s]3140 examples [00:03, 925.82 examples/s]3233 examples [00:03, 921.29 examples/s]3326 examples [00:03, 918.35 examples/s]3420 examples [00:03, 924.45 examples/s]3513 examples [00:03, 901.92 examples/s]3605 examples [00:03, 906.82 examples/s]3697 examples [00:04, 909.17 examples/s]3788 examples [00:04, 888.40 examples/s]3881 examples [00:04, 899.25 examples/s]3976 examples [00:04, 912.39 examples/s]4069 examples [00:04, 916.60 examples/s]4161 examples [00:04, 915.99 examples/s]4253 examples [00:04, 910.86 examples/s]4345 examples [00:04, 910.51 examples/s]4439 examples [00:04, 916.42 examples/s]4534 examples [00:04, 923.70 examples/s]4628 examples [00:05, 925.82 examples/s]4721 examples [00:05, 921.26 examples/s]4814 examples [00:05, 919.94 examples/s]4907 examples [00:05, 909.38 examples/s]4998 examples [00:05, 899.96 examples/s]5089 examples [00:05, 896.48 examples/s]5181 examples [00:05, 903.24 examples/s]5274 examples [00:05, 910.03 examples/s]5367 examples [00:05, 915.35 examples/s]5459 examples [00:06, 903.01 examples/s]5555 examples [00:06, 917.83 examples/s]5647 examples [00:06, 914.32 examples/s]5742 examples [00:06, 922.97 examples/s]5835 examples [00:06, 923.19 examples/s]5931 examples [00:06, 931.76 examples/s]6025 examples [00:06, 916.56 examples/s]6120 examples [00:06, 924.24 examples/s]6214 examples [00:06, 927.75 examples/s]6310 examples [00:06, 935.89 examples/s]6404 examples [00:07, 921.78 examples/s]6497 examples [00:07, 917.30 examples/s]6589 examples [00:07, 916.06 examples/s]6681 examples [00:07, 893.22 examples/s]6777 examples [00:07, 911.80 examples/s]6874 examples [00:07, 926.30 examples/s]6969 examples [00:07, 931.58 examples/s]7065 examples [00:07, 938.58 examples/s]7160 examples [00:07, 941.67 examples/s]7258 examples [00:07, 950.08 examples/s]7354 examples [00:08, 938.81 examples/s]7448 examples [00:08, 929.01 examples/s]7543 examples [00:08, 933.22 examples/s]7637 examples [00:08, 934.11 examples/s]7732 examples [00:08, 938.42 examples/s]7826 examples [00:08, 934.09 examples/s]7920 examples [00:08, 890.30 examples/s]8015 examples [00:08, 905.78 examples/s]8112 examples [00:08, 921.92 examples/s]8209 examples [00:08, 933.66 examples/s]8303 examples [00:09, 907.64 examples/s]8395 examples [00:09, 909.22 examples/s]8487 examples [00:09, 898.62 examples/s]8578 examples [00:09, 893.49 examples/s]8668 examples [00:09, 893.22 examples/s]8761 examples [00:09, 903.73 examples/s]8852 examples [00:09, 904.63 examples/s]8944 examples [00:09, 907.90 examples/s]9037 examples [00:09, 912.53 examples/s]9133 examples [00:09, 924.29 examples/s]9226 examples [00:10, 911.87 examples/s]9318 examples [00:10, 889.96 examples/s]9412 examples [00:10, 903.77 examples/s]9505 examples [00:10, 909.23 examples/s]9599 examples [00:10, 915.67 examples/s]9691 examples [00:10, 905.28 examples/s]9782 examples [00:10, 896.12 examples/s]9872 examples [00:10, 874.95 examples/s]9960 examples [00:10, 874.58 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAPV9CQ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteAPV9CQ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc221a5a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc221a5a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc221a5a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc6db120f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdba839a0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdc221a5a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdc221a5a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdc221a5a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc0a058358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdc0a0581d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.53 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.92 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]2020-08-04 06:10:56.437395: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-04 06:10:56.443079: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-04 06:10:56.443279: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5632c1be2470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-04 06:10:56.443296: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 162262.52it/s] 74%|███████▍  | 7331840/9912422 [00:00<00:11, 231583.46it/s]9920512it [00:00, 44618730.34it/s]                           
0it [00:00, ?it/s]32768it [00:00, 522099.18it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 458927.36it/s]1654784it [00:00, 11654350.99it/s]                         
0it [00:00, ?it/s]8192it [00:00, 204009.80it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
