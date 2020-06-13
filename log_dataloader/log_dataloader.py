
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe6ffa3a6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe6ffa3a6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe76b0020d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe76b0020d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe784d2eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe784d2eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe71887f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe71887f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe71887f950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4038656/9912422 [00:00<00:00, 40286801.20it/s]9920512it [00:00, 39197470.05it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1119583.52it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163478.07it/s]1654784it [00:00, 11571057.05it/s]                         
0it [00:00, ?it/s]8192it [00:00, 221455.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe715dbfb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe715db5da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe71887f598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe71887f598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe71887f598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 43.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 71.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 74.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.71 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.71 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.52 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]
0 examples [00:00, ? examples/s]2020-06-13 06:08:34.788662: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 06:08:34.802350: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-13 06:08:34.802522: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ce33cd7ba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 06:08:34.802539: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 178.95 examples/s]123 examples [00:00, 238.08 examples/s]222 examples [00:00, 308.19 examples/s]323 examples [00:00, 389.06 examples/s]419 examples [00:00, 473.06 examples/s]530 examples [00:00, 570.78 examples/s]634 examples [00:00, 659.32 examples/s]733 examples [00:00, 732.16 examples/s]827 examples [00:00, 781.30 examples/s]932 examples [00:01, 844.61 examples/s]1035 examples [00:01, 890.94 examples/s]1141 examples [00:01, 935.51 examples/s]1242 examples [00:01, 949.71 examples/s]1352 examples [00:01, 989.41 examples/s]1461 examples [00:01, 1016.81 examples/s]1566 examples [00:01, 1011.71 examples/s]1679 examples [00:01, 1044.29 examples/s]1786 examples [00:01, 1021.77 examples/s]1890 examples [00:01, 1025.98 examples/s]1994 examples [00:02, 1020.09 examples/s]2097 examples [00:02, 1022.10 examples/s]2200 examples [00:02, 1022.22 examples/s]2303 examples [00:02, 1024.33 examples/s]2413 examples [00:02, 1044.37 examples/s]2518 examples [00:02, 1028.82 examples/s]2622 examples [00:02, 1024.66 examples/s]2726 examples [00:02, 1028.67 examples/s]2829 examples [00:02, 1024.49 examples/s]2932 examples [00:02, 1008.69 examples/s]3034 examples [00:03, 1009.64 examples/s]3136 examples [00:03, 996.60 examples/s] 3236 examples [00:03, 994.38 examples/s]3336 examples [00:03, 975.85 examples/s]3438 examples [00:03, 987.11 examples/s]3543 examples [00:03, 1003.09 examples/s]3646 examples [00:03, 1010.55 examples/s]3748 examples [00:03, 1009.08 examples/s]3851 examples [00:03, 1014.38 examples/s]3954 examples [00:03, 1016.77 examples/s]4060 examples [00:04, 1029.19 examples/s]4164 examples [00:04, 1030.68 examples/s]4268 examples [00:04, 1026.19 examples/s]4371 examples [00:04, 1013.62 examples/s]4480 examples [00:04, 1035.01 examples/s]4584 examples [00:04, 1008.51 examples/s]4686 examples [00:04, 987.07 examples/s] 4785 examples [00:04, 986.25 examples/s]4893 examples [00:04, 1011.58 examples/s]4996 examples [00:04, 1015.18 examples/s]5098 examples [00:05, 1005.93 examples/s]5199 examples [00:05, 1001.27 examples/s]5309 examples [00:05, 1027.29 examples/s]5412 examples [00:05, 1007.06 examples/s]5518 examples [00:05, 1022.01 examples/s]5621 examples [00:05, 1009.13 examples/s]5725 examples [00:05, 1015.78 examples/s]5830 examples [00:05, 1025.72 examples/s]5933 examples [00:05, 1016.68 examples/s]6036 examples [00:06, 1019.60 examples/s]6141 examples [00:06, 1026.96 examples/s]6244 examples [00:06, 1017.75 examples/s]6346 examples [00:06, 977.09 examples/s] 6445 examples [00:06, 977.89 examples/s]6547 examples [00:06, 989.16 examples/s]6652 examples [00:06, 1005.84 examples/s]6761 examples [00:06, 1028.56 examples/s]6865 examples [00:06, 1029.55 examples/s]6969 examples [00:06, 1022.42 examples/s]7072 examples [00:07, 1014.07 examples/s]7175 examples [00:07, 1016.27 examples/s]7281 examples [00:07, 1025.91 examples/s]7385 examples [00:07, 1027.34 examples/s]7488 examples [00:07, 1028.04 examples/s]7591 examples [00:07, 1026.08 examples/s]7694 examples [00:07, 1022.23 examples/s]7797 examples [00:07, 997.23 examples/s] 7897 examples [00:07, 993.66 examples/s]7997 examples [00:07, 978.38 examples/s]8099 examples [00:08, 988.65 examples/s]8204 examples [00:08, 1005.44 examples/s]8309 examples [00:08, 1017.17 examples/s]8414 examples [00:08, 1024.74 examples/s]8517 examples [00:08, 1003.44 examples/s]8618 examples [00:08, 977.29 examples/s] 8725 examples [00:08, 1002.10 examples/s]8826 examples [00:08, 999.39 examples/s] 8927 examples [00:08, 971.73 examples/s]9028 examples [00:08, 980.90 examples/s]9133 examples [00:09, 998.05 examples/s]9236 examples [00:09, 1006.16 examples/s]9337 examples [00:09, 999.22 examples/s] 9439 examples [00:09, 1002.97 examples/s]9545 examples [00:09, 1017.50 examples/s]9647 examples [00:09, 1016.42 examples/s]9749 examples [00:09, 1003.69 examples/s]9853 examples [00:09, 1013.42 examples/s]9955 examples [00:09, 1013.75 examples/s]10057 examples [00:10, 957.47 examples/s]10155 examples [00:10, 962.45 examples/s]10264 examples [00:10, 997.30 examples/s]10373 examples [00:10, 1022.38 examples/s]10481 examples [00:10, 1035.81 examples/s]10586 examples [00:10, 1032.62 examples/s]10690 examples [00:10, 1031.29 examples/s]10794 examples [00:10, 1031.36 examples/s]10899 examples [00:10, 1034.38 examples/s]11007 examples [00:10, 1045.52 examples/s]11113 examples [00:11, 1047.00 examples/s]11221 examples [00:11, 1054.76 examples/s]11327 examples [00:11, 1041.15 examples/s]11432 examples [00:11, 1021.64 examples/s]11535 examples [00:11, 1020.85 examples/s]11638 examples [00:11, 1014.32 examples/s]11744 examples [00:11, 1026.75 examples/s]11847 examples [00:11, 1023.76 examples/s]11950 examples [00:11, 1014.28 examples/s]12052 examples [00:11, 1002.54 examples/s]12153 examples [00:12, 993.75 examples/s] 12259 examples [00:12, 1011.94 examples/s]12361 examples [00:12, 1001.02 examples/s]12464 examples [00:12, 1007.15 examples/s]12573 examples [00:12, 1027.96 examples/s]12681 examples [00:12, 1041.56 examples/s]12787 examples [00:12, 1044.45 examples/s]12894 examples [00:12, 1049.16 examples/s]13000 examples [00:12, 1027.27 examples/s]13103 examples [00:12, 1020.26 examples/s]13207 examples [00:13, 1023.54 examples/s]13310 examples [00:13, 1021.17 examples/s]13415 examples [00:13, 1028.65 examples/s]13520 examples [00:13, 1033.11 examples/s]13632 examples [00:13, 1055.13 examples/s]13738 examples [00:13, 1027.31 examples/s]13841 examples [00:13, 1017.42 examples/s]13944 examples [00:13, 1020.55 examples/s]14047 examples [00:13, 1018.88 examples/s]14150 examples [00:14, 1021.44 examples/s]14253 examples [00:14, 1016.44 examples/s]14358 examples [00:14, 1023.52 examples/s]14462 examples [00:14, 1028.37 examples/s]14567 examples [00:14, 1032.41 examples/s]14671 examples [00:14, 1016.50 examples/s]14773 examples [00:14, 986.81 examples/s] 14873 examples [00:14, 988.33 examples/s]14976 examples [00:14, 999.10 examples/s]15085 examples [00:14, 1024.35 examples/s]15192 examples [00:15, 1034.87 examples/s]15296 examples [00:15, 1022.75 examples/s]15402 examples [00:15, 1031.75 examples/s]15507 examples [00:15, 1036.11 examples/s]15611 examples [00:15, 1033.86 examples/s]15716 examples [00:15, 1037.23 examples/s]15820 examples [00:15, 1037.10 examples/s]15925 examples [00:15, 1039.43 examples/s]16032 examples [00:15, 1045.85 examples/s]16138 examples [00:15, 1047.57 examples/s]16243 examples [00:16, 1046.55 examples/s]16348 examples [00:16, 1047.17 examples/s]16453 examples [00:16, 1023.75 examples/s]16556 examples [00:16, 1008.08 examples/s]16663 examples [00:16, 1025.29 examples/s]16767 examples [00:16, 1028.91 examples/s]16871 examples [00:16, 1028.38 examples/s]16974 examples [00:16, 1019.07 examples/s]17081 examples [00:16, 1033.50 examples/s]17185 examples [00:16, 1034.13 examples/s]17290 examples [00:17, 1035.84 examples/s]17394 examples [00:17, 1030.59 examples/s]17499 examples [00:17, 1033.61 examples/s]17603 examples [00:17, 1034.77 examples/s]17707 examples [00:17, 1033.37 examples/s]17811 examples [00:17, 1016.68 examples/s]17922 examples [00:17, 1040.20 examples/s]18027 examples [00:17, 1035.21 examples/s]18131 examples [00:17, 1003.66 examples/s]18236 examples [00:17, 1014.95 examples/s]18339 examples [00:18, 1019.28 examples/s]18442 examples [00:18, 1005.38 examples/s]18543 examples [00:18, 1002.22 examples/s]18648 examples [00:18, 1013.69 examples/s]18754 examples [00:18, 1025.35 examples/s]18858 examples [00:18, 1027.16 examples/s]18966 examples [00:18, 1041.10 examples/s]19071 examples [00:18, 993.07 examples/s] 19174 examples [00:18, 1002.94 examples/s]19277 examples [00:19, 1009.75 examples/s]19382 examples [00:19, 1019.41 examples/s]19490 examples [00:19, 1034.91 examples/s]19598 examples [00:19, 1047.97 examples/s]19703 examples [00:19, 1044.79 examples/s]19808 examples [00:19, 1024.69 examples/s]19911 examples [00:19, 1012.32 examples/s]20013 examples [00:19, 982.25 examples/s] 20118 examples [00:19, 1000.51 examples/s]20221 examples [00:19, 1008.83 examples/s]20323 examples [00:20, 1006.40 examples/s]20424 examples [00:20, 994.61 examples/s] 20528 examples [00:20, 1005.48 examples/s]20633 examples [00:20, 1016.06 examples/s]20737 examples [00:20, 1021.23 examples/s]20840 examples [00:20, 1023.12 examples/s]20948 examples [00:20, 1037.23 examples/s]21052 examples [00:20, 1036.28 examples/s]21156 examples [00:20, 1033.38 examples/s]21260 examples [00:20, 1030.05 examples/s]21364 examples [00:21, 1028.30 examples/s]21467 examples [00:21, 998.54 examples/s] 21568 examples [00:21, 965.90 examples/s]21669 examples [00:21, 977.00 examples/s]21777 examples [00:21, 1004.24 examples/s]21885 examples [00:21, 1025.52 examples/s]21992 examples [00:21, 1037.20 examples/s]22099 examples [00:21, 1046.01 examples/s]22204 examples [00:21, 1042.36 examples/s]22309 examples [00:21, 1030.18 examples/s]22413 examples [00:22, 1030.56 examples/s]22517 examples [00:22, 1030.32 examples/s]22626 examples [00:22, 1045.52 examples/s]22733 examples [00:22, 1051.32 examples/s]22842 examples [00:22, 1060.70 examples/s]22950 examples [00:22, 1065.74 examples/s]23057 examples [00:22, 1036.48 examples/s]23161 examples [00:22, 1018.48 examples/s]23264 examples [00:22, 1010.95 examples/s]23371 examples [00:23, 1025.78 examples/s]23479 examples [00:23, 1039.55 examples/s]23590 examples [00:23, 1057.06 examples/s]23696 examples [00:23, 1054.13 examples/s]23802 examples [00:23, 1053.69 examples/s]23908 examples [00:23, 1047.03 examples/s]24018 examples [00:23, 1061.32 examples/s]24125 examples [00:23, 1015.47 examples/s]24228 examples [00:23, 1012.63 examples/s]24330 examples [00:23, 983.73 examples/s] 24429 examples [00:24, 962.06 examples/s]24532 examples [00:24, 980.37 examples/s]24634 examples [00:24, 990.57 examples/s]24734 examples [00:24, 986.99 examples/s]24837 examples [00:24, 998.29 examples/s]24940 examples [00:24, 1007.06 examples/s]25043 examples [00:24, 1012.97 examples/s]25145 examples [00:24, 1014.58 examples/s]25254 examples [00:24, 1035.75 examples/s]25363 examples [00:24, 1048.84 examples/s]25472 examples [00:25, 1060.65 examples/s]25579 examples [00:25, 1060.10 examples/s]25686 examples [00:25, 1041.44 examples/s]25794 examples [00:25, 1050.42 examples/s]25901 examples [00:25, 1054.63 examples/s]26008 examples [00:25, 1057.27 examples/s]26115 examples [00:25, 1058.89 examples/s]26221 examples [00:25, 1044.18 examples/s]26327 examples [00:25, 1047.44 examples/s]26434 examples [00:25, 1051.56 examples/s]26540 examples [00:26, 1034.64 examples/s]26644 examples [00:26, 1024.10 examples/s]26748 examples [00:26, 1026.16 examples/s]26857 examples [00:26, 1043.41 examples/s]26965 examples [00:26, 1052.27 examples/s]27071 examples [00:26, 1044.40 examples/s]27176 examples [00:26, 1038.02 examples/s]27281 examples [00:26, 1041.31 examples/s]27388 examples [00:26, 1047.76 examples/s]27495 examples [00:26, 1052.42 examples/s]27601 examples [00:27, 1035.77 examples/s]27705 examples [00:27, 1024.83 examples/s]27809 examples [00:27, 1026.11 examples/s]27913 examples [00:27, 1029.27 examples/s]28020 examples [00:27, 1040.72 examples/s]28128 examples [00:27, 1050.53 examples/s]28235 examples [00:27, 1053.94 examples/s]28341 examples [00:27, 1035.10 examples/s]28445 examples [00:27, 1002.70 examples/s]28548 examples [00:28, 1008.33 examples/s]28658 examples [00:28, 1032.07 examples/s]28762 examples [00:28, 1019.72 examples/s]28867 examples [00:28, 1028.38 examples/s]28978 examples [00:28, 1049.70 examples/s]29084 examples [00:28, 1025.60 examples/s]29187 examples [00:28, 1021.02 examples/s]29290 examples [00:28, 1010.57 examples/s]29396 examples [00:28, 1023.13 examples/s]29499 examples [00:28, 1019.87 examples/s]29602 examples [00:29, 1005.79 examples/s]29703 examples [00:29, 983.61 examples/s] 29802 examples [00:29, 981.69 examples/s]29911 examples [00:29, 1010.43 examples/s]30013 examples [00:29, 966.99 examples/s] 30111 examples [00:29, 959.27 examples/s]30215 examples [00:29, 979.29 examples/s]30316 examples [00:29, 985.98 examples/s]30418 examples [00:29, 995.42 examples/s]30519 examples [00:29, 997.57 examples/s]30620 examples [00:30, 1001.18 examples/s]30727 examples [00:30, 1020.36 examples/s]30831 examples [00:30, 1024.22 examples/s]30934 examples [00:30, 1015.54 examples/s]31037 examples [00:30, 1017.72 examples/s]31143 examples [00:30, 1027.36 examples/s]31248 examples [00:30, 1033.38 examples/s]31352 examples [00:30, 962.22 examples/s] 31450 examples [00:30, 941.97 examples/s]31548 examples [00:31, 947.57 examples/s]31650 examples [00:31, 968.07 examples/s]31748 examples [00:31, 939.74 examples/s]31843 examples [00:31, 922.88 examples/s]31940 examples [00:31, 934.42 examples/s]32034 examples [00:31, 935.93 examples/s]32129 examples [00:31, 937.92 examples/s]32223 examples [00:31, 934.15 examples/s]32324 examples [00:31, 955.45 examples/s]32425 examples [00:31, 970.43 examples/s]32526 examples [00:32, 981.04 examples/s]32626 examples [00:32, 985.51 examples/s]32737 examples [00:32, 1018.46 examples/s]32841 examples [00:32, 1023.95 examples/s]32953 examples [00:32, 1049.00 examples/s]33059 examples [00:32, 1050.91 examples/s]33165 examples [00:32, 1041.08 examples/s]33276 examples [00:32, 1059.14 examples/s]33383 examples [00:32, 1026.52 examples/s]33487 examples [00:32, 1030.10 examples/s]33591 examples [00:33, 1024.74 examples/s]33700 examples [00:33, 1042.69 examples/s]33809 examples [00:33, 1054.32 examples/s]33915 examples [00:33, 1048.53 examples/s]34023 examples [00:33, 1056.89 examples/s]34132 examples [00:33, 1064.88 examples/s]34240 examples [00:33, 1066.83 examples/s]34347 examples [00:33, 1057.42 examples/s]34453 examples [00:33, 1031.84 examples/s]34562 examples [00:33, 1046.79 examples/s]34672 examples [00:34, 1060.48 examples/s]34780 examples [00:34, 1066.26 examples/s]34887 examples [00:34, 1052.30 examples/s]34993 examples [00:34, 1040.95 examples/s]35098 examples [00:34, 1025.72 examples/s]35201 examples [00:34, 1013.70 examples/s]35308 examples [00:34, 1029.59 examples/s]35412 examples [00:34, 1015.73 examples/s]35514 examples [00:34, 1010.22 examples/s]35620 examples [00:35, 1023.42 examples/s]35725 examples [00:35, 1030.45 examples/s]35829 examples [00:35, 1033.22 examples/s]35935 examples [00:35, 1039.75 examples/s]36040 examples [00:35, 1035.63 examples/s]36145 examples [00:35, 1039.64 examples/s]36253 examples [00:35, 1049.75 examples/s]36359 examples [00:35, 1046.85 examples/s]36464 examples [00:35, 1040.41 examples/s]36569 examples [00:35, 1035.22 examples/s]36676 examples [00:36, 1043.67 examples/s]36781 examples [00:36, 1022.68 examples/s]36884 examples [00:36, 1012.73 examples/s]36989 examples [00:36, 1022.29 examples/s]37094 examples [00:36, 1029.28 examples/s]37198 examples [00:36, 1027.31 examples/s]37303 examples [00:36, 1031.89 examples/s]37407 examples [00:36, 1032.57 examples/s]37515 examples [00:36, 1044.40 examples/s]37620 examples [00:36, 1017.35 examples/s]37722 examples [00:37, 1017.97 examples/s]37827 examples [00:37, 1027.23 examples/s]37934 examples [00:37, 1038.36 examples/s]38041 examples [00:37, 1045.03 examples/s]38146 examples [00:37, 1030.73 examples/s]38250 examples [00:37, 1022.16 examples/s]38356 examples [00:37, 1032.15 examples/s]38460 examples [00:37, 1023.32 examples/s]38563 examples [00:37, 1007.45 examples/s]38664 examples [00:37, 1007.16 examples/s]38771 examples [00:38, 1023.31 examples/s]38874 examples [00:38, 1023.76 examples/s]38981 examples [00:38, 1036.95 examples/s]39085 examples [00:38, 1035.39 examples/s]39189 examples [00:38, 1020.26 examples/s]39292 examples [00:38, 1018.87 examples/s]39394 examples [00:38, 1011.29 examples/s]39501 examples [00:38, 1026.95 examples/s]39610 examples [00:38, 1043.08 examples/s]39715 examples [00:38, 1033.96 examples/s]39821 examples [00:39, 1041.11 examples/s]39926 examples [00:39, 1032.17 examples/s]40030 examples [00:39, 961.97 examples/s] 40128 examples [00:39, 907.48 examples/s]40221 examples [00:39, 904.40 examples/s]40332 examples [00:39, 955.61 examples/s]40433 examples [00:39, 970.04 examples/s]40536 examples [00:39, 987.14 examples/s]40636 examples [00:39, 986.81 examples/s]40741 examples [00:40, 1003.01 examples/s]40848 examples [00:40, 1020.79 examples/s]40957 examples [00:40, 1038.39 examples/s]41062 examples [00:40, 1032.31 examples/s]41166 examples [00:40, 1018.68 examples/s]41269 examples [00:40, 987.74 examples/s] 41369 examples [00:40, 981.12 examples/s]41468 examples [00:40, 948.17 examples/s]41564 examples [00:40, 941.27 examples/s]41665 examples [00:40, 960.55 examples/s]41762 examples [00:41, 929.45 examples/s]41860 examples [00:41, 943.64 examples/s]41958 examples [00:41, 953.81 examples/s]42058 examples [00:41, 964.87 examples/s]42164 examples [00:41, 990.47 examples/s]42271 examples [00:41, 1012.40 examples/s]42379 examples [00:41, 1031.55 examples/s]42486 examples [00:41, 1040.88 examples/s]42593 examples [00:41, 1049.28 examples/s]42699 examples [00:42, 1026.30 examples/s]42803 examples [00:42, 1028.05 examples/s]42906 examples [00:42, 1013.64 examples/s]43013 examples [00:42, 1028.29 examples/s]43120 examples [00:42, 1038.96 examples/s]43225 examples [00:42, 1031.63 examples/s]43330 examples [00:42, 1034.23 examples/s]43434 examples [00:42, 1030.52 examples/s]43538 examples [00:42, 1023.70 examples/s]43644 examples [00:42, 1033.80 examples/s]43748 examples [00:43, 1003.22 examples/s]43855 examples [00:43, 1021.16 examples/s]43958 examples [00:43, 1021.01 examples/s]44064 examples [00:43, 1032.15 examples/s]44168 examples [00:43, 1033.37 examples/s]44272 examples [00:43, 1032.66 examples/s]44383 examples [00:43, 1053.06 examples/s]44489 examples [00:43, 1051.98 examples/s]44595 examples [00:43, 1025.91 examples/s]44698 examples [00:43, 1018.02 examples/s]44804 examples [00:44, 1028.99 examples/s]44908 examples [00:44, 1026.98 examples/s]45011 examples [00:44, 1026.37 examples/s]45114 examples [00:44, 1015.21 examples/s]45216 examples [00:44, 1009.37 examples/s]45317 examples [00:44, 984.90 examples/s] 45417 examples [00:44, 988.60 examples/s]45521 examples [00:44, 1003.46 examples/s]45625 examples [00:44, 1013.35 examples/s]45727 examples [00:44, 999.91 examples/s] 45828 examples [00:45, 996.44 examples/s]45933 examples [00:45, 1010.74 examples/s]46039 examples [00:45, 1022.88 examples/s]46147 examples [00:45, 1038.14 examples/s]46254 examples [00:45, 1046.04 examples/s]46359 examples [00:45, 1032.27 examples/s]46465 examples [00:45, 1040.02 examples/s]46571 examples [00:45, 1044.00 examples/s]46683 examples [00:45, 1063.45 examples/s]46793 examples [00:45, 1073.05 examples/s]46901 examples [00:46, 1025.41 examples/s]47005 examples [00:46, 1021.92 examples/s]47108 examples [00:46, 1016.90 examples/s]47214 examples [00:46, 1026.20 examples/s]47317 examples [00:46, 1022.65 examples/s]47420 examples [00:46, 1011.88 examples/s]47525 examples [00:46, 1020.83 examples/s]47632 examples [00:46, 1033.89 examples/s]47736 examples [00:46, 1035.28 examples/s]47841 examples [00:47, 1039.42 examples/s]47946 examples [00:47, 1021.74 examples/s]48049 examples [00:47, 1023.90 examples/s]48153 examples [00:47, 1027.93 examples/s]48261 examples [00:47, 1042.31 examples/s]48368 examples [00:47, 1050.37 examples/s]48474 examples [00:47, 1030.57 examples/s]48578 examples [00:47, 1028.46 examples/s]48682 examples [00:47, 1031.47 examples/s]48795 examples [00:47, 1058.93 examples/s]48902 examples [00:48, 1054.41 examples/s]49008 examples [00:48, 1035.43 examples/s]49114 examples [00:48, 1042.07 examples/s]49219 examples [00:48, 1013.17 examples/s]49327 examples [00:48, 1031.45 examples/s]49431 examples [00:48, 1031.64 examples/s]49540 examples [00:48, 1046.08 examples/s]49645 examples [00:48, 1012.73 examples/s]49752 examples [00:48, 1028.78 examples/s]49856 examples [00:48, 1025.34 examples/s]49962 examples [00:49, 1033.77 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7458/50000 [00:00<00:00, 74579.36 examples/s] 39%|███▉      | 19685/50000 [00:00<00:00, 84461.86 examples/s] 62%|██████▏   | 30934/50000 [00:00<00:00, 91285.25 examples/s] 86%|████████▌ | 43102/50000 [00:00<00:00, 98678.08 examples/s]                                                               0 examples [00:00, ? examples/s]82 examples [00:00, 815.26 examples/s]182 examples [00:00, 862.45 examples/s]284 examples [00:00, 901.04 examples/s]387 examples [00:00, 936.09 examples/s]488 examples [00:00, 954.66 examples/s]587 examples [00:00, 963.33 examples/s]676 examples [00:00, 938.53 examples/s]768 examples [00:00, 930.86 examples/s]879 examples [00:00, 977.10 examples/s]985 examples [00:01, 999.64 examples/s]1089 examples [00:01, 1009.10 examples/s]1194 examples [00:01, 1018.61 examples/s]1298 examples [00:01, 1023.26 examples/s]1400 examples [00:01, 805.56 examples/s] 1496 examples [00:01, 845.28 examples/s]1599 examples [00:01, 891.66 examples/s]1694 examples [00:01, 906.46 examples/s]1800 examples [00:01, 946.78 examples/s]1905 examples [00:02, 974.45 examples/s]2007 examples [00:02, 986.55 examples/s]2110 examples [00:02, 997.79 examples/s]2211 examples [00:02, 993.15 examples/s]2312 examples [00:02, 984.46 examples/s]2411 examples [00:02, 954.67 examples/s]2509 examples [00:02, 961.47 examples/s]2606 examples [00:02, 954.09 examples/s]2709 examples [00:02, 973.48 examples/s]2814 examples [00:02, 994.70 examples/s]2915 examples [00:03, 998.46 examples/s]3016 examples [00:03, 1001.45 examples/s]3121 examples [00:03, 1015.34 examples/s]3228 examples [00:03, 1029.64 examples/s]3337 examples [00:03, 1044.84 examples/s]3445 examples [00:03, 1054.52 examples/s]3551 examples [00:03, 1052.91 examples/s]3657 examples [00:03, 1016.73 examples/s]3760 examples [00:03, 1017.50 examples/s]3867 examples [00:03, 1030.21 examples/s]3976 examples [00:04, 1046.88 examples/s]4081 examples [00:04, 1038.76 examples/s]4191 examples [00:04, 1054.73 examples/s]4297 examples [00:04, 1023.39 examples/s]4401 examples [00:04, 1027.56 examples/s]4505 examples [00:04, 1029.96 examples/s]4609 examples [00:04, 1016.57 examples/s]4711 examples [00:04, 991.98 examples/s] 4811 examples [00:04, 987.51 examples/s]4916 examples [00:04, 1004.47 examples/s]5021 examples [00:05, 1015.67 examples/s]5125 examples [00:05, 1021.61 examples/s]5228 examples [00:05, 1016.72 examples/s]5330 examples [00:05, 984.12 examples/s] 5434 examples [00:05, 999.76 examples/s]5536 examples [00:05, 1004.37 examples/s]5637 examples [00:05, 999.10 examples/s] 5738 examples [00:05, 985.31 examples/s]5841 examples [00:05, 996.72 examples/s]5945 examples [00:05, 1007.55 examples/s]6049 examples [00:06, 1016.45 examples/s]6158 examples [00:06, 1036.85 examples/s]6266 examples [00:06, 1047.37 examples/s]6371 examples [00:06, 1030.53 examples/s]6475 examples [00:06, 1026.73 examples/s]6583 examples [00:06, 1040.50 examples/s]6689 examples [00:06, 1046.05 examples/s]6794 examples [00:06, 1045.56 examples/s]6905 examples [00:06, 1063.24 examples/s]7012 examples [00:07, 1061.95 examples/s]7120 examples [00:07, 1066.81 examples/s]7227 examples [00:07, 1034.04 examples/s]7334 examples [00:07, 1044.41 examples/s]7439 examples [00:07, 1041.47 examples/s]7544 examples [00:07, 1038.86 examples/s]7655 examples [00:07, 1057.97 examples/s]7761 examples [00:07, 1047.04 examples/s]7866 examples [00:07, 1042.76 examples/s]7974 examples [00:07, 1050.96 examples/s]8080 examples [00:08, 1043.71 examples/s]8185 examples [00:08, 1026.23 examples/s]8288 examples [00:08, 999.06 examples/s] 8392 examples [00:08, 1009.54 examples/s]8495 examples [00:08, 1015.37 examples/s]8598 examples [00:08, 1019.09 examples/s]8701 examples [00:08, 1017.09 examples/s]8812 examples [00:08, 1041.42 examples/s]8917 examples [00:08, 1000.39 examples/s]9020 examples [00:08, 1008.10 examples/s]9127 examples [00:09, 1024.28 examples/s]9230 examples [00:09, 993.22 examples/s] 9335 examples [00:09, 1009.02 examples/s]9441 examples [00:09, 1021.95 examples/s]9546 examples [00:09, 1028.76 examples/s]9651 examples [00:09, 1032.41 examples/s]9755 examples [00:09, 1032.27 examples/s]9859 examples [00:09, 1020.42 examples/s]9962 examples [00:09, 1008.18 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteL2DJCM/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteL2DJCM/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe71887f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe71887f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe71887f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe6a2dbb898>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe6a1d1df28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe71887f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe71887f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe71887f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe71886ac50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe6f6479128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe6902a82f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe6902a82f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fe6902a82f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe6902a8400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe6902a8400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe6902a8400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=638e36e48426c4842bbbfda8d91189c87e5b61dbf1512828519f5601b4d7838a
  Stored in directory: /tmp/pip-ephem-wheel-cache-q_adr93z/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:11:47, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:22:17, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:06:52, 23.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:05:19, 33.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<4:56:58, 48.2kB/s].vector_cache/glove.6B.zip:   1%|          | 7.95M/862M [00:01<3:26:55, 68.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:24:13, 98.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:40:23, 140kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<1:09:51, 200kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<48:52, 285kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:01<34:03, 406kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:01<23:53, 577kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<16:42, 819kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:02<11:42, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<08:16, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<06:41, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:52, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:17, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<03:50, 3.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<29:57, 446kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<22:41, 588kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<16:17, 818kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<14:00, 949kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<12:56, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<09:41, 1.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<06:58, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<09:20, 1.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<07:54, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<07:08, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<07:43, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<05:57, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<04:23, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:28, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<04:56, 2.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:28, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<07:12, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<05:36, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<04:07, 3.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<07:43, 1.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<06:32, 1.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<04:50, 2.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<03:33, 3.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<36:18, 355kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<28:04, 458kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<20:17, 634kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<16:13, 789kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<12:39, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<09:10, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:21, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<09:15, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<07:08, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:08, 2.46MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:34:03, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:07:07, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<47:13, 268kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<35:54, 351kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<27:42, 455kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<19:56, 631kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<14:03, 892kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<21:34, 581kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<16:23, 764kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<11:46, 1.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:07, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:03, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:38, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:34, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:34, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:54, 2.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:20, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:41, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:17, 2.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:53, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:24, 2.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<04:05, 2.99MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:44, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:29, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:05, 2.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:40, 3.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:19, 742kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:41, 955kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<09:10, 1.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:13, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:55, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:51, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:43, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:57, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:28, 2.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:53, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:32, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:10, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:31, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:04, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:48, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:00, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:47, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:10, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:55, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:33, 3.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<11:13:17, 17.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<7:52:13, 24.6kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<5:30:06, 35.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<3:53:04, 49.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<2:44:13, 70.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<1:54:57, 100kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:22:55, 139kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<1:00:22, 190kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<42:47, 268kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<31:40, 361kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<23:21, 489kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<16:36, 687kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<14:13, 799kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<11:07, 1.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:00, 1.42MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:17, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:57, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:08, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:14, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:44, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:17, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:29, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:00, 2.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:47, 2.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:54, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:37, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:21, 3.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:26, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:43, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<05:40, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:32, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:48, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:15, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:48, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:29, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:46, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<05:42, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:31, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:51, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:15, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:49, 2.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:02, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:33, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:48, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<03:30, 3.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<29:22, 363kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<22:44, 470kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<16:22, 651kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<11:31, 921kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<21:22, 496kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<16:02, 661kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<11:25, 926kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:27, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:23, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<06:07, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:45, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:34, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:48, 2.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:04, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:48, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:37, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:48, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:34, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:21, 3.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:49, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:18, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<03:58, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:52, 2.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:37, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:32, 2.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:33, 2.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:40, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:34, 2.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:20, 3.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<09:38, 1.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:56, 1.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:07, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:36, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:07, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:44, 2.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:50, 1.70MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:16, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:55, 2.52MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:45, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:31, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:17, 2.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:59, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:53, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:24, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:05, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:49, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:40, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:22, 2.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:24, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:01, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:04, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:49, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:37, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:33, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:27, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:13, 2.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:10, 1.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:50, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:03, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:27, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:57, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:41, 2.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:41, 3.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<39:04, 238kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<29:37, 314kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<21:12, 438kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<14:53, 620kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<14:47, 623kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<11:27, 805kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:16, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:39, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:36, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:52, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<09:33, 951kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:46, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:42, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:50, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:09, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:50, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:31, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:21, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:16, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:06, 2.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<08:32, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:02, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:10, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:26, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:57, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:36, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:22, 2.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:32, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:23, 2.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:09, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:56, 2.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:58, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:51, 2.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:47, 1.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:49, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<02:45, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:18, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:07, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:29, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:53, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:28, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:16, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<03:04, 2.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<06:17, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:25, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:00, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:07, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:05, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:57, 2.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<07:13, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:02, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:26, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:46, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:14, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:04, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:56, 2.76MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:05, 1.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<05:14, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:51, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:21, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:02, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:55, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:52, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<04:34, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:10, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:08, 2.52MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:49, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:31, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:34, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:36, 3.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:46, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:16, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:11, 2.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:49, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:34, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:37, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:38, 2.92MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:48, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<05:41, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:11, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:29, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:05, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:02, 2.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:42, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:29, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:37, 2.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:22, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:06, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:15, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:23, 3.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:11, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:49, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:51, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:30, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:20, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:26, 2.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:43, 1.96MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:48, 2.58MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:16, 2.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:30, 2.87MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:12, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:59, 1.79MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:09, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<02:17, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:23, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:36, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:24, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:49, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:19, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:29, 2.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:22, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:17, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:53, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:07, 1.66MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:29, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:33, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<02:34, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:40, 1.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:25, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:58, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:14, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:36, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:33, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:35, 2.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:28, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:35, 2.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:10, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:49, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:04, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:38, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:43, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:28, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:44, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:10, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:18, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<02:23, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:01, 1.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:59, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:38, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:50, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:15, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:19, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:24, 2.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:41, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:44, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:29, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:42, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:02, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:08, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:16, 2.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:12, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:39, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:44, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:09, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:40, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:06, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:11, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:20, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:12, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:27, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:47, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:00, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<02:09, 2.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:24, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:29, 1.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:16, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:28, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:50, 1.50MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:00, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:16, 2.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:34, 2.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:57, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:30, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:08, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:31, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:50, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:18, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:22, 1.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:11, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:21, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:59, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:13, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:40, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:30, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:48, 2.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:33, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:49, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:48, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:02, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:46, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:03, 2.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:30, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:23, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:44, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:56, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:03, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:59, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:07, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:47, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:04, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:28, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:19, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:46, 2.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:14, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:43, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:12, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:35, 3.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:21, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:37, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:40, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:50, 1.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:33, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:54, 2.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:18, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:10, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:35, 2.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:32, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:20, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:46, 2.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:10, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:35, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:05, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:30, 3.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:03, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:19, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:39, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:15, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:37, 2.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:47, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:28, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:50, 2.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:09, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:30, 2.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:58, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:24, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:56, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:24, 2.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<03:57, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:16, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:23, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:30, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:42, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:08, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:32, 2.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:35, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:59, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:10, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:19, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:35, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:00, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:27, 2.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:11, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:00, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:29, 2.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:49, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<02:12, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:46, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:16, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:36, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:58, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:09, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:15, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:26, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:55, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:23, 2.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:33, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:54, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:07, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:12, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:57, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:28, 2.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:44, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:03, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:36, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:10, 2.92MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:59, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:47, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:20, 2.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:37, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:54, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:30, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:05, 3.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:27, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:06, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:32, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:44, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:59, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:33, 2.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:07, 2.79MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:52, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:41, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:15, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:29, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:47, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:24, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:00, 2.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:51, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:39, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:13, 2.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:26, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:41, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:19, 2.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<00:57, 2.98MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:51, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:38, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:24, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:18, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:59, 2.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:14, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:29, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:12, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:52, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:31, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:04, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:30, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:04, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:49, 671kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:59, 858kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:08, 1.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:59, 1.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:56, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<01:03, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:44, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:30, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:06, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:14, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:27, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:08, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:49, 2.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:10, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:04, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:48, 2.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:13, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:41, 3.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:15, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:06, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:49, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:13, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:41, 2.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:06, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:00, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:45, 2.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:55, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:52, 2.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:39, 2.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:49, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:36, 3.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:47, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:46, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:33, 3.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:04, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:57, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:42, 2.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:49, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:59, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:41<00:32, 2.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:03, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:55, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:40, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:46, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:55, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:31, 2.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:21, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:05, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:47, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:50, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:54, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:30, 2.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:10, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:58, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:42, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:44, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:38, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:27, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:06, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:54, 1.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:41, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:43, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:33, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:23, 2.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:45, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:39, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:28, 2.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:31, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:35, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:19, 2.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:35, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:22, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:16, 2.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:41, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:34, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:24, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:22, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 2.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:16, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:11, 2.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 3.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 1.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.78MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<19:03:05,  5.83it/s]  0%|          | 840/400000 [00:00<13:18:43,  8.33it/s]  0%|          | 1635/400000 [00:00<9:18:14, 11.89it/s]  1%|          | 2483/400000 [00:00<6:30:10, 16.98it/s]  1%|          | 3344/400000 [00:00<4:32:45, 24.24it/s]  1%|          | 4191/400000 [00:00<3:10:45, 34.58it/s]  1%|▏         | 5059/400000 [00:00<2:13:27, 49.32it/s]  1%|▏         | 5843/400000 [00:00<1:33:29, 70.27it/s]  2%|▏         | 6677/400000 [00:00<1:05:32, 100.02it/s]  2%|▏         | 7554/400000 [00:01<46:00, 142.19it/s]    2%|▏         | 8387/400000 [00:01<32:22, 201.65it/s]  2%|▏         | 9263/400000 [00:01<22:49, 285.25it/s]  3%|▎         | 10098/400000 [00:01<16:11, 401.46it/s]  3%|▎         | 10979/400000 [00:01<11:31, 562.52it/s]  3%|▎         | 11852/400000 [00:01<08:16, 781.99it/s]  3%|▎         | 12714/400000 [00:01<06:00, 1075.30it/s]  3%|▎         | 13592/400000 [00:01<04:24, 1459.49it/s]  4%|▎         | 14454/400000 [00:01<03:18, 1938.93it/s]  4%|▍         | 15307/400000 [00:01<02:32, 2520.46it/s]  4%|▍         | 16156/400000 [00:02<02:00, 3191.46it/s]  4%|▍         | 17003/400000 [00:02<01:38, 3902.67it/s]  4%|▍         | 17888/400000 [00:02<01:21, 4688.71it/s]  5%|▍         | 18740/400000 [00:02<01:11, 5360.69it/s]  5%|▍         | 19614/400000 [00:02<01:02, 6063.61it/s]  5%|▌         | 20491/400000 [00:02<00:56, 6682.24it/s]  5%|▌         | 21348/400000 [00:02<00:53, 7128.93it/s]  6%|▌         | 22220/400000 [00:02<00:50, 7541.32it/s]  6%|▌         | 23079/400000 [00:02<00:48, 7776.33it/s]  6%|▌         | 23967/400000 [00:02<00:46, 8075.60it/s]  6%|▌         | 24831/400000 [00:03<00:45, 8201.96it/s]  6%|▋         | 25691/400000 [00:03<00:45, 8239.01it/s]  7%|▋         | 26572/400000 [00:03<00:44, 8402.12it/s]  7%|▋         | 27433/400000 [00:03<00:44, 8440.65it/s]  7%|▋         | 28292/400000 [00:03<00:43, 8448.69it/s]  7%|▋         | 29147/400000 [00:03<00:43, 8430.42it/s]  7%|▋         | 29997/400000 [00:03<00:44, 8246.43it/s]  8%|▊         | 30851/400000 [00:03<00:44, 8330.88it/s]  8%|▊         | 31689/400000 [00:03<00:44, 8320.37it/s]  8%|▊         | 32572/400000 [00:04<00:43, 8465.76it/s]  8%|▊         | 33437/400000 [00:04<00:43, 8518.78it/s]  9%|▊         | 34291/400000 [00:04<00:43, 8416.09it/s]  9%|▉         | 35182/400000 [00:04<00:42, 8558.02it/s]  9%|▉         | 36040/400000 [00:04<00:42, 8553.13it/s]  9%|▉         | 36897/400000 [00:04<00:42, 8557.13it/s]  9%|▉         | 37761/400000 [00:04<00:42, 8580.36it/s] 10%|▉         | 38620/400000 [00:04<00:42, 8580.67it/s] 10%|▉         | 39500/400000 [00:04<00:41, 8644.52it/s] 10%|█         | 40365/400000 [00:04<00:41, 8622.43it/s] 10%|█         | 41235/400000 [00:05<00:41, 8645.44it/s] 11%|█         | 42108/400000 [00:05<00:41, 8667.89it/s] 11%|█         | 42975/400000 [00:05<00:41, 8551.43it/s] 11%|█         | 43832/400000 [00:05<00:41, 8553.87it/s] 11%|█         | 44688/400000 [00:05<00:41, 8463.08it/s] 11%|█▏        | 45553/400000 [00:05<00:41, 8515.72it/s] 12%|█▏        | 46416/400000 [00:05<00:41, 8548.68it/s] 12%|█▏        | 47272/400000 [00:05<00:41, 8457.76it/s] 12%|█▏        | 48153/400000 [00:05<00:41, 8559.55it/s] 12%|█▏        | 49021/400000 [00:05<00:40, 8593.01it/s] 12%|█▏        | 49881/400000 [00:06<00:40, 8545.35it/s] 13%|█▎        | 50741/400000 [00:06<00:40, 8560.52it/s] 13%|█▎        | 51598/400000 [00:06<00:40, 8561.87it/s] 13%|█▎        | 52474/400000 [00:06<00:40, 8617.94it/s] 13%|█▎        | 53372/400000 [00:06<00:39, 8723.08it/s] 14%|█▎        | 54245/400000 [00:06<00:39, 8669.13it/s] 14%|█▍        | 55125/400000 [00:06<00:39, 8706.44it/s] 14%|█▍        | 55996/400000 [00:06<00:39, 8629.15it/s] 14%|█▍        | 56876/400000 [00:06<00:39, 8677.35it/s] 14%|█▍        | 57745/400000 [00:06<00:39, 8652.16it/s] 15%|█▍        | 58611/400000 [00:07<00:39, 8536.59it/s] 15%|█▍        | 59476/400000 [00:07<00:39, 8567.65it/s] 15%|█▌        | 60334/400000 [00:07<00:39, 8518.08it/s] 15%|█▌        | 61191/400000 [00:07<00:39, 8533.27it/s] 16%|█▌        | 62073/400000 [00:07<00:39, 8615.81it/s] 16%|█▌        | 62935/400000 [00:07<00:39, 8557.95it/s] 16%|█▌        | 63814/400000 [00:07<00:38, 8624.38it/s] 16%|█▌        | 64677/400000 [00:07<00:39, 8577.50it/s] 16%|█▋        | 65557/400000 [00:07<00:38, 8640.49it/s] 17%|█▋        | 66424/400000 [00:07<00:38, 8646.89it/s] 17%|█▋        | 67289/400000 [00:08<00:38, 8568.83it/s] 17%|█▋        | 68149/400000 [00:08<00:38, 8577.44it/s] 17%|█▋        | 69007/400000 [00:08<00:39, 8481.43it/s] 17%|█▋        | 69883/400000 [00:08<00:38, 8562.37it/s] 18%|█▊        | 70766/400000 [00:08<00:38, 8640.38it/s] 18%|█▊        | 71631/400000 [00:08<00:38, 8569.97it/s] 18%|█▊        | 72489/400000 [00:08<00:39, 8382.46it/s] 18%|█▊        | 73329/400000 [00:08<00:39, 8355.53it/s] 19%|█▊        | 74211/400000 [00:08<00:38, 8488.49it/s] 19%|█▉        | 75079/400000 [00:08<00:38, 8544.45it/s] 19%|█▉        | 75935/400000 [00:09<00:37, 8544.22it/s] 19%|█▉        | 76810/400000 [00:09<00:37, 8603.75it/s] 19%|█▉        | 77671/400000 [00:09<00:37, 8500.08it/s] 20%|█▉        | 78554/400000 [00:09<00:37, 8595.44it/s] 20%|█▉        | 79422/400000 [00:09<00:37, 8619.07it/s] 20%|██        | 80285/400000 [00:09<00:37, 8525.50it/s] 20%|██        | 81142/400000 [00:09<00:37, 8536.59it/s] 20%|██        | 81997/400000 [00:09<00:37, 8447.48it/s] 21%|██        | 82870/400000 [00:09<00:37, 8529.98it/s] 21%|██        | 83731/400000 [00:09<00:36, 8552.12it/s] 21%|██        | 84587/400000 [00:10<00:37, 8412.28it/s] 21%|██▏       | 85438/400000 [00:10<00:37, 8439.65it/s] 22%|██▏       | 86283/400000 [00:10<00:37, 8307.58it/s] 22%|██▏       | 87133/400000 [00:10<00:37, 8362.67it/s] 22%|██▏       | 88011/400000 [00:10<00:36, 8481.14it/s] 22%|██▏       | 88870/400000 [00:10<00:36, 8511.40it/s] 22%|██▏       | 89766/400000 [00:10<00:35, 8639.58it/s] 23%|██▎       | 90631/400000 [00:10<00:36, 8554.02it/s] 23%|██▎       | 91490/400000 [00:10<00:36, 8561.39it/s] 23%|██▎       | 92368/400000 [00:10<00:35, 8624.35it/s] 23%|██▎       | 93231/400000 [00:11<00:35, 8564.69it/s] 24%|██▎       | 94088/400000 [00:11<00:36, 8426.23it/s] 24%|██▎       | 94932/400000 [00:11<00:37, 8209.59it/s] 24%|██▍       | 95816/400000 [00:11<00:36, 8387.34it/s] 24%|██▍       | 96663/400000 [00:11<00:36, 8411.83it/s] 24%|██▍       | 97506/400000 [00:11<00:36, 8333.71it/s] 25%|██▍       | 98341/400000 [00:11<00:36, 8283.87it/s] 25%|██▍       | 99175/400000 [00:11<00:36, 8296.91it/s] 25%|██▌       | 100006/400000 [00:11<00:36, 8251.20it/s] 25%|██▌       | 100849/400000 [00:12<00:36, 8301.77it/s] 25%|██▌       | 101680/400000 [00:12<00:36, 8250.54it/s] 26%|██▌       | 102538/400000 [00:12<00:35, 8344.28it/s] 26%|██▌       | 103379/400000 [00:12<00:35, 8354.25it/s] 26%|██▌       | 104236/400000 [00:12<00:35, 8416.80it/s] 26%|██▋       | 105099/400000 [00:12<00:34, 8479.14it/s] 26%|██▋       | 105951/400000 [00:12<00:34, 8489.66it/s] 27%|██▋       | 106808/400000 [00:12<00:34, 8512.04it/s] 27%|██▋       | 107674/400000 [00:12<00:34, 8555.22it/s] 27%|██▋       | 108530/400000 [00:12<00:34, 8382.03it/s] 27%|██▋       | 109397/400000 [00:13<00:34, 8464.85it/s] 28%|██▊       | 110245/400000 [00:13<00:34, 8463.00it/s] 28%|██▊       | 111107/400000 [00:13<00:33, 8509.07it/s] 28%|██▊       | 111973/400000 [00:13<00:33, 8552.32it/s] 28%|██▊       | 112833/400000 [00:13<00:33, 8564.10it/s] 28%|██▊       | 113692/400000 [00:13<00:33, 8570.28it/s] 29%|██▊       | 114550/400000 [00:13<00:34, 8321.22it/s] 29%|██▉       | 115384/400000 [00:13<00:34, 8298.53it/s] 29%|██▉       | 116262/400000 [00:13<00:33, 8436.22it/s] 29%|██▉       | 117113/400000 [00:13<00:33, 8457.80it/s] 29%|██▉       | 117997/400000 [00:14<00:32, 8565.99it/s] 30%|██▉       | 118855/400000 [00:14<00:33, 8487.77it/s] 30%|██▉       | 119731/400000 [00:14<00:32, 8567.13it/s] 30%|███       | 120589/400000 [00:14<00:32, 8552.48it/s] 30%|███       | 121445/400000 [00:14<00:32, 8521.59it/s] 31%|███       | 122331/400000 [00:14<00:32, 8618.59it/s] 31%|███       | 123194/400000 [00:14<00:32, 8587.72it/s] 31%|███       | 124084/400000 [00:14<00:31, 8677.70it/s] 31%|███       | 124953/400000 [00:14<00:32, 8564.05it/s] 31%|███▏      | 125817/400000 [00:14<00:31, 8585.68it/s] 32%|███▏      | 126696/400000 [00:15<00:31, 8643.39it/s] 32%|███▏      | 127561/400000 [00:15<00:32, 8426.54it/s] 32%|███▏      | 128441/400000 [00:15<00:31, 8535.15it/s] 32%|███▏      | 129302/400000 [00:15<00:31, 8557.05it/s] 33%|███▎      | 130159/400000 [00:15<00:31, 8488.49it/s] 33%|███▎      | 131027/400000 [00:15<00:31, 8542.93it/s] 33%|███▎      | 131882/400000 [00:15<00:31, 8456.65it/s] 33%|███▎      | 132762/400000 [00:15<00:31, 8556.01it/s] 33%|███▎      | 133628/400000 [00:15<00:31, 8585.85it/s] 34%|███▎      | 134488/400000 [00:15<00:30, 8571.48it/s] 34%|███▍      | 135361/400000 [00:16<00:30, 8618.13it/s] 34%|███▍      | 136224/400000 [00:16<00:31, 8465.65it/s] 34%|███▍      | 137102/400000 [00:16<00:30, 8552.53it/s] 34%|███▍      | 137959/400000 [00:16<00:32, 8176.66it/s] 35%|███▍      | 138788/400000 [00:16<00:31, 8208.61it/s] 35%|███▍      | 139666/400000 [00:16<00:31, 8370.92it/s] 35%|███▌      | 140519/400000 [00:16<00:30, 8415.97it/s] 35%|███▌      | 141404/400000 [00:16<00:30, 8537.81it/s] 36%|███▌      | 142260/400000 [00:16<00:30, 8348.08it/s] 36%|███▌      | 143119/400000 [00:17<00:30, 8418.35it/s] 36%|███▌      | 143995/400000 [00:17<00:30, 8517.37it/s] 36%|███▌      | 144849/400000 [00:17<00:30, 8459.48it/s] 36%|███▋      | 145730/400000 [00:17<00:29, 8560.84it/s] 37%|███▋      | 146588/400000 [00:17<00:29, 8531.61it/s] 37%|███▋      | 147467/400000 [00:17<00:29, 8606.63it/s] 37%|███▋      | 148336/400000 [00:17<00:29, 8629.57it/s] 37%|███▋      | 149200/400000 [00:17<00:29, 8545.48it/s] 38%|███▊      | 150083/400000 [00:17<00:28, 8628.63it/s] 38%|███▊      | 150947/400000 [00:17<00:28, 8604.30it/s] 38%|███▊      | 151822/400000 [00:18<00:28, 8647.05it/s] 38%|███▊      | 152696/400000 [00:18<00:28, 8671.41it/s] 38%|███▊      | 153564/400000 [00:18<00:28, 8574.10it/s] 39%|███▊      | 154441/400000 [00:18<00:28, 8630.57it/s] 39%|███▉      | 155305/400000 [00:18<00:28, 8580.33it/s] 39%|███▉      | 156164/400000 [00:18<00:28, 8431.57it/s] 39%|███▉      | 157019/400000 [00:18<00:28, 8465.84it/s] 39%|███▉      | 157867/400000 [00:18<00:28, 8468.95it/s] 40%|███▉      | 158733/400000 [00:18<00:28, 8524.54it/s] 40%|███▉      | 159586/400000 [00:18<00:28, 8455.61it/s] 40%|████      | 160470/400000 [00:19<00:27, 8567.32it/s] 40%|████      | 161328/400000 [00:19<00:27, 8558.68it/s] 41%|████      | 162185/400000 [00:19<00:27, 8533.93it/s] 41%|████      | 163062/400000 [00:19<00:27, 8601.35it/s] 41%|████      | 163924/400000 [00:19<00:27, 8604.65it/s] 41%|████      | 164810/400000 [00:19<00:27, 8678.51it/s] 41%|████▏     | 165680/400000 [00:19<00:26, 8682.27it/s] 42%|████▏     | 166549/400000 [00:19<00:27, 8643.32it/s] 42%|████▏     | 167414/400000 [00:19<00:26, 8627.96it/s] 42%|████▏     | 168277/400000 [00:19<00:26, 8589.72it/s] 42%|████▏     | 169162/400000 [00:20<00:26, 8663.84it/s] 43%|████▎     | 170053/400000 [00:20<00:26, 8735.23it/s] 43%|████▎     | 170927/400000 [00:20<00:26, 8664.25it/s] 43%|████▎     | 171794/400000 [00:20<00:26, 8619.19it/s] 43%|████▎     | 172657/400000 [00:20<00:26, 8515.75it/s] 43%|████▎     | 173544/400000 [00:20<00:26, 8618.06it/s] 44%|████▎     | 174435/400000 [00:20<00:25, 8702.87it/s] 44%|████▍     | 175330/400000 [00:20<00:25, 8772.86it/s] 44%|████▍     | 176208/400000 [00:20<00:25, 8714.88it/s] 44%|████▍     | 177080/400000 [00:20<00:25, 8624.43it/s] 44%|████▍     | 177951/400000 [00:21<00:25, 8649.35it/s] 45%|████▍     | 178836/400000 [00:21<00:25, 8705.97it/s] 45%|████▍     | 179717/400000 [00:21<00:25, 8734.45it/s] 45%|████▌     | 180591/400000 [00:21<00:25, 8647.41it/s] 45%|████▌     | 181457/400000 [00:21<00:25, 8487.21it/s] 46%|████▌     | 182307/400000 [00:21<00:25, 8469.89it/s] 46%|████▌     | 183155/400000 [00:21<00:25, 8470.68it/s] 46%|████▌     | 184036/400000 [00:21<00:25, 8567.70it/s] 46%|████▌     | 184894/400000 [00:21<00:25, 8494.06it/s] 46%|████▋     | 185744/400000 [00:21<00:25, 8326.26it/s] 47%|████▋     | 186629/400000 [00:22<00:25, 8475.28it/s] 47%|████▋     | 187518/400000 [00:22<00:24, 8594.76it/s] 47%|████▋     | 188391/400000 [00:22<00:24, 8632.54it/s] 47%|████▋     | 189256/400000 [00:22<00:24, 8492.78it/s] 48%|████▊     | 190107/400000 [00:22<00:24, 8492.83it/s] 48%|████▊     | 190975/400000 [00:22<00:24, 8545.49it/s] 48%|████▊     | 191852/400000 [00:22<00:24, 8610.45it/s] 48%|████▊     | 192714/400000 [00:22<00:24, 8506.97it/s] 48%|████▊     | 193566/400000 [00:22<00:24, 8449.40it/s] 49%|████▊     | 194431/400000 [00:22<00:24, 8507.26it/s] 49%|████▉     | 195322/400000 [00:23<00:23, 8622.05it/s] 49%|████▉     | 196210/400000 [00:23<00:23, 8697.25it/s] 49%|████▉     | 197084/400000 [00:23<00:23, 8706.65it/s] 49%|████▉     | 197956/400000 [00:23<00:23, 8614.84it/s] 50%|████▉     | 198819/400000 [00:23<00:23, 8531.26it/s] 50%|████▉     | 199673/400000 [00:23<00:23, 8527.26it/s] 50%|█████     | 200552/400000 [00:23<00:23, 8603.51it/s] 50%|█████     | 201436/400000 [00:23<00:22, 8670.96it/s] 51%|█████     | 202304/400000 [00:23<00:23, 8477.27it/s] 51%|█████     | 203182/400000 [00:23<00:22, 8564.49it/s] 51%|█████     | 204045/400000 [00:24<00:22, 8582.56it/s] 51%|█████     | 204917/400000 [00:24<00:22, 8620.75it/s] 51%|█████▏    | 205809/400000 [00:24<00:22, 8708.33it/s] 52%|█████▏    | 206681/400000 [00:24<00:22, 8462.00it/s] 52%|█████▏    | 207530/400000 [00:24<00:23, 8331.59it/s] 52%|█████▏    | 208410/400000 [00:24<00:22, 8464.64it/s] 52%|█████▏    | 209299/400000 [00:24<00:22, 8585.21it/s] 53%|█████▎    | 210179/400000 [00:24<00:21, 8646.42it/s] 53%|█████▎    | 211045/400000 [00:24<00:22, 8513.47it/s] 53%|█████▎    | 211940/400000 [00:25<00:21, 8638.19it/s] 53%|█████▎    | 212817/400000 [00:25<00:21, 8676.48it/s] 53%|█████▎    | 213686/400000 [00:25<00:21, 8624.30it/s] 54%|█████▎    | 214550/400000 [00:25<00:21, 8619.78it/s] 54%|█████▍    | 215413/400000 [00:25<00:21, 8441.00it/s] 54%|█████▍    | 216297/400000 [00:25<00:21, 8554.37it/s] 54%|█████▍    | 217175/400000 [00:25<00:21, 8619.71it/s] 55%|█████▍    | 218052/400000 [00:25<00:21, 8661.38it/s] 55%|█████▍    | 218932/400000 [00:25<00:20, 8699.89it/s] 55%|█████▍    | 219803/400000 [00:25<00:21, 8558.14it/s] 55%|█████▌    | 220687/400000 [00:26<00:20, 8637.65it/s] 55%|█████▌    | 221552/400000 [00:26<00:20, 8586.51it/s] 56%|█████▌    | 222434/400000 [00:26<00:20, 8653.07it/s] 56%|█████▌    | 223316/400000 [00:26<00:20, 8700.52it/s] 56%|█████▌    | 224187/400000 [00:26<00:20, 8604.32it/s] 56%|█████▋    | 225087/400000 [00:26<00:20, 8716.92it/s] 56%|█████▋    | 225960/400000 [00:26<00:19, 8720.44it/s] 57%|█████▋    | 226833/400000 [00:26<00:19, 8676.90it/s] 57%|█████▋    | 227702/400000 [00:26<00:19, 8615.15it/s] 57%|█████▋    | 228564/400000 [00:26<00:20, 8370.87it/s] 57%|█████▋    | 229403/400000 [00:27<00:20, 8367.12it/s] 58%|█████▊    | 230269/400000 [00:27<00:20, 8400.08it/s] 58%|█████▊    | 231124/400000 [00:27<00:20, 8443.73it/s] 58%|█████▊    | 231976/400000 [00:27<00:19, 8466.00it/s] 58%|█████▊    | 232824/400000 [00:27<00:19, 8400.03it/s] 58%|█████▊    | 233716/400000 [00:27<00:19, 8547.62it/s] 59%|█████▊    | 234602/400000 [00:27<00:19, 8638.60it/s] 59%|█████▉    | 235494/400000 [00:27<00:18, 8720.13it/s] 59%|█████▉    | 236367/400000 [00:27<00:18, 8616.87it/s] 59%|█████▉    | 237230/400000 [00:27<00:19, 8490.21it/s] 60%|█████▉    | 238115/400000 [00:28<00:18, 8593.00it/s] 60%|█████▉    | 239006/400000 [00:28<00:18, 8683.96it/s] 60%|█████▉    | 239876/400000 [00:28<00:18, 8667.26it/s] 60%|██████    | 240744/400000 [00:28<00:18, 8414.49it/s] 60%|██████    | 241588/400000 [00:28<00:19, 8326.11it/s] 61%|██████    | 242425/400000 [00:28<00:18, 8339.09it/s] 61%|██████    | 243301/400000 [00:28<00:18, 8458.70it/s] 61%|██████    | 244178/400000 [00:28<00:18, 8547.32it/s] 61%|██████▏   | 245034/400000 [00:28<00:18, 8486.39it/s] 61%|██████▏   | 245892/400000 [00:28<00:18, 8513.31it/s] 62%|██████▏   | 246777/400000 [00:29<00:17, 8611.24it/s] 62%|██████▏   | 247644/400000 [00:29<00:17, 8628.01it/s] 62%|██████▏   | 248531/400000 [00:29<00:17, 8698.47it/s] 62%|██████▏   | 249402/400000 [00:29<00:17, 8672.14it/s] 63%|██████▎   | 250270/400000 [00:29<00:17, 8650.94it/s] 63%|██████▎   | 251158/400000 [00:29<00:17, 8715.56it/s] 63%|██████▎   | 252043/400000 [00:29<00:16, 8755.03it/s] 63%|██████▎   | 252935/400000 [00:29<00:16, 8801.16it/s] 63%|██████▎   | 253816/400000 [00:29<00:16, 8778.56it/s] 64%|██████▎   | 254695/400000 [00:29<00:16, 8680.55it/s] 64%|██████▍   | 255564/400000 [00:30<00:16, 8637.22it/s] 64%|██████▍   | 256429/400000 [00:30<00:16, 8605.84it/s] 64%|██████▍   | 257293/400000 [00:30<00:16, 8614.16it/s] 65%|██████▍   | 258155/400000 [00:30<00:16, 8529.42it/s] 65%|██████▍   | 259009/400000 [00:30<00:16, 8498.52it/s] 65%|██████▍   | 259888/400000 [00:30<00:16, 8581.75it/s] 65%|██████▌   | 260770/400000 [00:30<00:16, 8649.45it/s] 65%|██████▌   | 261651/400000 [00:30<00:15, 8696.82it/s] 66%|██████▌   | 262522/400000 [00:30<00:16, 8554.38it/s] 66%|██████▌   | 263379/400000 [00:31<00:15, 8552.69it/s] 66%|██████▌   | 264253/400000 [00:31<00:15, 8606.74it/s] 66%|██████▋   | 265115/400000 [00:31<00:15, 8593.23it/s] 66%|██████▋   | 265993/400000 [00:31<00:15, 8646.12it/s] 67%|██████▋   | 266858/400000 [00:31<00:15, 8511.73it/s] 67%|██████▋   | 267715/400000 [00:31<00:15, 8527.67it/s] 67%|██████▋   | 268579/400000 [00:31<00:15, 8558.64it/s] 67%|██████▋   | 269468/400000 [00:31<00:15, 8653.09it/s] 68%|██████▊   | 270334/400000 [00:31<00:15, 8585.12it/s] 68%|██████▊   | 271193/400000 [00:31<00:15, 8409.64it/s] 68%|██████▊   | 272045/400000 [00:32<00:15, 8440.42it/s] 68%|██████▊   | 272932/400000 [00:32<00:14, 8562.32it/s] 68%|██████▊   | 273790/400000 [00:32<00:14, 8551.13it/s] 69%|██████▊   | 274673/400000 [00:32<00:14, 8630.35it/s] 69%|██████▉   | 275537/400000 [00:32<00:14, 8472.83it/s] 69%|██████▉   | 276386/400000 [00:32<00:14, 8467.42it/s] 69%|██████▉   | 277279/400000 [00:32<00:14, 8598.40it/s] 70%|██████▉   | 278140/400000 [00:32<00:14, 8396.60it/s] 70%|██████▉   | 279018/400000 [00:32<00:14, 8507.06it/s] 70%|██████▉   | 279871/400000 [00:32<00:14, 8445.43it/s] 70%|███████   | 280728/400000 [00:33<00:14, 8480.70it/s] 70%|███████   | 281603/400000 [00:33<00:13, 8557.68it/s] 71%|███████   | 282485/400000 [00:33<00:13, 8632.82it/s] 71%|███████   | 283349/400000 [00:33<00:13, 8601.82it/s] 71%|███████   | 284210/400000 [00:33<00:13, 8530.04it/s] 71%|███████▏  | 285064/400000 [00:33<00:13, 8435.35it/s] 71%|███████▏  | 285940/400000 [00:33<00:13, 8529.98it/s] 72%|███████▏  | 286820/400000 [00:33<00:13, 8607.27it/s] 72%|███████▏  | 287683/400000 [00:33<00:13, 8612.33it/s] 72%|███████▏  | 288545/400000 [00:33<00:13, 8547.34it/s] 72%|███████▏  | 289401/400000 [00:34<00:12, 8523.14it/s] 73%|███████▎  | 290268/400000 [00:34<00:12, 8565.46it/s] 73%|███████▎  | 291149/400000 [00:34<00:12, 8634.91it/s] 73%|███████▎  | 292013/400000 [00:34<00:12, 8583.61it/s] 73%|███████▎  | 292894/400000 [00:34<00:12, 8649.39it/s] 73%|███████▎  | 293760/400000 [00:34<00:12, 8649.77it/s] 74%|███████▎  | 294651/400000 [00:34<00:12, 8723.28it/s] 74%|███████▍  | 295549/400000 [00:34<00:11, 8796.43it/s] 74%|███████▍  | 296430/400000 [00:34<00:11, 8766.71it/s] 74%|███████▍  | 297320/400000 [00:34<00:11, 8805.76it/s] 75%|███████▍  | 298201/400000 [00:35<00:11, 8734.07it/s] 75%|███████▍  | 299075/400000 [00:35<00:11, 8615.20it/s] 75%|███████▍  | 299947/400000 [00:35<00:11, 8646.10it/s] 75%|███████▌  | 300813/400000 [00:35<00:11, 8590.50it/s] 75%|███████▌  | 301697/400000 [00:35<00:11, 8662.47it/s] 76%|███████▌  | 302564/400000 [00:35<00:11, 8655.81it/s] 76%|███████▌  | 303453/400000 [00:35<00:11, 8722.09it/s] 76%|███████▌  | 304340/400000 [00:35<00:10, 8764.59it/s] 76%|███████▋  | 305217/400000 [00:35<00:10, 8741.84it/s] 77%|███████▋  | 306110/400000 [00:35<00:10, 8795.78it/s] 77%|███████▋  | 306990/400000 [00:36<00:10, 8676.56it/s] 77%|███████▋  | 307860/400000 [00:36<00:10, 8683.41it/s] 77%|███████▋  | 308736/400000 [00:36<00:10, 8705.79it/s] 77%|███████▋  | 309607/400000 [00:36<00:10, 8596.34it/s] 78%|███████▊  | 310472/400000 [00:36<00:10, 8611.67it/s] 78%|███████▊  | 311334/400000 [00:36<00:10, 8571.06it/s] 78%|███████▊  | 312210/400000 [00:36<00:10, 8625.44it/s] 78%|███████▊  | 313073/400000 [00:36<00:10, 8532.03it/s] 78%|███████▊  | 313927/400000 [00:36<00:10, 8437.75it/s] 79%|███████▊  | 314772/400000 [00:36<00:10, 8382.56it/s] 79%|███████▉  | 315611/400000 [00:37<00:10, 8263.77it/s] 79%|███████▉  | 316498/400000 [00:37<00:09, 8435.06it/s] 79%|███████▉  | 317352/400000 [00:37<00:09, 8465.46it/s] 80%|███████▉  | 318205/400000 [00:37<00:09, 8484.04it/s] 80%|███████▉  | 319089/400000 [00:37<00:09, 8587.04it/s] 80%|███████▉  | 319952/400000 [00:37<00:09, 8596.98it/s] 80%|████████  | 320825/400000 [00:37<00:09, 8634.60it/s] 80%|████████  | 321689/400000 [00:37<00:09, 8530.98it/s] 81%|████████  | 322577/400000 [00:37<00:08, 8632.76it/s] 81%|████████  | 323470/400000 [00:37<00:08, 8719.23it/s] 81%|████████  | 324343/400000 [00:38<00:08, 8535.40it/s] 81%|████████▏ | 325235/400000 [00:38<00:08, 8646.92it/s] 82%|████████▏ | 326101/400000 [00:38<00:08, 8646.31it/s] 82%|████████▏ | 326967/400000 [00:38<00:08, 8511.82it/s] 82%|████████▏ | 327820/400000 [00:38<00:08, 8516.95it/s] 82%|████████▏ | 328675/400000 [00:38<00:08, 8524.44it/s] 82%|████████▏ | 329574/400000 [00:38<00:08, 8656.69it/s] 83%|████████▎ | 330441/400000 [00:38<00:08, 8630.07it/s] 83%|████████▎ | 331338/400000 [00:38<00:07, 8728.63it/s] 83%|████████▎ | 332230/400000 [00:39<00:07, 8782.33it/s] 83%|████████▎ | 333109/400000 [00:39<00:07, 8670.35it/s] 83%|████████▎ | 333983/400000 [00:39<00:07, 8689.54it/s] 84%|████████▎ | 334853/400000 [00:39<00:07, 8663.00it/s] 84%|████████▍ | 335733/400000 [00:39<00:07, 8702.61it/s] 84%|████████▍ | 336604/400000 [00:39<00:07, 8686.26it/s] 84%|████████▍ | 337473/400000 [00:39<00:07, 8614.23it/s] 85%|████████▍ | 338361/400000 [00:39<00:07, 8690.32it/s] 85%|████████▍ | 339231/400000 [00:39<00:06, 8683.47it/s] 85%|████████▌ | 340125/400000 [00:39<00:06, 8757.78it/s] 85%|████████▌ | 341019/400000 [00:40<00:06, 8809.39it/s] 85%|████████▌ | 341901/400000 [00:40<00:06, 8540.16it/s] 86%|████████▌ | 342767/400000 [00:40<00:06, 8573.37it/s] 86%|████████▌ | 343626/400000 [00:40<00:06, 8494.73it/s] 86%|████████▌ | 344508/400000 [00:40<00:06, 8588.54it/s] 86%|████████▋ | 345396/400000 [00:40<00:06, 8672.70it/s] 87%|████████▋ | 346265/400000 [00:40<00:06, 8639.95it/s] 87%|████████▋ | 347158/400000 [00:40<00:06, 8724.18it/s] 87%|████████▋ | 348032/400000 [00:40<00:05, 8686.52it/s] 87%|████████▋ | 348918/400000 [00:40<00:05, 8737.20it/s] 87%|████████▋ | 349805/400000 [00:41<00:05, 8776.24it/s] 88%|████████▊ | 350683/400000 [00:41<00:05, 8613.08it/s] 88%|████████▊ | 351546/400000 [00:41<00:05, 8534.41it/s] 88%|████████▊ | 352416/400000 [00:41<00:05, 8581.75it/s] 88%|████████▊ | 353300/400000 [00:41<00:05, 8654.11it/s] 89%|████████▊ | 354166/400000 [00:41<00:05, 8560.91it/s] 89%|████████▉ | 355023/400000 [00:41<00:05, 8427.43it/s] 89%|████████▉ | 355867/400000 [00:41<00:05, 8360.43it/s] 89%|████████▉ | 356704/400000 [00:41<00:05, 8291.93it/s] 89%|████████▉ | 357590/400000 [00:41<00:05, 8453.61it/s] 90%|████████▉ | 358437/400000 [00:42<00:05, 8283.70it/s] 90%|████████▉ | 359267/400000 [00:42<00:05, 7830.83it/s] 90%|█████████ | 360074/400000 [00:42<00:05, 7900.99it/s] 90%|█████████ | 360939/400000 [00:42<00:04, 8111.69it/s] 90%|█████████ | 361814/400000 [00:42<00:04, 8291.73it/s] 91%|█████████ | 362704/400000 [00:42<00:04, 8463.71it/s] 91%|█████████ | 363561/400000 [00:42<00:04, 8492.86it/s] 91%|█████████ | 364413/400000 [00:42<00:04, 8363.80it/s] 91%|█████████▏| 365294/400000 [00:42<00:04, 8490.35it/s] 92%|█████████▏| 366159/400000 [00:42<00:03, 8535.74it/s] 92%|█████████▏| 367015/400000 [00:43<00:03, 8541.19it/s] 92%|█████████▏| 367871/400000 [00:43<00:03, 8491.03it/s] 92%|█████████▏| 368737/400000 [00:43<00:03, 8539.48it/s] 92%|█████████▏| 369608/400000 [00:43<00:03, 8587.06it/s] 93%|█████████▎| 370468/400000 [00:43<00:03, 8367.35it/s] 93%|█████████▎| 371331/400000 [00:43<00:03, 8443.25it/s] 93%|█████████▎| 372178/400000 [00:43<00:03, 8450.22it/s] 93%|█████████▎| 373024/400000 [00:43<00:03, 8450.32it/s] 93%|█████████▎| 373897/400000 [00:43<00:03, 8531.60it/s] 94%|█████████▎| 374793/400000 [00:44<00:02, 8653.76it/s] 94%|█████████▍| 375668/400000 [00:44<00:02, 8681.34it/s] 94%|█████████▍| 376537/400000 [00:44<00:02, 8663.31it/s] 94%|█████████▍| 377404/400000 [00:44<00:02, 8590.32it/s] 95%|█████████▍| 378297/400000 [00:44<00:02, 8688.99it/s] 95%|█████████▍| 379174/400000 [00:44<00:02, 8711.44it/s] 95%|█████████▌| 380055/400000 [00:44<00:02, 8738.31it/s] 95%|█████████▌| 380930/400000 [00:44<00:02, 8415.21it/s] 95%|█████████▌| 381775/400000 [00:44<00:02, 8363.03it/s] 96%|█████████▌| 382644/400000 [00:44<00:02, 8457.88it/s] 96%|█████████▌| 383501/400000 [00:45<00:01, 8488.52it/s] 96%|█████████▌| 384352/400000 [00:45<00:01, 8343.97it/s] 96%|█████████▋| 385202/400000 [00:45<00:01, 8387.47it/s] 97%|█████████▋| 386069/400000 [00:45<00:01, 8470.01it/s] 97%|█████████▋| 386944/400000 [00:45<00:01, 8551.14it/s] 97%|█████████▋| 387821/400000 [00:45<00:01, 8615.54it/s] 97%|█████████▋| 388704/400000 [00:45<00:01, 8677.02it/s] 97%|█████████▋| 389573/400000 [00:45<00:01, 8627.39it/s] 98%|█████████▊| 390437/400000 [00:45<00:01, 8582.56it/s] 98%|█████████▊| 391307/400000 [00:45<00:01, 8617.45it/s] 98%|█████████▊| 392176/400000 [00:46<00:00, 8637.69it/s] 98%|█████████▊| 393040/400000 [00:46<00:00, 8514.65it/s] 98%|█████████▊| 393893/400000 [00:46<00:00, 8472.97it/s] 99%|█████████▊| 394741/400000 [00:46<00:00, 8457.90it/s] 99%|█████████▉| 395604/400000 [00:46<00:00, 8507.29it/s] 99%|█████████▉| 396477/400000 [00:46<00:00, 8571.87it/s] 99%|█████████▉| 397356/400000 [00:46<00:00, 8634.15it/s]100%|█████████▉| 398220/400000 [00:46<00:00, 8462.58it/s]100%|█████████▉| 399068/400000 [00:46<00:00, 8359.75it/s]100%|█████████▉| 399955/400000 [00:46<00:00, 8505.96it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8517.17it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe784ab00f0>, <torchtext.data.dataset.TabularDataset object at 0x7fe784ab0240>, <torchtext.vocab.Vocab object at 0x7fe784ab0160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.35 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.35 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.35 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.43 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.62 file/s]2020-06-13 06:17:43.506264: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 06:17:43.510228: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-13 06:17:43.510390: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bc0eeb00e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 06:17:43.510407: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4055040/9912422 [00:00<00:00, 40484320.94it/s]9920512it [00:00, 34845853.86it/s]                             
0it [00:00, ?it/s]32768it [00:00, 540612.97it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 134894.31it/s]1654784it [00:00, 8905232.73it/s]                          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 69548.07it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
