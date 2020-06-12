
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fbde0798620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fbde0798620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fbe4bd5f0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fbe4bd5f0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fbe65a8bea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fbe65a8bea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbdf95dc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbdf95dc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbdf95dc950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 470454.71it/s] 76%|███████▋  | 7569408/9912422 [00:00<00:03, 670273.75it/s]9920512it [00:00, 43640222.73it/s]                           
0it [00:00, ?it/s]32768it [00:00, 716081.49it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1012750.39it/s]1654784it [00:00, 12596103.48it/s]                           
0it [00:00, ?it/s]8192it [00:00, 245721.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbdf6b1cb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbdf6b20d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbdf95dc598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbdf95dc598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fbdf95dc598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:41,  1.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:39,  3.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:39,  3.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:39,  3.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:29,  5.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:29,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:29,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:22,  6.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:18,  8.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:18,  8.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:18,  8.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:14,  9.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:14,  9.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:14,  9.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 11.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 11.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:12, 11.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:11, 12.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:11, 12.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:11, 12.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:01<00:09, 14.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:09, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:09, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 15.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:09, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:08, 16.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:08, 16.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:07, 17.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:07, 17.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 18.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:07, 18.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:07, 18.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:07, 18.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 19.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 19.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:06, 19.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:06, 19.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:02<00:06, 19.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:06, 19.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:06, 19.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:06, 19.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:02<00:06, 20.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:06, 20.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:05, 20.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:05, 20.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:02<00:05, 20.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:05, 20.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:05, 20.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:05, 20.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:02<00:05, 20.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:05, 20.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:05, 20.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:05, 20.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:05, 21.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:05, 21.09 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:03<00:05, 21.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:05, 21.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:05, 21.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:03<00:05, 21.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 21.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 21.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:03<00:04, 21.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:03<00:04, 21.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:03<00:04, 21.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:03<00:04, 21.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:03<00:04, 21.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:03<00:04, 21.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:03<00:04, 21.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:03<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:03<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:03<00:04, 21.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:03<00:04, 21.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:03<00:04, 21.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:03<00:04, 21.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:03<00:04, 21.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:03<00:04, 22.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:03<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:03<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:03<00:04, 22.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:04<00:04, 22.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:04<00:04, 22.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:04<00:04, 22.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:04<00:04, 22.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 22.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 22.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:04<00:03, 22.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:04<00:03, 22.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 22.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 22.98 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:04<00:03, 22.98 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:04<00:03, 22.98 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 23.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:04<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:04<00:03, 23.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 23.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 23.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:04<00:03, 23.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:04<00:03, 23.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:03, 23.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:03, 23.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:04<00:03, 23.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:04<00:03, 23.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:03, 23.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:03, 23.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:04<00:03, 23.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:04<00:03, 23.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:04<00:03, 23.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:04<00:03, 23.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:04<00:02, 23.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:04<00:02, 23.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:02, 23.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:02, 23.73 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:05<00:02, 23.73 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:05<00:02, 23.73 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 23.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 23.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:05<00:02, 23.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:05<00:02, 23.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 23.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 23.91 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:05<00:02, 23.91 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:05<00:02, 23.91 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 24.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 24.07 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:05<00:02, 24.07 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:05<00:02, 24.07 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 24.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 24.01 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:05<00:02, 24.01 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:05<00:02, 24.01 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:02, 24.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:02, 24.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:05<00:02, 24.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:05<00:02, 24.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:02, 24.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:02, 24.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:05<00:02, 24.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:05<00:02, 24.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:05<00:01, 24.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:05<00:01, 24.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:05<00:01, 24.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:05<00:01, 24.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:01, 24.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:01, 24.67 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:01, 24.67 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:06<00:01, 24.67 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:01, 24.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:01, 24.72 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:06<00:01, 24.72 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:06<00:01, 24.72 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:01, 25.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:01, 25.47 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:06<00:01, 25.47 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:06<00:01, 25.47 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:06<00:01, 25.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:06<00:01, 25.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:06<00:01, 25.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:06<00:01, 25.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 25.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 25.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:06<00:01, 25.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:06<00:01, 25.10 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:01, 25.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:01, 25.08 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:06<00:01, 25.08 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:06<00:01, 25.08 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:01, 24.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:01, 24.74 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:01, 24.74 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:06<00:00, 24.74 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:00, 24.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:00, 24.93 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:00, 24.93 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:00, 24.93 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:00, 24.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:00, 24.49 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:07<00:00, 24.49 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:07<00:00, 24.49 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:00, 24.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:00, 24.50 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 24.50 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:07<00:00, 24.50 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:00, 24.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:00, 24.47 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:00, 24.47 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:00, 24.47 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 24.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 24.57 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 24.57 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00, 24.57 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00, 24.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00, 24.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00, 24.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00, 24.67 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 24.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 24.03 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 24.03 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:07<00:00, 24.03 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:07<00:00, 24.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:07<00:00, 24.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 24.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 24.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.81s/ url]Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 24.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 24.32 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:07<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.99s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  7.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 24.32 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.99s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.99s/ file]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 16.22 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.99s/ url]
0 examples [00:00, ? examples/s]2020-06-12 12:10:12.599228: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 12:10:12.612434: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-06-12 12:10:12.612625: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5593b2a540e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 12:10:12.612642: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2 examples [00:00, 19.83 examples/s]106 examples [00:00, 28.10 examples/s]210 examples [00:00, 39.67 examples/s]308 examples [00:00, 55.70 examples/s]413 examples [00:00, 77.79 examples/s]519 examples [00:00, 107.72 examples/s]624 examples [00:00, 147.36 examples/s]729 examples [00:00, 198.56 examples/s]829 examples [00:00, 261.40 examples/s]934 examples [00:01, 337.25 examples/s]1037 examples [00:01, 422.32 examples/s]1142 examples [00:01, 514.52 examples/s]1248 examples [00:01, 607.51 examples/s]1351 examples [00:01, 686.46 examples/s]1453 examples [00:01, 755.61 examples/s]1558 examples [00:01, 824.83 examples/s]1665 examples [00:01, 885.04 examples/s]1771 examples [00:01, 929.78 examples/s]1876 examples [00:01, 962.54 examples/s]1981 examples [00:02, 984.75 examples/s]2086 examples [00:02, 998.64 examples/s]2190 examples [00:02, 990.49 examples/s]2295 examples [00:02, 1006.84 examples/s]2399 examples [00:02, 1016.48 examples/s]2503 examples [00:02, 1007.67 examples/s]2607 examples [00:02, 1017.07 examples/s]2711 examples [00:02, 1023.01 examples/s]2816 examples [00:02, 1030.41 examples/s]2922 examples [00:02, 1036.98 examples/s]3026 examples [00:03, 1027.05 examples/s]3129 examples [00:03, 1010.43 examples/s]3231 examples [00:03, 1005.83 examples/s]3335 examples [00:03, 1014.96 examples/s]3441 examples [00:03, 1026.01 examples/s]3545 examples [00:03, 1028.39 examples/s]3649 examples [00:03, 1030.28 examples/s]3753 examples [00:03, 1019.90 examples/s]3858 examples [00:03, 1027.68 examples/s]3965 examples [00:03, 1037.80 examples/s]4069 examples [00:04, 1034.69 examples/s]4173 examples [00:04, 1036.23 examples/s]4279 examples [00:04, 1040.47 examples/s]4385 examples [00:04, 1044.19 examples/s]4491 examples [00:04, 1048.14 examples/s]4596 examples [00:04, 1033.14 examples/s]4700 examples [00:04, 1031.50 examples/s]4804 examples [00:04, 1034.03 examples/s]4908 examples [00:04, 1035.30 examples/s]5013 examples [00:04, 1036.87 examples/s]5117 examples [00:05, 1031.74 examples/s]5221 examples [00:05, 1017.36 examples/s]5323 examples [00:05, 999.96 examples/s] 5424 examples [00:05, 1001.82 examples/s]5526 examples [00:05, 1006.26 examples/s]5630 examples [00:05, 1013.55 examples/s]5735 examples [00:05, 1022.73 examples/s]5841 examples [00:05, 1032.59 examples/s]5948 examples [00:05, 1041.93 examples/s]6053 examples [00:05, 1044.25 examples/s]6158 examples [00:06, 1044.87 examples/s]6263 examples [00:06, 1042.83 examples/s]6369 examples [00:06, 1047.08 examples/s]6474 examples [00:06, 1047.54 examples/s]6580 examples [00:06, 1050.94 examples/s]6686 examples [00:06, 1036.89 examples/s]6790 examples [00:06, 1030.71 examples/s]6894 examples [00:06, 1012.44 examples/s]6997 examples [00:06, 1017.58 examples/s]7103 examples [00:06, 1027.37 examples/s]7206 examples [00:07, 1027.92 examples/s]7309 examples [00:07, 1018.67 examples/s]7414 examples [00:07, 1027.45 examples/s]7518 examples [00:07, 1029.06 examples/s]7622 examples [00:07, 1032.14 examples/s]7726 examples [00:07, 1012.45 examples/s]7828 examples [00:07, 1014.61 examples/s]7934 examples [00:07, 1027.28 examples/s]8039 examples [00:07, 1032.42 examples/s]8145 examples [00:08, 1039.06 examples/s]8249 examples [00:08, 1034.33 examples/s]8353 examples [00:08, 1032.74 examples/s]8457 examples [00:08, 988.10 examples/s] 8557 examples [00:08, 974.50 examples/s]8662 examples [00:08, 995.50 examples/s]8766 examples [00:08, 1005.89 examples/s]8867 examples [00:08, 977.41 examples/s] 8966 examples [00:08, 971.22 examples/s]9069 examples [00:08, 987.21 examples/s]9168 examples [00:09, 961.07 examples/s]9269 examples [00:09, 974.09 examples/s]9369 examples [00:09, 980.29 examples/s]9473 examples [00:09, 997.41 examples/s]9578 examples [00:09, 1012.49 examples/s]9684 examples [00:09, 1024.32 examples/s]9788 examples [00:09, 1028.59 examples/s]9894 examples [00:09, 1037.55 examples/s]9999 examples [00:09, 1040.02 examples/s]10104 examples [00:09, 984.63 examples/s]10204 examples [00:10, 971.47 examples/s]10305 examples [00:10, 980.23 examples/s]10404 examples [00:10, 982.53 examples/s]10508 examples [00:10, 998.06 examples/s]10613 examples [00:10, 1012.37 examples/s]10718 examples [00:10, 1021.13 examples/s]10821 examples [00:10, 978.38 examples/s] 10920 examples [00:10, 961.07 examples/s]11021 examples [00:10, 974.69 examples/s]11127 examples [00:11, 996.40 examples/s]11233 examples [00:11, 1012.12 examples/s]11337 examples [00:11, 1019.06 examples/s]11440 examples [00:11, 970.04 examples/s] 11544 examples [00:11, 989.76 examples/s]11649 examples [00:11, 1005.88 examples/s]11755 examples [00:11, 1020.71 examples/s]11859 examples [00:11, 1025.39 examples/s]11963 examples [00:11, 1028.47 examples/s]12067 examples [00:11, 1030.56 examples/s]12171 examples [00:12, 1026.42 examples/s]12275 examples [00:12, 1029.98 examples/s]12379 examples [00:12, 1013.21 examples/s]12485 examples [00:12, 1024.26 examples/s]12590 examples [00:12, 1031.07 examples/s]12694 examples [00:12, 1029.65 examples/s]12798 examples [00:12, 1030.34 examples/s]12902 examples [00:12, 1028.38 examples/s]13006 examples [00:12, 1031.26 examples/s]13110 examples [00:12, 1028.78 examples/s]13214 examples [00:13, 1031.77 examples/s]13318 examples [00:13, 1029.13 examples/s]13421 examples [00:13, 996.03 examples/s] 13524 examples [00:13, 998.12 examples/s]13624 examples [00:13, 991.80 examples/s]13727 examples [00:13, 1000.04 examples/s]13832 examples [00:13, 1013.81 examples/s]13934 examples [00:13, 1011.98 examples/s]14036 examples [00:13, 953.67 examples/s] 14133 examples [00:13, 955.30 examples/s]14230 examples [00:14, 954.16 examples/s]14328 examples [00:14, 960.52 examples/s]14428 examples [00:14, 971.06 examples/s]14526 examples [00:14, 964.58 examples/s]14628 examples [00:14, 979.72 examples/s]14733 examples [00:14, 997.88 examples/s]14838 examples [00:14, 1012.65 examples/s]14941 examples [00:14, 1017.75 examples/s]15043 examples [00:14, 997.80 examples/s] 15146 examples [00:15, 1004.99 examples/s]15251 examples [00:15, 1015.96 examples/s]15353 examples [00:15, 1013.39 examples/s]15455 examples [00:15, 1015.19 examples/s]15560 examples [00:15, 1023.85 examples/s]15663 examples [00:15, 1021.91 examples/s]15766 examples [00:15, 1023.79 examples/s]15870 examples [00:15, 1027.63 examples/s]15973 examples [00:15, 1025.17 examples/s]16076 examples [00:15, 1007.99 examples/s]16177 examples [00:16, 995.64 examples/s] 16277 examples [00:16, 972.58 examples/s]16381 examples [00:16, 989.16 examples/s]16481 examples [00:16, 991.75 examples/s]16584 examples [00:16, 1002.54 examples/s]16685 examples [00:16, 989.32 examples/s] 16790 examples [00:16, 1004.06 examples/s]16895 examples [00:16, 1017.29 examples/s]16997 examples [00:16, 999.41 examples/s] 17101 examples [00:16, 1010.08 examples/s]17204 examples [00:17, 1013.56 examples/s]17310 examples [00:17, 1024.38 examples/s]17414 examples [00:17, 1028.95 examples/s]17517 examples [00:17, 1027.68 examples/s]17620 examples [00:17, 997.45 examples/s] 17721 examples [00:17, 998.42 examples/s]17826 examples [00:17, 1011.93 examples/s]17931 examples [00:17, 1022.00 examples/s]18035 examples [00:17, 1026.00 examples/s]18141 examples [00:17, 1032.94 examples/s]18245 examples [00:18, 1029.39 examples/s]18350 examples [00:18, 1035.26 examples/s]18454 examples [00:18, 1005.03 examples/s]18555 examples [00:18, 1002.51 examples/s]18656 examples [00:18, 988.19 examples/s] 18756 examples [00:18, 989.27 examples/s]18856 examples [00:18, 967.37 examples/s]18960 examples [00:18, 986.54 examples/s]19062 examples [00:18, 995.22 examples/s]19167 examples [00:18, 1009.11 examples/s]19269 examples [00:19, 1003.76 examples/s]19371 examples [00:19, 1007.23 examples/s]19472 examples [00:19, 992.70 examples/s] 19574 examples [00:19, 1000.26 examples/s]19678 examples [00:19, 1009.67 examples/s]19781 examples [00:19, 1015.24 examples/s]19885 examples [00:19, 1020.78 examples/s]19989 examples [00:19, 1024.05 examples/s]20092 examples [00:19, 975.63 examples/s] 20196 examples [00:20, 991.60 examples/s]20298 examples [00:20, 999.36 examples/s]20399 examples [00:20, 986.52 examples/s]20500 examples [00:20, 991.29 examples/s]20600 examples [00:20, 982.15 examples/s]20699 examples [00:20, 956.65 examples/s]20802 examples [00:20, 975.71 examples/s]20905 examples [00:20, 989.82 examples/s]21009 examples [00:20, 1001.69 examples/s]21110 examples [00:20, 993.13 examples/s] 21210 examples [00:21, 981.32 examples/s]21309 examples [00:21, 962.90 examples/s]21410 examples [00:21, 974.44 examples/s]21512 examples [00:21, 986.60 examples/s]21611 examples [00:21, 982.47 examples/s]21714 examples [00:21, 994.98 examples/s]21817 examples [00:21, 1004.26 examples/s]21921 examples [00:21, 1014.03 examples/s]22024 examples [00:21, 1015.52 examples/s]22126 examples [00:21, 1009.98 examples/s]22228 examples [00:22, 1010.20 examples/s]22330 examples [00:22, 1004.17 examples/s]22433 examples [00:22, 1010.26 examples/s]22537 examples [00:22, 1016.38 examples/s]22640 examples [00:22, 1018.29 examples/s]22742 examples [00:22, 1008.98 examples/s]22843 examples [00:22, 989.10 examples/s] 22943 examples [00:22, 986.94 examples/s]23046 examples [00:22, 997.60 examples/s]23148 examples [00:22, 1002.03 examples/s]23251 examples [00:23, 1007.72 examples/s]23356 examples [00:23, 1017.63 examples/s]23458 examples [00:23, 1017.86 examples/s]23562 examples [00:23, 1022.34 examples/s]23665 examples [00:23, 1022.66 examples/s]23768 examples [00:23, 1014.29 examples/s]23870 examples [00:23, 1001.25 examples/s]23975 examples [00:23, 1012.77 examples/s]24079 examples [00:23, 1019.61 examples/s]24183 examples [00:23, 1022.99 examples/s]24286 examples [00:24, 1014.41 examples/s]24388 examples [00:24, 1013.48 examples/s]24490 examples [00:24, 989.84 examples/s] 24590 examples [00:24, 991.44 examples/s]24694 examples [00:24, 1004.67 examples/s]24798 examples [00:24, 1014.19 examples/s]24900 examples [00:24, 999.39 examples/s] 25002 examples [00:24, 1003.18 examples/s]25104 examples [00:24, 1006.50 examples/s]25208 examples [00:25, 1014.63 examples/s]25313 examples [00:25, 1022.17 examples/s]25417 examples [00:25, 1026.22 examples/s]25520 examples [00:25, 1023.23 examples/s]25623 examples [00:25, 1020.60 examples/s]25727 examples [00:25, 1023.66 examples/s]25832 examples [00:25, 1029.06 examples/s]25935 examples [00:25, 1025.50 examples/s]26039 examples [00:25, 1027.77 examples/s]26142 examples [00:25, 1013.14 examples/s]26244 examples [00:26, 1012.42 examples/s]26347 examples [00:26, 1015.51 examples/s]26449 examples [00:26, 1008.05 examples/s]26553 examples [00:26, 1016.41 examples/s]26655 examples [00:26, 1015.38 examples/s]26759 examples [00:26, 1019.86 examples/s]26864 examples [00:26, 1026.07 examples/s]26968 examples [00:26, 1029.91 examples/s]27072 examples [00:26, 1031.81 examples/s]27176 examples [00:26, 1026.02 examples/s]27279 examples [00:27, 1023.21 examples/s]27382 examples [00:27, 1014.15 examples/s]27486 examples [00:27, 1021.32 examples/s]27589 examples [00:27, 1012.42 examples/s]27691 examples [00:27, 987.62 examples/s] 27795 examples [00:27, 1001.01 examples/s]27896 examples [00:27, 1000.57 examples/s]28000 examples [00:27, 1011.68 examples/s]28104 examples [00:27, 1018.60 examples/s]28207 examples [00:27, 1019.50 examples/s]28310 examples [00:28, 993.14 examples/s] 28414 examples [00:28, 1003.71 examples/s]28515 examples [00:28, 1001.31 examples/s]28621 examples [00:28, 1016.38 examples/s]28723 examples [00:28, 1013.49 examples/s]28828 examples [00:28, 1021.53 examples/s]28931 examples [00:28, 1022.66 examples/s]29036 examples [00:28, 1028.46 examples/s]29139 examples [00:28, 1027.91 examples/s]29242 examples [00:28, 1027.04 examples/s]29347 examples [00:29, 1031.37 examples/s]29451 examples [00:29, 1032.73 examples/s]29555 examples [00:29, 1027.76 examples/s]29658 examples [00:29, 1024.98 examples/s]29761 examples [00:29, 1005.73 examples/s]29862 examples [00:29, 982.10 examples/s] 29964 examples [00:29, 992.58 examples/s]30064 examples [00:29, 951.28 examples/s]30162 examples [00:29, 957.63 examples/s]30263 examples [00:30, 972.75 examples/s]30365 examples [00:30, 985.30 examples/s]30467 examples [00:30, 993.12 examples/s]30570 examples [00:30, 1001.61 examples/s]30672 examples [00:30, 1005.78 examples/s]30773 examples [00:30, 978.87 examples/s] 30876 examples [00:30, 993.34 examples/s]30980 examples [00:30, 1006.80 examples/s]31083 examples [00:30, 1013.21 examples/s]31187 examples [00:30, 1018.79 examples/s]31289 examples [00:31, 1015.02 examples/s]31392 examples [00:31, 1019.13 examples/s]31494 examples [00:31, 1016.38 examples/s]31596 examples [00:31, 1008.58 examples/s]31699 examples [00:31, 1013.24 examples/s]31801 examples [00:31, 1008.65 examples/s]31902 examples [00:31, 1000.64 examples/s]32003 examples [00:31, 979.52 examples/s] 32106 examples [00:31, 993.65 examples/s]32211 examples [00:31, 1008.82 examples/s]32314 examples [00:32, 1014.22 examples/s]32419 examples [00:32, 1023.40 examples/s]32524 examples [00:32, 1029.42 examples/s]32628 examples [00:32, 1020.25 examples/s]32731 examples [00:32, 1016.79 examples/s]32833 examples [00:32, 1003.43 examples/s]32934 examples [00:32, 996.60 examples/s] 33035 examples [00:32, 999.24 examples/s]33135 examples [00:32, 996.21 examples/s]33238 examples [00:32, 1005.83 examples/s]33339 examples [00:33, 1002.11 examples/s]33443 examples [00:33, 1011.59 examples/s]33547 examples [00:33, 1018.19 examples/s]33649 examples [00:33, 1017.73 examples/s]33753 examples [00:33, 1024.00 examples/s]33856 examples [00:33, 983.56 examples/s] 33959 examples [00:33, 996.38 examples/s]34063 examples [00:33, 1007.24 examples/s]34167 examples [00:33, 1015.82 examples/s]34271 examples [00:33, 1022.81 examples/s]34374 examples [00:34, 1021.09 examples/s]34478 examples [00:34, 1025.44 examples/s]34581 examples [00:34, 1021.08 examples/s]34685 examples [00:34, 1024.36 examples/s]34788 examples [00:34, 1023.98 examples/s]34891 examples [00:34, 1018.38 examples/s]34996 examples [00:34, 1026.29 examples/s]35100 examples [00:34, 1029.03 examples/s]35204 examples [00:34, 1029.97 examples/s]35308 examples [00:34, 1031.74 examples/s]35412 examples [00:35, 1011.18 examples/s]35516 examples [00:35, 1018.65 examples/s]35618 examples [00:35, 1018.84 examples/s]35721 examples [00:35, 1021.92 examples/s]35824 examples [00:35, 1022.08 examples/s]35927 examples [00:35, 1020.68 examples/s]36030 examples [00:35, 1019.86 examples/s]36136 examples [00:35, 1029.82 examples/s]36241 examples [00:35, 1035.48 examples/s]36345 examples [00:36, 1035.79 examples/s]36449 examples [00:36, 998.20 examples/s] 36550 examples [00:36, 996.69 examples/s]36652 examples [00:36, 1002.48 examples/s]36753 examples [00:36, 1002.86 examples/s]36857 examples [00:36, 1013.31 examples/s]36959 examples [00:36, 980.70 examples/s] 37063 examples [00:36, 996.70 examples/s]37167 examples [00:36, 1006.47 examples/s]37269 examples [00:36, 1007.75 examples/s]37370 examples [00:37, 1004.97 examples/s]37472 examples [00:37, 1008.17 examples/s]37575 examples [00:37, 1014.30 examples/s]37677 examples [00:37, 1013.38 examples/s]37780 examples [00:37, 1017.01 examples/s]37884 examples [00:37, 1023.70 examples/s]37987 examples [00:37, 1010.90 examples/s]38090 examples [00:37, 1015.86 examples/s]38194 examples [00:37, 1022.48 examples/s]38297 examples [00:37, 1022.77 examples/s]38400 examples [00:38, 1023.81 examples/s]38503 examples [00:38, 1019.65 examples/s]38606 examples [00:38, 1021.68 examples/s]38709 examples [00:38, 1023.32 examples/s]38812 examples [00:38, 1022.25 examples/s]38916 examples [00:38, 1024.67 examples/s]39019 examples [00:38, 1024.52 examples/s]39123 examples [00:38, 1026.81 examples/s]39226 examples [00:38, 1027.32 examples/s]39329 examples [00:38, 1025.88 examples/s]39432 examples [00:39, 1020.14 examples/s]39535 examples [00:39, 1017.34 examples/s]39637 examples [00:39, 1009.85 examples/s]39739 examples [00:39, 1007.03 examples/s]39841 examples [00:39, 1009.04 examples/s]39942 examples [00:39, 998.05 examples/s] 40042 examples [00:39, 911.40 examples/s]40146 examples [00:39, 945.66 examples/s]40250 examples [00:39, 970.89 examples/s]40350 examples [00:40, 978.54 examples/s]40449 examples [00:40, 965.04 examples/s]40550 examples [00:40, 977.48 examples/s]40653 examples [00:40, 990.24 examples/s]40756 examples [00:40, 1001.09 examples/s]40859 examples [00:40, 1007.95 examples/s]40963 examples [00:40, 1014.67 examples/s]41066 examples [00:40, 1016.65 examples/s]41171 examples [00:40, 1024.33 examples/s]41275 examples [00:40, 1026.15 examples/s]41378 examples [00:41, 1025.95 examples/s]41481 examples [00:41, 995.50 examples/s] 41581 examples [00:41, 980.34 examples/s]41683 examples [00:41, 989.77 examples/s]41783 examples [00:41, 992.63 examples/s]41888 examples [00:41, 1007.00 examples/s]41992 examples [00:41, 1015.63 examples/s]42095 examples [00:41, 1018.46 examples/s]42199 examples [00:41, 1024.70 examples/s]42302 examples [00:41, 1020.07 examples/s]42405 examples [00:42, 1019.26 examples/s]42507 examples [00:42, 1015.95 examples/s]42609 examples [00:42, 1004.06 examples/s]42710 examples [00:42, 978.56 examples/s] 42810 examples [00:42, 983.64 examples/s]42913 examples [00:42, 997.02 examples/s]43013 examples [00:42, 987.97 examples/s]43112 examples [00:42, 955.84 examples/s]43208 examples [00:42, 957.00 examples/s]43309 examples [00:42, 969.73 examples/s]43410 examples [00:43, 978.53 examples/s]43514 examples [00:43, 994.78 examples/s]43617 examples [00:43, 1002.34 examples/s]43721 examples [00:43, 1012.41 examples/s]43824 examples [00:43, 1015.26 examples/s]43928 examples [00:43, 1019.80 examples/s]44033 examples [00:43, 1027.50 examples/s]44136 examples [00:43, 1022.85 examples/s]44239 examples [00:43, 1016.15 examples/s]44341 examples [00:43, 1001.08 examples/s]44443 examples [00:44, 1004.93 examples/s]44544 examples [00:44, 1006.44 examples/s]44645 examples [00:44, 991.74 examples/s] 44749 examples [00:44, 1004.30 examples/s]44850 examples [00:44, 999.95 examples/s] 44951 examples [00:44, 995.73 examples/s]45051 examples [00:44, 970.98 examples/s]45149 examples [00:44, 964.72 examples/s]45255 examples [00:44, 989.88 examples/s]45360 examples [00:45, 1005.95 examples/s]45464 examples [00:45, 1013.35 examples/s]45568 examples [00:45, 1018.79 examples/s]45671 examples [00:45, 1021.43 examples/s]45777 examples [00:45, 1030.34 examples/s]45881 examples [00:45, 1031.33 examples/s]45986 examples [00:45, 1036.59 examples/s]46090 examples [00:45, 1036.14 examples/s]46194 examples [00:45, 1004.80 examples/s]46298 examples [00:45, 1012.87 examples/s]46400 examples [00:46, 1012.33 examples/s]46504 examples [00:46, 1019.57 examples/s]46607 examples [00:46, 1000.38 examples/s]46708 examples [00:46, 992.93 examples/s] 46811 examples [00:46, 1001.85 examples/s]46912 examples [00:46, 1001.60 examples/s]47016 examples [00:46, 1011.28 examples/s]47119 examples [00:46, 1014.89 examples/s]47222 examples [00:46, 1018.40 examples/s]47324 examples [00:46, 1003.59 examples/s]47425 examples [00:47, 999.98 examples/s] 47527 examples [00:47, 1003.55 examples/s]47628 examples [00:47, 1003.87 examples/s]47729 examples [00:47, 1002.57 examples/s]47833 examples [00:47, 1011.25 examples/s]47936 examples [00:47, 1015.54 examples/s]48038 examples [00:47, 1009.42 examples/s]48141 examples [00:47, 1013.27 examples/s]48245 examples [00:47, 1019.77 examples/s]48349 examples [00:47, 1025.54 examples/s]48454 examples [00:48, 1030.59 examples/s]48558 examples [00:48, 1029.11 examples/s]48661 examples [00:48, 1021.87 examples/s]48764 examples [00:48, 1017.59 examples/s]48868 examples [00:48, 1022.98 examples/s]48971 examples [00:48, 1024.67 examples/s]49074 examples [00:48, 1023.04 examples/s]49177 examples [00:48, 1017.93 examples/s]49279 examples [00:48, 983.04 examples/s] 49382 examples [00:48, 996.21 examples/s]49487 examples [00:49, 1009.22 examples/s]49591 examples [00:49, 1016.10 examples/s]49695 examples [00:49, 1021.16 examples/s]49798 examples [00:49, 1021.77 examples/s]49901 examples [00:49, 1023.14 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6448/50000 [00:00<00:00, 64474.37 examples/s] 38%|███▊      | 19223/50000 [00:00<00:00, 75726.40 examples/s] 63%|██████▎   | 31631/50000 [00:00<00:00, 85751.26 examples/s] 89%|████████▉ | 44639/50000 [00:00<00:00, 95515.91 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 869.45 examples/s]193 examples [00:00, 918.28 examples/s]295 examples [00:00, 946.19 examples/s]394 examples [00:00, 956.83 examples/s]500 examples [00:00, 982.77 examples/s]599 examples [00:00, 982.37 examples/s]705 examples [00:00, 1003.89 examples/s]811 examples [00:00, 1017.62 examples/s]914 examples [00:00, 1020.22 examples/s]1019 examples [00:01, 1028.56 examples/s]1120 examples [00:01, 1016.36 examples/s]1226 examples [00:01, 1026.81 examples/s]1328 examples [00:01, 1011.80 examples/s]1429 examples [00:01, 792.31 examples/s] 1535 examples [00:01, 857.05 examples/s]1641 examples [00:01, 907.67 examples/s]1746 examples [00:01, 944.14 examples/s]1845 examples [00:01, 938.85 examples/s]1949 examples [00:02, 965.32 examples/s]2055 examples [00:02, 991.70 examples/s]2159 examples [00:02, 1005.03 examples/s]2265 examples [00:02, 1018.04 examples/s]2368 examples [00:02, 1019.75 examples/s]2471 examples [00:02, 1008.44 examples/s]2573 examples [00:02, 942.57 examples/s] 2671 examples [00:02, 953.17 examples/s]2775 examples [00:02, 976.51 examples/s]2876 examples [00:02, 985.79 examples/s]2976 examples [00:03, 979.31 examples/s]3080 examples [00:03, 996.72 examples/s]3181 examples [00:03, 997.95 examples/s]3282 examples [00:03, 999.47 examples/s]3384 examples [00:03, 1002.65 examples/s]3487 examples [00:03, 1010.05 examples/s]3593 examples [00:03, 1023.35 examples/s]3699 examples [00:03, 1032.97 examples/s]3803 examples [00:03, 1019.16 examples/s]3906 examples [00:03, 988.15 examples/s] 4007 examples [00:04, 981.90 examples/s]4112 examples [00:04, 999.74 examples/s]4217 examples [00:04, 1012.16 examples/s]4322 examples [00:04, 1021.04 examples/s]4425 examples [00:04, 1020.93 examples/s]4529 examples [00:04, 1026.51 examples/s]4632 examples [00:04, 1023.55 examples/s]4737 examples [00:04, 1030.24 examples/s]4842 examples [00:04, 1035.42 examples/s]4946 examples [00:04, 992.35 examples/s] 5048 examples [00:05, 998.18 examples/s]5153 examples [00:05, 1012.28 examples/s]5255 examples [00:05, 1008.14 examples/s]5360 examples [00:05, 1019.78 examples/s]5463 examples [00:05, 1017.72 examples/s]5568 examples [00:05, 1024.75 examples/s]5671 examples [00:05, 1003.42 examples/s]5772 examples [00:05, 954.99 examples/s] 5871 examples [00:05, 963.44 examples/s]5968 examples [00:06, 950.87 examples/s]6070 examples [00:06, 968.97 examples/s]6175 examples [00:06, 989.67 examples/s]6278 examples [00:06, 1000.59 examples/s]6383 examples [00:06, 1013.49 examples/s]6488 examples [00:06, 1021.54 examples/s]6593 examples [00:06, 1028.80 examples/s]6698 examples [00:06, 1034.66 examples/s]6803 examples [00:06, 1037.79 examples/s]6910 examples [00:06, 1044.94 examples/s]7015 examples [00:07, 1043.40 examples/s]7120 examples [00:07, 1039.13 examples/s]7225 examples [00:07, 1040.98 examples/s]7330 examples [00:07, 1031.26 examples/s]7434 examples [00:07, 1026.79 examples/s]7539 examples [00:07, 1030.82 examples/s]7643 examples [00:07, 1022.08 examples/s]7746 examples [00:07, 1010.37 examples/s]7848 examples [00:07, 1011.45 examples/s]7952 examples [00:07, 1018.71 examples/s]8054 examples [00:08, 984.80 examples/s] 8154 examples [00:08, 986.98 examples/s]8259 examples [00:08, 1002.88 examples/s]8365 examples [00:08, 1016.99 examples/s]8470 examples [00:08, 1026.64 examples/s]8573 examples [00:08, 1015.07 examples/s]8677 examples [00:08, 1019.55 examples/s]8780 examples [00:08, 1018.51 examples/s]8883 examples [00:08, 1020.76 examples/s]8986 examples [00:08, 1003.50 examples/s]9089 examples [00:09, 1009.90 examples/s]9191 examples [00:09, 997.84 examples/s] 9296 examples [00:09, 1010.75 examples/s]9402 examples [00:09, 1024.37 examples/s]9508 examples [00:09, 1033.27 examples/s]9612 examples [00:09, 1035.27 examples/s]9718 examples [00:09, 1040.44 examples/s]9823 examples [00:09, 1028.96 examples/s]9926 examples [00:09, 1014.94 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7R0T92/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7R0T92/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbdf95dc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbdf95dc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbdf95dc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbe5f867b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd9405af28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbdf95dc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbdf95dc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbdf95dc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbd82a2e2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbddf999080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fbd70f9d268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fbd70f9d268> 

  function with postional parmater data_info <function split_train_valid at 0x7fbd70f9d268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fbd70f9d378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fbd70f9d378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fbd70f9d378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c02e63079812501c8f021742bb3d083f42c99e2b63bded47bcb60168849fc12f
  Stored in directory: /tmp/pip-ephem-wheel-cache-f37us0tf/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<50:16:02, 4.76kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:25:10, 6.76kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<24:50:41, 9.64kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:23:31, 13.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<12:08:21, 19.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.49M/862M [00:02<8:26:28, 28.1kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:02<5:52:25, 40.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:02<4:05:53, 57.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.2M/862M [00:02<2:51:08, 81.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:02<1:59:26, 117kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:02<1:23:09, 166kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:03<58:07, 237kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.1M/862M [00:03<40:31, 338kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:03<28:21, 481kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:03<19:46, 685kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:03<14:31, 930kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<12:02, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:05<12:18, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<09:23, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:06<06:47, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<08:33, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<07:35, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<05:39, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:09<06:41, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<04:19, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<03:12, 4.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:11<52:14, 253kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:11<39:18, 336kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<28:03, 471kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:11<19:49, 665kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:13<18:30, 711kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<14:17, 920kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:13<10:19, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:15<10:16, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<08:31, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<06:14, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:17<07:25, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<08:15, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<06:31, 1.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<04:42, 2.75MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<14:00, 925kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<11:16, 1.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<08:15, 1.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:21<08:30, 1.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<07:25, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<05:30, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<06:36, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<06:06, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:23<04:37, 2.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<07:09, 1.78MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:25<05:44, 2.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:09, 3.05MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<14:21, 882kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<11:29, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<08:24, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<08:33, 1.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<07:25, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<05:30, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:32, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:00, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<04:30, 2.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:49, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:58, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:30, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<03:58, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<12:26, 996kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<10:07, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:25, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<07:49, 1.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<08:15, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:23, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<04:38, 2.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<08:12, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:08, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:20, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<07:17, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<04:12, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<14:11, 854kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<11:18, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<08:15, 1.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:20, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:39, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<06:38, 1.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<04:47, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<09:49, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<08:14, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:06, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:49, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:07, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<04:35, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:43, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:39, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:13, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:48, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<07:39, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:41, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:57, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:58, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<06:55, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<05:30, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<03:59, 2.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<12:24, 937kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<10:00, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<07:17, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<07:33, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<07:51, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<06:09, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:26, 2.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<13:29, 852kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<10:46, 1.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<07:48, 1.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<07:54, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<06:14, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:29, 2.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<09:19, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<07:37, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<05:34, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:04, 2.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<11:36, 972kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<10:45, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<08:06, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:47, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<10:46, 1.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<08:49, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:26, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<06:53, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:11<06:05, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<04:31, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:31, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<06:20, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<04:57, 2.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<03:38, 3.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<06:14, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<05:38, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<04:12, 2.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<05:17, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<06:15, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:00, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:38, 2.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<12:37, 861kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<10:04, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:20, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<07:25, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<07:36, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:21<05:50, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:14, 2.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<07:17, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<06:19, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<04:41, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<05:33, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<06:17, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<04:55, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<03:42, 2.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:02, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<04:39, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<03:32, 2.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:49, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<06:12, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<05:30, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<04:19, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<03:12, 3.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:49, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<05:32, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<04:14, 2.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:54, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<04:38, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:32, 2.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<05:34, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<04:24, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<03:14, 3.17MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:37<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<03:49, 2.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:51, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<04:33, 2.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<03:26, 2.95MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:35, 2.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<05:28, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:18, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<03:10, 3.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<05:12, 1.93MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<03:53, 2.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:50, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<05:31, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<04:23, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:11, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<31:55, 310kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<23:25, 422kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<16:37, 594kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<13:46, 714kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<11:55, 824kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<08:48, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:18, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:38, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:27, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:44, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:02, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<04:43, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:24, 2.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<09:51, 975kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<08:00, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:51, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:07, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:19, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<03:56, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<04:53, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:30, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<03:25, 2.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<04:23, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:10, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:01<04:05, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:02, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<04:36, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<03:14, 2.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<02:22, 3.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<38:49, 239kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<29:09, 318kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<20:47, 445kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<14:42, 628kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<12:48, 719kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<09:50, 935kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:08, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<06:56, 1.32MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<05:54, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:23, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<04:59, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<05:36, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:22, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<03:12, 2.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:21, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:16, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:25, 3.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<08:37, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<07:04, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:12, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:30, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:56, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:35, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<03:18, 2.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<06:56, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:51, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:18, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:52, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:25, 1.97MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:20, 2.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:10, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:56, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:53, 2.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:53, 2.98MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<04:18, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:59, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:00, 2.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:42, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<02:50, 2.98MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:47, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:36, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<02:44, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:41, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:28, 2.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:37, 3.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<03:28, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:36, 3.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<03:41, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:27, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<02:38, 3.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:40, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:27, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<02:37, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<03:38, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:18, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<03:21, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:35, 3.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:35, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:27, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<02:38, 3.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:17, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<02:23, 3.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<07:29, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<06:06, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:27, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<04:49, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<05:02, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:54, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:49, 2.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<15:27, 501kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<11:41, 662kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:23, 920kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<07:29, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<07:00, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<05:17, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:47, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<06:01, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:05, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:44, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<04:13, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<04:38, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:40, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<02:40, 2.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<09:13, 810kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<07:18, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:16, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<05:15, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<05:20, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:06, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<02:57, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<05:26, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<04:38, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:25, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<03:55, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:02<04:26, 1.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:32, 2.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:52, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<03:29, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<02:37, 2.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:25, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:57, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:04, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:17, 3.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:36, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:17, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:29, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:16, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:47, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<02:58, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:10, 3.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<04:15, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:44, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:47, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:09, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<02:21, 2.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<03:10, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:48, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<02:59, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:12, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:38, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:19, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<02:28, 2.71MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<01:49, 3.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<07:00, 948kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:39, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:08, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<04:16, 1.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<04:32, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:29, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:33, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<03:37, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<03:16, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<02:28, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:05, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:40, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<02:56, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:07, 3.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<06:44, 944kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<05:19, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:51, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:47, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<07:47, 807kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<06:03, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<04:22, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:08, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<07:38, 814kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<06:40, 932kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:57, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:31, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:39, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:39, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:25, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:40, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:59, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:05, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:14, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:15, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:44, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:08, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:47, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:04, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<01:33, 3.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:39, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<05:16, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<04:01, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:52, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:43, 752kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<06:02, 960kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:22, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:19, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:40, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:43, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:06, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<03:22, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<02:40, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<01:55, 2.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<18:16, 307kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<13:24, 417kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<09:29, 587kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<07:47, 711kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<06:44, 819kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<04:59, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:34, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:01, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:20, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:28, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<01:48, 3.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<05:29, 981kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:27, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<03:14, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:23, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:36, 1.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:03, 2.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:51, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:33, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<01:55, 2.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:29, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:51, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<02:13, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<01:39, 3.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:23, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:47, 2.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:18, 3.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<12:32, 402kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<09:51, 512kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<07:08, 704kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:00, 993kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<17:48, 279kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<13:02, 381kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<09:12, 538kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<06:27, 760kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<14:41, 334kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<11:20, 432kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<08:08, 601kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<05:43, 847kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<05:58, 809kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<04:44, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:25, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:26, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:38, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:54, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:10, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:45, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:03, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:24, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:41, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:07, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:32, 2.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<14:45, 309kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<10:49, 421kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:38, 593kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<06:18, 713kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<05:23, 834kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:59, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:48, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<10:17, 430kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<07:41, 574kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:28, 804kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<04:43, 922kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<04:20, 1.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:14, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:18, 1.86MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<03:55, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<03:13, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:21, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:32, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:32, 2.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:12, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:40, 2.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:12, 3.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<04:00, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<03:45, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<02:51, 1.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:02, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<05:07, 781kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<03:58, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:55, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:04, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<04:38, 849kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:41, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:40, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:40, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:41, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:03, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:29, 2.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:19, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:04, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:32, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<01:52, 2.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:08, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:40, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:16, 2.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:41, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:36, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:12, 2.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:57, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:34, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:08, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<03:43, 947kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<02:58, 1.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:09, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<02:15, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:58, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:27, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:44, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:56, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:31, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:06, 3.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<10:42, 309kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<07:50, 421kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<05:32, 592kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<04:33, 713kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<03:56, 821kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<02:54, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:04, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:24, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:02, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:29, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:53, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:28, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:03, 2.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:17, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:56, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:25, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:37, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:47, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:25, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:01, 2.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<03:22, 860kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:41, 1.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:56, 1.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:56, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:40, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:14, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:27, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:20, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<00:59, 2.75MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:27, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:08, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:49, 3.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:37, 997kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:07, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:32, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:37, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:42, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:19, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:57, 2.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:24, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:16, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<00:56, 2.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:10, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:01, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:47, 3.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:34, 4.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<03:13, 727kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<02:46, 841kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<02:03, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:27, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:37, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:23, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:00, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:11, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:18, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:44, 2.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:13, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<01:05, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:47, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:00, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:55, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:41, 2.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:04, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:50, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:36, 3.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:06, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:43, 2.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:50, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:37, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:48, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<00:46, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:34, 3.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:46, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:44, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:32, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:43, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:40, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:29, 3.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:57, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:50, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:37, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:45, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<00:53, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:42, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:29, 2.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:45, 818kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:23, 1.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:00, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:58, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:59, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:32, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:48, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:42, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:31, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:43, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:33, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:23, 2.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:46, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:40, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:29, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:21, 2.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:40, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:29, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:27, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:19, 2.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:23, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:16, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:31, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:27, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:22, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:26, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:20, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 3.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:25, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:22, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:18, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:16, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:11, 2.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:14, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:14, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:10, 3.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:11, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:10, 2.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:10, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:08, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:06, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:03, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:01, 3.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.54MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 516/400000 [00:00<01:17, 5155.92it/s]  0%|          | 1364/400000 [00:00<01:08, 5842.66it/s]  1%|          | 2181/400000 [00:00<01:02, 6388.30it/s]  1%|          | 3014/400000 [00:00<00:57, 6867.79it/s]  1%|          | 3862/400000 [00:00<00:54, 7281.44it/s]  1%|          | 4689/400000 [00:00<00:52, 7550.62it/s]  1%|▏         | 5496/400000 [00:00<00:51, 7697.27it/s]  2%|▏         | 6331/400000 [00:00<00:49, 7879.88it/s]  2%|▏         | 7174/400000 [00:00<00:48, 8036.38it/s]  2%|▏         | 7996/400000 [00:01<00:48, 8088.49it/s]  2%|▏         | 8846/400000 [00:01<00:47, 8205.69it/s]  2%|▏         | 9703/400000 [00:01<00:46, 8310.11it/s]  3%|▎         | 10544/400000 [00:01<00:46, 8337.21it/s]  3%|▎         | 11387/400000 [00:01<00:46, 8362.81it/s]  3%|▎         | 12231/400000 [00:01<00:46, 8383.18it/s]  3%|▎         | 13071/400000 [00:01<00:46, 8386.32it/s]  3%|▎         | 13923/400000 [00:01<00:45, 8424.69it/s]  4%|▎         | 14778/400000 [00:01<00:45, 8459.50it/s]  4%|▍         | 15624/400000 [00:01<00:45, 8453.92it/s]  4%|▍         | 16474/400000 [00:02<00:45, 8465.39it/s]  4%|▍         | 17321/400000 [00:02<00:45, 8374.69it/s]  5%|▍         | 18163/400000 [00:02<00:45, 8385.65it/s]  5%|▍         | 19024/400000 [00:02<00:45, 8449.71it/s]  5%|▍         | 19870/400000 [00:02<00:44, 8452.68it/s]  5%|▌         | 20726/400000 [00:02<00:44, 8482.33it/s]  5%|▌         | 21575/400000 [00:02<00:44, 8464.28it/s]  6%|▌         | 22428/400000 [00:02<00:44, 8482.30it/s]  6%|▌         | 23286/400000 [00:02<00:44, 8510.47it/s]  6%|▌         | 24138/400000 [00:02<00:44, 8488.64it/s]  6%|▌         | 24987/400000 [00:03<00:44, 8481.23it/s]  6%|▋         | 25836/400000 [00:03<00:44, 8470.18it/s]  7%|▋         | 26692/400000 [00:03<00:43, 8494.89it/s]  7%|▋         | 27548/400000 [00:03<00:43, 8512.79it/s]  7%|▋         | 28402/400000 [00:03<00:43, 8518.84it/s]  7%|▋         | 29254/400000 [00:03<00:43, 8510.70it/s]  8%|▊         | 30106/400000 [00:03<00:43, 8480.29it/s]  8%|▊         | 30962/400000 [00:03<00:43, 8503.00it/s]  8%|▊         | 31817/400000 [00:03<00:43, 8514.23it/s]  8%|▊         | 32669/400000 [00:03<00:43, 8469.81it/s]  8%|▊         | 33526/400000 [00:04<00:43, 8498.28it/s]  9%|▊         | 34378/400000 [00:04<00:42, 8503.16it/s]  9%|▉         | 35234/400000 [00:04<00:42, 8517.29it/s]  9%|▉         | 36089/400000 [00:04<00:42, 8526.55it/s]  9%|▉         | 36942/400000 [00:04<00:42, 8515.55it/s]  9%|▉         | 37803/400000 [00:04<00:42, 8542.99it/s] 10%|▉         | 38658/400000 [00:04<00:42, 8483.76it/s] 10%|▉         | 39515/400000 [00:04<00:42, 8508.52it/s] 10%|█         | 40366/400000 [00:04<00:42, 8465.94it/s] 10%|█         | 41213/400000 [00:04<00:42, 8420.13it/s] 11%|█         | 42063/400000 [00:05<00:42, 8443.40it/s] 11%|█         | 42908/400000 [00:05<00:42, 8419.07it/s] 11%|█         | 43768/400000 [00:05<00:42, 8469.73it/s] 11%|█         | 44616/400000 [00:05<00:42, 8460.90it/s] 11%|█▏        | 45463/400000 [00:05<00:42, 8415.99it/s] 12%|█▏        | 46305/400000 [00:05<00:42, 8416.22it/s] 12%|█▏        | 47148/400000 [00:05<00:41, 8419.51it/s] 12%|█▏        | 48008/400000 [00:05<00:41, 8471.66it/s] 12%|█▏        | 48866/400000 [00:05<00:41, 8501.63it/s] 12%|█▏        | 49717/400000 [00:05<00:41, 8476.64it/s] 13%|█▎        | 50565/400000 [00:06<00:41, 8474.47it/s] 13%|█▎        | 51413/400000 [00:06<00:41, 8346.78it/s] 13%|█▎        | 52260/400000 [00:06<00:41, 8383.08it/s] 13%|█▎        | 53117/400000 [00:06<00:41, 8435.43it/s] 13%|█▎        | 53961/400000 [00:06<00:41, 8389.16it/s] 14%|█▎        | 54819/400000 [00:06<00:40, 8445.44it/s] 14%|█▍        | 55670/400000 [00:06<00:40, 8464.13it/s] 14%|█▍        | 56529/400000 [00:06<00:40, 8500.81it/s] 14%|█▍        | 57383/400000 [00:06<00:40, 8510.22it/s] 15%|█▍        | 58235/400000 [00:06<00:40, 8415.17it/s] 15%|█▍        | 59089/400000 [00:07<00:40, 8451.06it/s] 15%|█▍        | 59943/400000 [00:07<00:40, 8475.10it/s] 15%|█▌        | 60792/400000 [00:07<00:40, 8477.26it/s] 15%|█▌        | 61653/400000 [00:07<00:39, 8515.44it/s] 16%|█▌        | 62505/400000 [00:07<00:39, 8515.29it/s] 16%|█▌        | 63357/400000 [00:07<00:39, 8505.75it/s] 16%|█▌        | 64217/400000 [00:07<00:39, 8531.34it/s] 16%|█▋        | 65071/400000 [00:07<00:39, 8517.15it/s] 16%|█▋        | 65931/400000 [00:07<00:39, 8540.00it/s] 17%|█▋        | 66786/400000 [00:07<00:39, 8494.77it/s] 17%|█▋        | 67639/400000 [00:08<00:39, 8504.06it/s] 17%|█▋        | 68490/400000 [00:08<00:39, 8490.15it/s] 17%|█▋        | 69340/400000 [00:08<00:39, 8440.62it/s] 18%|█▊        | 70192/400000 [00:08<00:38, 8461.49it/s] 18%|█▊        | 71039/400000 [00:08<00:39, 8430.17it/s] 18%|█▊        | 71886/400000 [00:08<00:38, 8439.28it/s] 18%|█▊        | 72743/400000 [00:08<00:38, 8477.96it/s] 18%|█▊        | 73591/400000 [00:08<00:38, 8474.62it/s] 19%|█▊        | 74444/400000 [00:08<00:38, 8489.87it/s] 19%|█▉        | 75294/400000 [00:08<00:39, 8289.18it/s] 19%|█▉        | 76142/400000 [00:09<00:38, 8343.58it/s] 19%|█▉        | 76999/400000 [00:09<00:38, 8408.34it/s] 19%|█▉        | 77848/400000 [00:09<00:38, 8429.90it/s] 20%|█▉        | 78704/400000 [00:09<00:37, 8467.16it/s] 20%|█▉        | 79552/400000 [00:09<00:37, 8443.24it/s] 20%|██        | 80400/400000 [00:09<00:37, 8452.18it/s] 20%|██        | 81259/400000 [00:09<00:37, 8491.84it/s] 21%|██        | 82109/400000 [00:09<00:37, 8488.18it/s] 21%|██        | 82965/400000 [00:09<00:37, 8507.50it/s] 21%|██        | 83816/400000 [00:09<00:37, 8489.65it/s] 21%|██        | 84666/400000 [00:10<00:37, 8451.67it/s] 21%|██▏       | 85521/400000 [00:10<00:37, 8480.05it/s] 22%|██▏       | 86370/400000 [00:10<00:37, 8434.20it/s] 22%|██▏       | 87214/400000 [00:10<00:37, 8421.73it/s] 22%|██▏       | 88061/400000 [00:10<00:36, 8431.99it/s] 22%|██▏       | 88905/400000 [00:10<00:36, 8422.21it/s] 22%|██▏       | 89755/400000 [00:10<00:36, 8443.88it/s] 23%|██▎       | 90606/400000 [00:10<00:36, 8461.41it/s] 23%|██▎       | 91468/400000 [00:10<00:36, 8505.51it/s] 23%|██▎       | 92319/400000 [00:10<00:36, 8474.64it/s] 23%|██▎       | 93179/400000 [00:11<00:36, 8509.38it/s] 24%|██▎       | 94037/400000 [00:11<00:35, 8528.09it/s] 24%|██▎       | 94890/400000 [00:11<00:36, 8464.40it/s] 24%|██▍       | 95737/400000 [00:11<00:36, 8293.31it/s] 24%|██▍       | 96586/400000 [00:11<00:36, 8348.84it/s] 24%|██▍       | 97440/400000 [00:11<00:36, 8404.29it/s] 25%|██▍       | 98304/400000 [00:11<00:35, 8472.42it/s] 25%|██▍       | 99152/400000 [00:11<00:35, 8470.28it/s] 25%|██▌       | 100010/400000 [00:11<00:35, 8500.63it/s] 25%|██▌       | 100871/400000 [00:11<00:35, 8533.08it/s] 25%|██▌       | 101725/400000 [00:12<00:35, 8517.50it/s] 26%|██▌       | 102588/400000 [00:12<00:34, 8549.95it/s] 26%|██▌       | 103444/400000 [00:12<00:34, 8487.36it/s] 26%|██▌       | 104293/400000 [00:12<00:35, 8286.23it/s] 26%|██▋       | 105155/400000 [00:12<00:35, 8380.78it/s] 26%|██▋       | 105995/400000 [00:12<00:35, 8381.87it/s] 27%|██▋       | 106850/400000 [00:12<00:34, 8431.10it/s] 27%|██▋       | 107694/400000 [00:12<00:34, 8423.16it/s] 27%|██▋       | 108540/400000 [00:12<00:34, 8431.56it/s] 27%|██▋       | 109384/400000 [00:12<00:34, 8433.70it/s] 28%|██▊       | 110231/400000 [00:13<00:34, 8443.67it/s] 28%|██▊       | 111084/400000 [00:13<00:34, 8468.61it/s] 28%|██▊       | 111934/400000 [00:13<00:33, 8475.67it/s] 28%|██▊       | 112788/400000 [00:13<00:33, 8493.83it/s] 28%|██▊       | 113650/400000 [00:13<00:33, 8530.69it/s] 29%|██▊       | 114504/400000 [00:13<00:34, 8348.18it/s] 29%|██▉       | 115343/400000 [00:13<00:34, 8357.90it/s] 29%|██▉       | 116190/400000 [00:13<00:33, 8390.47it/s] 29%|██▉       | 117049/400000 [00:13<00:33, 8446.55it/s] 29%|██▉       | 117906/400000 [00:13<00:33, 8482.67it/s] 30%|██▉       | 118755/400000 [00:14<00:33, 8474.54it/s] 30%|██▉       | 119609/400000 [00:14<00:33, 8492.11it/s] 30%|███       | 120459/400000 [00:14<00:32, 8474.30it/s] 30%|███       | 121307/400000 [00:14<00:33, 8444.27it/s] 31%|███       | 122162/400000 [00:14<00:32, 8473.72it/s] 31%|███       | 123010/400000 [00:14<00:32, 8445.14it/s] 31%|███       | 123865/400000 [00:14<00:32, 8476.09it/s] 31%|███       | 124719/400000 [00:14<00:32, 8495.03it/s] 31%|███▏      | 125572/400000 [00:14<00:32, 8503.07it/s] 32%|███▏      | 126423/400000 [00:14<00:32, 8451.93it/s] 32%|███▏      | 127269/400000 [00:15<00:33, 8172.13it/s] 32%|███▏      | 128126/400000 [00:15<00:32, 8284.69it/s] 32%|███▏      | 128967/400000 [00:15<00:32, 8320.59it/s] 32%|███▏      | 129811/400000 [00:15<00:32, 8354.85it/s] 33%|███▎      | 130673/400000 [00:15<00:31, 8430.95it/s] 33%|███▎      | 131524/400000 [00:15<00:31, 8452.07it/s] 33%|███▎      | 132374/400000 [00:15<00:31, 8464.76it/s] 33%|███▎      | 133225/400000 [00:15<00:31, 8477.17it/s] 34%|███▎      | 134074/400000 [00:15<00:31, 8454.22it/s] 34%|███▎      | 134922/400000 [00:16<00:31, 8460.35it/s] 34%|███▍      | 135769/400000 [00:16<00:31, 8378.12it/s] 34%|███▍      | 136622/400000 [00:16<00:31, 8420.53it/s] 34%|███▍      | 137482/400000 [00:16<00:30, 8473.07it/s] 35%|███▍      | 138330/400000 [00:16<00:30, 8446.04it/s] 35%|███▍      | 139193/400000 [00:16<00:30, 8500.33it/s] 35%|███▌      | 140047/400000 [00:16<00:30, 8509.60it/s] 35%|███▌      | 140905/400000 [00:16<00:30, 8530.52it/s] 35%|███▌      | 141764/400000 [00:16<00:30, 8547.83it/s] 36%|███▌      | 142619/400000 [00:16<00:30, 8514.03it/s] 36%|███▌      | 143483/400000 [00:17<00:30, 8548.76it/s] 36%|███▌      | 144338/400000 [00:17<00:29, 8537.11it/s] 36%|███▋      | 145200/400000 [00:17<00:29, 8560.10it/s] 37%|███▋      | 146062/400000 [00:17<00:29, 8576.82it/s] 37%|███▋      | 146920/400000 [00:17<00:29, 8510.29it/s] 37%|███▋      | 147772/400000 [00:17<00:29, 8502.72it/s] 37%|███▋      | 148623/400000 [00:17<00:29, 8486.30it/s] 37%|███▋      | 149477/400000 [00:17<00:29, 8501.58it/s] 38%|███▊      | 150331/400000 [00:17<00:29, 8511.32it/s] 38%|███▊      | 151183/400000 [00:17<00:29, 8486.41it/s] 38%|███▊      | 152034/400000 [00:18<00:29, 8491.20it/s] 38%|███▊      | 152886/400000 [00:18<00:29, 8497.90it/s] 38%|███▊      | 153748/400000 [00:18<00:28, 8534.00it/s] 39%|███▊      | 154606/400000 [00:18<00:28, 8544.87it/s] 39%|███▉      | 155461/400000 [00:18<00:28, 8482.26it/s] 39%|███▉      | 156319/400000 [00:18<00:28, 8510.05it/s] 39%|███▉      | 157176/400000 [00:18<00:28, 8525.79it/s] 40%|███▉      | 158029/400000 [00:18<00:28, 8421.95it/s] 40%|███▉      | 158872/400000 [00:18<00:28, 8357.72it/s] 40%|███▉      | 159709/400000 [00:18<00:28, 8360.53it/s] 40%|████      | 160563/400000 [00:19<00:28, 8411.05it/s] 40%|████      | 161405/400000 [00:19<00:28, 8393.54it/s] 41%|████      | 162264/400000 [00:19<00:28, 8451.29it/s] 41%|████      | 163125/400000 [00:19<00:27, 8495.53it/s] 41%|████      | 163975/400000 [00:19<00:27, 8455.33it/s] 41%|████      | 164829/400000 [00:19<00:27, 8478.95it/s] 41%|████▏     | 165678/400000 [00:19<00:27, 8446.00it/s] 42%|████▏     | 166529/400000 [00:19<00:27, 8462.85it/s] 42%|████▏     | 167376/400000 [00:19<00:28, 8228.33it/s] 42%|████▏     | 168201/400000 [00:19<00:28, 8169.07it/s] 42%|████▏     | 169059/400000 [00:20<00:27, 8286.55it/s] 42%|████▏     | 169909/400000 [00:20<00:27, 8346.76it/s] 43%|████▎     | 170772/400000 [00:20<00:27, 8428.62it/s] 43%|████▎     | 171633/400000 [00:20<00:26, 8479.62it/s] 43%|████▎     | 172482/400000 [00:20<00:26, 8464.81it/s] 43%|████▎     | 173337/400000 [00:20<00:26, 8489.22it/s] 44%|████▎     | 174188/400000 [00:20<00:26, 8493.42it/s] 44%|████▍     | 175038/400000 [00:20<00:26, 8393.12it/s] 44%|████▍     | 175880/400000 [00:20<00:26, 8400.56it/s] 44%|████▍     | 176729/400000 [00:20<00:26, 8424.74it/s] 44%|████▍     | 177572/400000 [00:21<00:26, 8419.79it/s] 45%|████▍     | 178427/400000 [00:21<00:26, 8458.35it/s] 45%|████▍     | 179286/400000 [00:21<00:25, 8496.72it/s] 45%|████▌     | 180146/400000 [00:21<00:25, 8526.39it/s] 45%|████▌     | 180999/400000 [00:21<00:25, 8437.61it/s] 45%|████▌     | 181854/400000 [00:21<00:25, 8470.25it/s] 46%|████▌     | 182711/400000 [00:21<00:25, 8499.28it/s] 46%|████▌     | 183566/400000 [00:21<00:25, 8513.32it/s] 46%|████▌     | 184422/400000 [00:21<00:25, 8525.67it/s] 46%|████▋     | 185275/400000 [00:21<00:25, 8439.38it/s] 47%|████▋     | 186131/400000 [00:22<00:25, 8472.61it/s] 47%|████▋     | 186979/400000 [00:22<00:25, 8428.78it/s] 47%|████▋     | 187824/400000 [00:22<00:25, 8434.31it/s] 47%|████▋     | 188682/400000 [00:22<00:24, 8474.79it/s] 47%|████▋     | 189530/400000 [00:22<00:25, 8388.27it/s] 48%|████▊     | 190384/400000 [00:22<00:24, 8430.41it/s] 48%|████▊     | 191228/400000 [00:22<00:24, 8387.49it/s] 48%|████▊     | 192081/400000 [00:22<00:24, 8429.71it/s] 48%|████▊     | 192925/400000 [00:22<00:24, 8428.90it/s] 48%|████▊     | 193769/400000 [00:22<00:25, 8176.85it/s] 49%|████▊     | 194618/400000 [00:23<00:24, 8267.00it/s] 49%|████▉     | 195449/400000 [00:23<00:24, 8278.72it/s] 49%|████▉     | 196278/400000 [00:23<00:24, 8204.95it/s] 49%|████▉     | 197110/400000 [00:23<00:24, 8238.15it/s] 49%|████▉     | 197956/400000 [00:23<00:24, 8303.12it/s] 50%|████▉     | 198816/400000 [00:23<00:23, 8388.62it/s] 50%|████▉     | 199664/400000 [00:23<00:23, 8414.43it/s] 50%|█████     | 200506/400000 [00:23<00:23, 8400.22it/s] 50%|█████     | 201363/400000 [00:23<00:23, 8449.31it/s] 51%|█████     | 202209/400000 [00:23<00:23, 8407.56it/s] 51%|█████     | 203058/400000 [00:24<00:23, 8431.64it/s] 51%|█████     | 203903/400000 [00:24<00:23, 8434.58it/s] 51%|█████     | 204762/400000 [00:24<00:23, 8479.02it/s] 51%|█████▏    | 205611/400000 [00:24<00:22, 8477.53it/s] 52%|█████▏    | 206459/400000 [00:24<00:23, 8393.89it/s] 52%|█████▏    | 207318/400000 [00:24<00:22, 8450.44it/s] 52%|█████▏    | 208165/400000 [00:24<00:22, 8456.26it/s] 52%|█████▏    | 209011/400000 [00:24<00:22, 8436.94it/s] 52%|█████▏    | 209860/400000 [00:24<00:22, 8451.22it/s] 53%|█████▎    | 210706/400000 [00:24<00:22, 8296.58it/s] 53%|█████▎    | 211561/400000 [00:25<00:22, 8368.84it/s] 53%|█████▎    | 212415/400000 [00:25<00:22, 8418.59it/s] 53%|█████▎    | 213272/400000 [00:25<00:22, 8460.32it/s] 54%|█████▎    | 214119/400000 [00:25<00:21, 8458.55it/s] 54%|█████▎    | 214966/400000 [00:25<00:22, 8383.60it/s] 54%|█████▍    | 215827/400000 [00:25<00:21, 8447.44it/s] 54%|█████▍    | 216689/400000 [00:25<00:21, 8498.13it/s] 54%|█████▍    | 217540/400000 [00:25<00:21, 8486.61it/s] 55%|█████▍    | 218389/400000 [00:25<00:21, 8486.33it/s] 55%|█████▍    | 219238/400000 [00:26<00:21, 8425.44it/s] 55%|█████▌    | 220092/400000 [00:26<00:21, 8457.56it/s] 55%|█████▌    | 220948/400000 [00:26<00:21, 8487.65it/s] 55%|█████▌    | 221797/400000 [00:26<00:20, 8486.27it/s] 56%|█████▌    | 222659/400000 [00:26<00:20, 8523.41it/s] 56%|█████▌    | 223512/400000 [00:26<00:20, 8464.57it/s] 56%|█████▌    | 224370/400000 [00:26<00:20, 8497.31it/s] 56%|█████▋    | 225234/400000 [00:26<00:20, 8539.55it/s] 57%|█████▋    | 226089/400000 [00:26<00:20, 8508.55it/s] 57%|█████▋    | 226946/400000 [00:26<00:20, 8526.71it/s] 57%|█████▋    | 227799/400000 [00:27<00:20, 8467.48it/s] 57%|█████▋    | 228658/400000 [00:27<00:20, 8502.49it/s] 57%|█████▋    | 229521/400000 [00:27<00:19, 8538.66it/s] 58%|█████▊    | 230376/400000 [00:27<00:19, 8486.69it/s] 58%|█████▊    | 231225/400000 [00:27<00:19, 8444.40it/s] 58%|█████▊    | 232070/400000 [00:27<00:20, 8338.93it/s] 58%|█████▊    | 232924/400000 [00:27<00:19, 8396.01it/s] 58%|█████▊    | 233781/400000 [00:27<00:19, 8446.84it/s] 59%|█████▊    | 234630/400000 [00:27<00:19, 8457.23it/s] 59%|█████▉    | 235489/400000 [00:27<00:19, 8494.89it/s] 59%|█████▉    | 236339/400000 [00:28<00:19, 8376.66it/s] 59%|█████▉    | 237180/400000 [00:28<00:19, 8384.65it/s] 60%|█████▉    | 238028/400000 [00:28<00:19, 8412.43it/s] 60%|█████▉    | 238870/400000 [00:28<00:19, 8262.76it/s] 60%|█████▉    | 239732/400000 [00:28<00:19, 8364.60it/s] 60%|██████    | 240581/400000 [00:28<00:18, 8399.14it/s] 60%|██████    | 241440/400000 [00:28<00:18, 8454.31it/s] 61%|██████    | 242300/400000 [00:28<00:18, 8496.13it/s] 61%|██████    | 243151/400000 [00:28<00:18, 8375.58it/s] 61%|██████    | 244009/400000 [00:28<00:18, 8434.94it/s] 61%|██████    | 244854/400000 [00:29<00:18, 8410.94it/s] 61%|██████▏   | 245710/400000 [00:29<00:18, 8452.47it/s] 62%|██████▏   | 246568/400000 [00:29<00:18, 8488.94it/s] 62%|██████▏   | 247418/400000 [00:29<00:18, 8471.73it/s] 62%|██████▏   | 248276/400000 [00:29<00:17, 8503.10it/s] 62%|██████▏   | 249127/400000 [00:29<00:17, 8485.37it/s] 62%|██████▏   | 249985/400000 [00:29<00:17, 8512.22it/s] 63%|██████▎   | 250847/400000 [00:29<00:17, 8543.51it/s] 63%|██████▎   | 251702/400000 [00:29<00:17, 8499.92it/s] 63%|██████▎   | 252560/400000 [00:29<00:17, 8522.03it/s] 63%|██████▎   | 253413/400000 [00:30<00:17, 8441.69it/s] 64%|██████▎   | 254266/400000 [00:30<00:17, 8466.97it/s] 64%|██████▍   | 255123/400000 [00:30<00:17, 8497.25it/s] 64%|██████▍   | 255978/400000 [00:30<00:16, 8512.97it/s] 64%|██████▍   | 256838/400000 [00:30<00:16, 8536.80it/s] 64%|██████▍   | 257692/400000 [00:30<00:16, 8439.44it/s] 65%|██████▍   | 258552/400000 [00:30<00:16, 8486.34it/s] 65%|██████▍   | 259412/400000 [00:30<00:16, 8518.39it/s] 65%|██████▌   | 260265/400000 [00:30<00:16, 8487.84it/s] 65%|██████▌   | 261114/400000 [00:30<00:16, 8470.12it/s] 65%|██████▌   | 261962/400000 [00:31<00:16, 8435.26it/s] 66%|██████▌   | 262822/400000 [00:31<00:16, 8483.30it/s] 66%|██████▌   | 263684/400000 [00:31<00:15, 8523.31it/s] 66%|██████▌   | 264537/400000 [00:31<00:15, 8497.93it/s] 66%|██████▋   | 265395/400000 [00:31<00:15, 8520.93it/s] 67%|██████▋   | 266248/400000 [00:31<00:15, 8401.86it/s] 67%|██████▋   | 267089/400000 [00:31<00:16, 8281.55it/s] 67%|██████▋   | 267918/400000 [00:31<00:16, 8037.00it/s] 67%|██████▋   | 268733/400000 [00:31<00:16, 8070.50it/s] 67%|██████▋   | 269583/400000 [00:31<00:15, 8193.38it/s] 68%|██████▊   | 270432/400000 [00:32<00:15, 8277.93it/s] 68%|██████▊   | 271278/400000 [00:32<00:15, 8331.65it/s] 68%|██████▊   | 272138/400000 [00:32<00:15, 8408.52it/s] 68%|██████▊   | 272988/400000 [00:32<00:15, 8433.40it/s] 68%|██████▊   | 273847/400000 [00:32<00:14, 8477.69it/s] 69%|██████▊   | 274697/400000 [00:32<00:14, 8482.61it/s] 69%|██████▉   | 275556/400000 [00:32<00:14, 8514.39it/s] 69%|██████▉   | 276413/400000 [00:32<00:14, 8529.17it/s] 69%|██████▉   | 277267/400000 [00:32<00:14, 8529.87it/s] 70%|██████▉   | 278121/400000 [00:32<00:14, 8507.31it/s] 70%|██████▉   | 278972/400000 [00:33<00:14, 8387.80it/s] 70%|██████▉   | 279823/400000 [00:33<00:14, 8421.29it/s] 70%|███████   | 280677/400000 [00:33<00:14, 8455.71it/s] 70%|███████   | 281530/400000 [00:33<00:13, 8475.70it/s] 71%|███████   | 282391/400000 [00:33<00:13, 8514.64it/s] 71%|███████   | 283243/400000 [00:33<00:13, 8505.63it/s] 71%|███████   | 284094/400000 [00:33<00:13, 8499.13it/s] 71%|███████   | 284945/400000 [00:33<00:13, 8441.49it/s] 71%|███████▏  | 285795/400000 [00:33<00:13, 8457.99it/s] 72%|███████▏  | 286652/400000 [00:33<00:13, 8488.87it/s] 72%|███████▏  | 287501/400000 [00:34<00:13, 8473.32it/s] 72%|███████▏  | 288356/400000 [00:34<00:13, 8494.23it/s] 72%|███████▏  | 289221/400000 [00:34<00:12, 8538.66it/s] 73%|███████▎  | 290075/400000 [00:34<00:12, 8516.21it/s] 73%|███████▎  | 290936/400000 [00:34<00:12, 8542.48it/s] 73%|███████▎  | 291791/400000 [00:34<00:12, 8486.33it/s] 73%|███████▎  | 292640/400000 [00:34<00:12, 8454.69it/s] 73%|███████▎  | 293500/400000 [00:34<00:12, 8495.62it/s] 74%|███████▎  | 294350/400000 [00:34<00:12, 8495.25it/s] 74%|███████▍  | 295200/400000 [00:34<00:12, 8361.23it/s] 74%|███████▍  | 296037/400000 [00:35<00:12, 8233.34it/s] 74%|███████▍  | 296890/400000 [00:35<00:12, 8318.10it/s] 74%|███████▍  | 297748/400000 [00:35<00:12, 8394.86it/s] 75%|███████▍  | 298589/400000 [00:35<00:12, 8392.04it/s] 75%|███████▍  | 299430/400000 [00:35<00:11, 8394.49it/s] 75%|███████▌  | 300282/400000 [00:35<00:11, 8431.17it/s] 75%|███████▌  | 301143/400000 [00:35<00:11, 8482.17it/s] 76%|███████▌  | 302000/400000 [00:35<00:11, 8506.16it/s] 76%|███████▌  | 302851/400000 [00:35<00:11, 8477.53it/s] 76%|███████▌  | 303711/400000 [00:35<00:11, 8511.23it/s] 76%|███████▌  | 304563/400000 [00:36<00:11, 8483.08it/s] 76%|███████▋  | 305412/400000 [00:36<00:11, 8475.60it/s] 77%|███████▋  | 306268/400000 [00:36<00:11, 8498.81it/s] 77%|███████▋  | 307119/400000 [00:36<00:10, 8502.01it/s] 77%|███████▋  | 307984/400000 [00:36<00:10, 8544.76it/s] 77%|███████▋  | 308839/400000 [00:36<00:10, 8518.03it/s] 77%|███████▋  | 309691/400000 [00:36<00:10, 8464.79it/s] 78%|███████▊  | 310546/400000 [00:36<00:10, 8486.64it/s] 78%|███████▊  | 311395/400000 [00:36<00:10, 8484.69it/s] 78%|███████▊  | 312247/400000 [00:37<00:10, 8494.92it/s] 78%|███████▊  | 313097/400000 [00:37<00:10, 8481.50it/s] 78%|███████▊  | 313954/400000 [00:37<00:10, 8505.21it/s] 79%|███████▊  | 314805/400000 [00:37<00:10, 8488.77it/s] 79%|███████▉  | 315659/400000 [00:37<00:09, 8501.76it/s] 79%|███████▉  | 316510/400000 [00:37<00:09, 8392.42it/s] 79%|███████▉  | 317352/400000 [00:37<00:09, 8400.40it/s] 80%|███████▉  | 318208/400000 [00:37<00:09, 8445.08it/s] 80%|███████▉  | 319067/400000 [00:37<00:09, 8486.96it/s] 80%|███████▉  | 319919/400000 [00:37<00:09, 8496.61it/s] 80%|████████  | 320769/400000 [00:38<00:09, 8485.73it/s] 80%|████████  | 321618/400000 [00:38<00:09, 8460.80it/s] 81%|████████  | 322465/400000 [00:38<00:09, 8456.19it/s] 81%|████████  | 323321/400000 [00:38<00:09, 8485.81it/s] 81%|████████  | 324180/400000 [00:38<00:08, 8514.18it/s] 81%|████████▏ | 325032/400000 [00:38<00:08, 8473.65it/s] 81%|████████▏ | 325880/400000 [00:38<00:08, 8434.33it/s] 82%|████████▏ | 326735/400000 [00:38<00:08, 8467.19it/s] 82%|████████▏ | 327587/400000 [00:38<00:08, 8481.15it/s] 82%|████████▏ | 328436/400000 [00:38<00:08, 8357.03it/s] 82%|████████▏ | 329286/400000 [00:39<00:08, 8395.60it/s] 83%|████████▎ | 330127/400000 [00:39<00:08, 8399.49it/s] 83%|████████▎ | 330988/400000 [00:39<00:08, 8460.17it/s] 83%|████████▎ | 331845/400000 [00:39<00:08, 8492.17it/s] 83%|████████▎ | 332695/400000 [00:39<00:07, 8494.07it/s] 83%|████████▎ | 333545/400000 [00:39<00:07, 8486.96it/s] 84%|████████▎ | 334394/400000 [00:39<00:07, 8451.46it/s] 84%|████████▍ | 335242/400000 [00:39<00:07, 8457.54it/s] 84%|████████▍ | 336102/400000 [00:39<00:07, 8497.53it/s] 84%|████████▍ | 336952/400000 [00:39<00:07, 8462.91it/s] 84%|████████▍ | 337799/400000 [00:40<00:07, 8447.03it/s] 85%|████████▍ | 338644/400000 [00:40<00:07, 8436.58it/s] 85%|████████▍ | 339490/400000 [00:40<00:07, 8442.10it/s] 85%|████████▌ | 340335/400000 [00:40<00:07, 8417.58it/s] 85%|████████▌ | 341181/400000 [00:40<00:06, 8428.93it/s] 86%|████████▌ | 342024/400000 [00:40<00:06, 8411.10it/s] 86%|████████▌ | 342866/400000 [00:40<00:06, 8392.34it/s] 86%|████████▌ | 343706/400000 [00:40<00:06, 8367.78it/s] 86%|████████▌ | 344565/400000 [00:40<00:06, 8433.01it/s] 86%|████████▋ | 345421/400000 [00:40<00:06, 8469.00it/s] 87%|████████▋ | 346271/400000 [00:41<00:06, 8477.72it/s] 87%|████████▋ | 347119/400000 [00:41<00:06, 8466.62it/s] 87%|████████▋ | 347966/400000 [00:41<00:06, 8448.50it/s] 87%|████████▋ | 348822/400000 [00:41<00:06, 8480.68it/s] 87%|████████▋ | 349672/400000 [00:41<00:05, 8485.64it/s] 88%|████████▊ | 350521/400000 [00:41<00:05, 8327.55it/s] 88%|████████▊ | 351355/400000 [00:41<00:05, 8127.81it/s] 88%|████████▊ | 352170/400000 [00:41<00:05, 8132.22it/s] 88%|████████▊ | 353031/400000 [00:41<00:05, 8269.48it/s] 88%|████████▊ | 353883/400000 [00:41<00:05, 8340.73it/s] 89%|████████▊ | 354734/400000 [00:42<00:05, 8388.04it/s] 89%|████████▉ | 355587/400000 [00:42<00:05, 8428.01it/s] 89%|████████▉ | 356431/400000 [00:42<00:05, 8399.06it/s] 89%|████████▉ | 357289/400000 [00:42<00:05, 8450.85it/s] 90%|████████▉ | 358144/400000 [00:42<00:04, 8480.10it/s] 90%|████████▉ | 358993/400000 [00:42<00:04, 8465.77it/s] 90%|████████▉ | 359840/400000 [00:42<00:04, 8435.92it/s] 90%|█████████ | 360684/400000 [00:42<00:04, 8399.74it/s] 90%|█████████ | 361537/400000 [00:42<00:04, 8437.80it/s] 91%|█████████ | 362381/400000 [00:42<00:04, 8383.26it/s] 91%|█████████ | 363221/400000 [00:43<00:04, 8386.76it/s] 91%|█████████ | 364065/400000 [00:43<00:04, 8401.16it/s] 91%|█████████ | 364906/400000 [00:43<00:04, 8341.33it/s] 91%|█████████▏| 365761/400000 [00:43<00:04, 8401.38it/s] 92%|█████████▏| 366619/400000 [00:43<00:03, 8453.26it/s] 92%|█████████▏| 367465/400000 [00:43<00:03, 8446.83it/s] 92%|█████████▏| 368310/400000 [00:43<00:03, 8445.07it/s] 92%|█████████▏| 369155/400000 [00:43<00:03, 8420.54it/s] 92%|█████████▏| 369998/400000 [00:43<00:03, 8379.00it/s] 93%|█████████▎| 370837/400000 [00:43<00:03, 8369.48it/s] 93%|█████████▎| 371682/400000 [00:44<00:03, 8391.64it/s] 93%|█████████▎| 372541/400000 [00:44<00:03, 8447.55it/s] 93%|█████████▎| 373389/400000 [00:44<00:03, 8456.62it/s] 94%|█████████▎| 374246/400000 [00:44<00:03, 8488.97it/s] 94%|█████████▍| 375103/400000 [00:44<00:02, 8512.32it/s] 94%|█████████▍| 375955/400000 [00:44<00:02, 8456.01it/s] 94%|█████████▍| 376806/400000 [00:44<00:02, 8468.83it/s] 94%|█████████▍| 377653/400000 [00:44<00:02, 8462.91it/s] 95%|█████████▍| 378500/400000 [00:44<00:02, 8464.86it/s] 95%|█████████▍| 379350/400000 [00:44<00:02, 8473.63it/s] 95%|█████████▌| 380203/400000 [00:45<00:02, 8487.90it/s] 95%|█████████▌| 381052/400000 [00:45<00:02, 8463.98it/s] 95%|█████████▌| 381899/400000 [00:45<00:02, 8387.51it/s] 96%|█████████▌| 382754/400000 [00:45<00:02, 8434.21it/s] 96%|█████████▌| 383611/400000 [00:45<00:01, 8474.43it/s] 96%|█████████▌| 384459/400000 [00:45<00:01, 8430.23it/s] 96%|█████████▋| 385310/400000 [00:45<00:01, 8453.72it/s] 97%|█████████▋| 386156/400000 [00:45<00:01, 8430.79it/s] 97%|█████████▋| 387000/400000 [00:45<00:01, 8413.89it/s] 97%|█████████▋| 387842/400000 [00:45<00:01, 8377.78it/s] 97%|█████████▋| 388680/400000 [00:46<00:01, 8367.33it/s] 97%|█████████▋| 389536/400000 [00:46<00:01, 8422.40it/s] 98%|█████████▊| 390379/400000 [00:46<00:01, 8405.17it/s] 98%|█████████▊| 391240/400000 [00:46<00:01, 8465.33it/s] 98%|█████████▊| 392099/400000 [00:46<00:00, 8500.58it/s] 98%|█████████▊| 392950/400000 [00:46<00:00, 8496.00it/s] 98%|█████████▊| 393800/400000 [00:46<00:00, 8420.73it/s] 99%|█████████▊| 394643/400000 [00:46<00:00, 8198.42it/s] 99%|█████████▉| 395491/400000 [00:46<00:00, 8280.86it/s] 99%|█████████▉| 396334/400000 [00:46<00:00, 8322.76it/s] 99%|█████████▉| 397169/400000 [00:47<00:00, 8330.32it/s]100%|█████████▉| 398020/400000 [00:47<00:00, 8381.61it/s]100%|█████████▉| 398867/400000 [00:47<00:00, 8405.12it/s]100%|█████████▉| 399717/400000 [00:47<00:00, 8431.58it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8435.35it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fbe6580c160>, <torchtext.data.dataset.TabularDataset object at 0x7fbe6580c2b0>, <torchtext.vocab.Vocab object at 0x7fbe6580c1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 34.99 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 58.19 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 30.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.99 file/s]2020-06-12 12:19:26.998992: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 12:19:27.003472: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-06-12 12:19:27.003651: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566f7d84010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 12:19:27.003666: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4087808/9912422 [00:00<00:00, 40333833.88it/s]9920512it [00:00, 35812002.30it/s]                             
0it [00:00, ?it/s]32768it [00:00, 710440.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467094.95it/s]1654784it [00:00, 11913835.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 227898.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
