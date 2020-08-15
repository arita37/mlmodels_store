
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9e78de6488> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9e78de6488> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9ee3a93400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9ee3a93400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9f02cafea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9f02cafea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9e90dcf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9e90dcf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9e90dcf0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 130273.40it/s] 45%|████▍     | 4448256/9912422 [00:00<00:29, 185870.40it/s]9920512it [00:00, 35537311.65it/s]                           
0it [00:00, ?it/s]32768it [00:00, 517456.20it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151763.62it/s]1654784it [00:00, 11043383.52it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207756.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e78d304a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e77fb9748>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9e90dc6d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9e90dc6d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9e90dc6d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:35,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:35,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:34,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:34,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:33,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:32,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:02,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:02,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:42,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:28,  4.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  8.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 12.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 44.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 42.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 42.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 42.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 41.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 41.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 41.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 41.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 41.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 41.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 41.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 41.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:01, 41.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:01, 41.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:01, 41.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:01, 41.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:01, 40.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 40.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 40.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 40.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 40.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 40.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 40.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 40.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 40.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 40.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 40.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 40.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 40.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 40.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 40.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 40.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 40.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 40.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 40.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 40.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 40.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 40.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 40.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 40.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 39.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 39.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 39.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 39.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 39.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 39.64 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 39.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 39.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 39.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 39.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 39.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 39.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 39.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 39.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 39.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 39.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 39.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 39.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 39.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 39.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 39.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.66s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 39.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 39.48 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.99s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 39.48 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.99s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.99s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.03 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.99s/ url]
0 examples [00:00, ? examples/s]2020-08-15 00:07:53.107353: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-15 00:07:53.122242: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-15 00:07:53.122446: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad5930e1c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-15 00:07:53.122468: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 189.10 examples/s]109 examples [00:00, 247.63 examples/s]197 examples [00:00, 315.40 examples/s]288 examples [00:00, 392.15 examples/s]380 examples [00:00, 473.54 examples/s]471 examples [00:00, 552.92 examples/s]562 examples [00:00, 626.11 examples/s]654 examples [00:00, 691.74 examples/s]742 examples [00:00, 737.34 examples/s]827 examples [00:01, 761.49 examples/s]914 examples [00:01, 790.27 examples/s]999 examples [00:01, 791.28 examples/s]1085 examples [00:01, 808.41 examples/s]1171 examples [00:01, 821.97 examples/s]1256 examples [00:01, 800.96 examples/s]1347 examples [00:01, 829.81 examples/s]1439 examples [00:01, 854.74 examples/s]1528 examples [00:01, 863.49 examples/s]1617 examples [00:01, 870.40 examples/s]1710 examples [00:02, 886.41 examples/s]1803 examples [00:02, 896.56 examples/s]1894 examples [00:02, 887.97 examples/s]1984 examples [00:02, 870.35 examples/s]2075 examples [00:02, 879.50 examples/s]2168 examples [00:02, 891.73 examples/s]2262 examples [00:02, 904.30 examples/s]2353 examples [00:02, 877.79 examples/s]2442 examples [00:02, 877.43 examples/s]2531 examples [00:02, 879.41 examples/s]2620 examples [00:03, 879.64 examples/s]2712 examples [00:03, 890.33 examples/s]2802 examples [00:03, 883.11 examples/s]2891 examples [00:03, 868.50 examples/s]2981 examples [00:03, 874.86 examples/s]3070 examples [00:03, 877.96 examples/s]3158 examples [00:03, 867.18 examples/s]3245 examples [00:03, 866.42 examples/s]3334 examples [00:03, 872.70 examples/s]3422 examples [00:03, 856.94 examples/s]3513 examples [00:04, 871.98 examples/s]3606 examples [00:04, 887.53 examples/s]3695 examples [00:04, 879.73 examples/s]3785 examples [00:04, 883.25 examples/s]3879 examples [00:04, 898.92 examples/s]3973 examples [00:04, 910.32 examples/s]4065 examples [00:04, 910.53 examples/s]4158 examples [00:04, 914.19 examples/s]4250 examples [00:04, 911.24 examples/s]4344 examples [00:04, 918.61 examples/s]4438 examples [00:05, 923.76 examples/s]4532 examples [00:05, 926.02 examples/s]4625 examples [00:05, 918.13 examples/s]4717 examples [00:05, 891.40 examples/s]4807 examples [00:05, 892.56 examples/s]4897 examples [00:05, 891.97 examples/s]4987 examples [00:05, 891.03 examples/s]5080 examples [00:05, 900.14 examples/s]5171 examples [00:05, 895.22 examples/s]5261 examples [00:06, 892.92 examples/s]5351 examples [00:06, 889.07 examples/s]5443 examples [00:06, 895.85 examples/s]5535 examples [00:06, 901.66 examples/s]5631 examples [00:06, 918.38 examples/s]5725 examples [00:06, 924.64 examples/s]5821 examples [00:06, 934.78 examples/s]5915 examples [00:06, 934.85 examples/s]6009 examples [00:06, 907.50 examples/s]6100 examples [00:06, 905.61 examples/s]6191 examples [00:07, 905.60 examples/s]6282 examples [00:07, 902.18 examples/s]6373 examples [00:07, 894.81 examples/s]6463 examples [00:07, 887.20 examples/s]6554 examples [00:07, 892.31 examples/s]6647 examples [00:07, 902.24 examples/s]6743 examples [00:07, 918.14 examples/s]6835 examples [00:07, 911.98 examples/s]6927 examples [00:07, 893.17 examples/s]7017 examples [00:07, 879.13 examples/s]7107 examples [00:08, 883.50 examples/s]7196 examples [00:08, 884.27 examples/s]7291 examples [00:08, 900.73 examples/s]7384 examples [00:08, 906.90 examples/s]7476 examples [00:08, 910.01 examples/s]7569 examples [00:08, 915.47 examples/s]7662 examples [00:08, 918.45 examples/s]7755 examples [00:08, 921.15 examples/s]7849 examples [00:08, 925.31 examples/s]7946 examples [00:08, 936.49 examples/s]8040 examples [00:09, 932.52 examples/s]8134 examples [00:09, 926.51 examples/s]8229 examples [00:09, 930.79 examples/s]8323 examples [00:09, 899.49 examples/s]8414 examples [00:09, 832.48 examples/s]8499 examples [00:09, 778.76 examples/s]8591 examples [00:09, 815.95 examples/s]8683 examples [00:09, 844.33 examples/s]8774 examples [00:09, 861.81 examples/s]8865 examples [00:10, 873.47 examples/s]8958 examples [00:10, 887.41 examples/s]9050 examples [00:10, 895.06 examples/s]9142 examples [00:10, 901.81 examples/s]9233 examples [00:10, 883.72 examples/s]9322 examples [00:10, 877.74 examples/s]9412 examples [00:10, 883.16 examples/s]9505 examples [00:10, 894.63 examples/s]9595 examples [00:10, 886.16 examples/s]9684 examples [00:10, 877.31 examples/s]9772 examples [00:11, 849.48 examples/s]9862 examples [00:11, 862.72 examples/s]9950 examples [00:11, 865.36 examples/s]10037 examples [00:11, 806.66 examples/s]10126 examples [00:11, 829.56 examples/s]10215 examples [00:11, 845.68 examples/s]10307 examples [00:11, 865.11 examples/s]10398 examples [00:11, 877.45 examples/s]10489 examples [00:11, 886.80 examples/s]10583 examples [00:12, 901.11 examples/s]10675 examples [00:12, 904.25 examples/s]10771 examples [00:12, 917.68 examples/s]10867 examples [00:12, 929.40 examples/s]10962 examples [00:12, 934.08 examples/s]11056 examples [00:12, 930.99 examples/s]11150 examples [00:12, 924.95 examples/s]11245 examples [00:12, 931.15 examples/s]11340 examples [00:12, 935.80 examples/s]11434 examples [00:12, 933.04 examples/s]11528 examples [00:13, 916.64 examples/s]11620 examples [00:13, 910.54 examples/s]11713 examples [00:13, 915.64 examples/s]11805 examples [00:13, 912.43 examples/s]11898 examples [00:13, 916.46 examples/s]11992 examples [00:13, 923.23 examples/s]12085 examples [00:13, 921.38 examples/s]12178 examples [00:13, 923.76 examples/s]12271 examples [00:13, 921.33 examples/s]12364 examples [00:13, 915.82 examples/s]12456 examples [00:14, 914.55 examples/s]12549 examples [00:14, 916.34 examples/s]12641 examples [00:14, 913.24 examples/s]12733 examples [00:14, 903.10 examples/s]12824 examples [00:14, 889.34 examples/s]12917 examples [00:14, 899.14 examples/s]13007 examples [00:14, 890.72 examples/s]13100 examples [00:14, 900.18 examples/s]13191 examples [00:14, 869.14 examples/s]13280 examples [00:14, 873.57 examples/s]13370 examples [00:15, 880.84 examples/s]13462 examples [00:15, 891.66 examples/s]13557 examples [00:15, 907.80 examples/s]13651 examples [00:15, 916.31 examples/s]13743 examples [00:15, 915.03 examples/s]13836 examples [00:15, 918.32 examples/s]13928 examples [00:15, 917.61 examples/s]14023 examples [00:15, 924.02 examples/s]14116 examples [00:15, 921.86 examples/s]14209 examples [00:15, 916.43 examples/s]14304 examples [00:16, 925.05 examples/s]14397 examples [00:16, 922.35 examples/s]14490 examples [00:16, 918.04 examples/s]14582 examples [00:16, 915.69 examples/s]14674 examples [00:16, 901.66 examples/s]14765 examples [00:16, 897.48 examples/s]14855 examples [00:16, 895.30 examples/s]14945 examples [00:16, 896.65 examples/s]15036 examples [00:16, 898.36 examples/s]15126 examples [00:16, 895.55 examples/s]15216 examples [00:17, 888.99 examples/s]15309 examples [00:17, 898.14 examples/s]15404 examples [00:17, 910.87 examples/s]15496 examples [00:17, 889.38 examples/s]15587 examples [00:17, 893.76 examples/s]15679 examples [00:17, 899.13 examples/s]15770 examples [00:17, 897.61 examples/s]15860 examples [00:17, 890.28 examples/s]15950 examples [00:17, 882.71 examples/s]16039 examples [00:18, 870.91 examples/s]16128 examples [00:18, 874.63 examples/s]16216 examples [00:18, 867.48 examples/s]16303 examples [00:18, 860.11 examples/s]16390 examples [00:18, 861.25 examples/s]16479 examples [00:18, 867.63 examples/s]16566 examples [00:18, 864.32 examples/s]16653 examples [00:18, 849.90 examples/s]16743 examples [00:18, 861.54 examples/s]16833 examples [00:18, 871.80 examples/s]16921 examples [00:19, 868.07 examples/s]17015 examples [00:19, 887.57 examples/s]17111 examples [00:19, 906.35 examples/s]17205 examples [00:19, 915.07 examples/s]17297 examples [00:19, 903.30 examples/s]17388 examples [00:19, 899.60 examples/s]17479 examples [00:19, 880.98 examples/s]17568 examples [00:19, 881.29 examples/s]17657 examples [00:19, 882.24 examples/s]17746 examples [00:19, 858.78 examples/s]17833 examples [00:20, 856.78 examples/s]17919 examples [00:20, 845.38 examples/s]18008 examples [00:20, 856.57 examples/s]18098 examples [00:20, 867.97 examples/s]18185 examples [00:20, 851.92 examples/s]18273 examples [00:20, 858.28 examples/s]18365 examples [00:20, 875.50 examples/s]18453 examples [00:20, 869.50 examples/s]18541 examples [00:20, 870.82 examples/s]18633 examples [00:20, 883.39 examples/s]18722 examples [00:21, 873.22 examples/s]18810 examples [00:21, 866.13 examples/s]18897 examples [00:21, 859.21 examples/s]18986 examples [00:21, 866.56 examples/s]19077 examples [00:21, 877.91 examples/s]19165 examples [00:21, 877.09 examples/s]19257 examples [00:21, 889.09 examples/s]19346 examples [00:21, 880.40 examples/s]19435 examples [00:21, 877.26 examples/s]19524 examples [00:22, 878.61 examples/s]19612 examples [00:22, 875.64 examples/s]19700 examples [00:22, 857.22 examples/s]19786 examples [00:22, 856.62 examples/s]19874 examples [00:22, 862.46 examples/s]19962 examples [00:22, 865.17 examples/s]20049 examples [00:22, 817.62 examples/s]20140 examples [00:22, 840.77 examples/s]20232 examples [00:22, 861.46 examples/s]20322 examples [00:22, 871.68 examples/s]20415 examples [00:23, 886.57 examples/s]20505 examples [00:23, 888.22 examples/s]20597 examples [00:23, 895.56 examples/s]20689 examples [00:23, 901.39 examples/s]20780 examples [00:23, 897.62 examples/s]20872 examples [00:23, 902.50 examples/s]20963 examples [00:23, 890.51 examples/s]21054 examples [00:23, 896.02 examples/s]21144 examples [00:23, 895.58 examples/s]21237 examples [00:23, 902.91 examples/s]21329 examples [00:24, 905.95 examples/s]21420 examples [00:24, 905.62 examples/s]21513 examples [00:24, 911.12 examples/s]21605 examples [00:24, 910.41 examples/s]21699 examples [00:24, 917.57 examples/s]21791 examples [00:24, 913.46 examples/s]21883 examples [00:24, 903.38 examples/s]21974 examples [00:24, 893.30 examples/s]22064 examples [00:24, 888.65 examples/s]22159 examples [00:24, 905.98 examples/s]22251 examples [00:25, 909.34 examples/s]22344 examples [00:25, 913.48 examples/s]22436 examples [00:25, 914.22 examples/s]22528 examples [00:25, 909.03 examples/s]22620 examples [00:25, 909.02 examples/s]22711 examples [00:25, 907.45 examples/s]22802 examples [00:25, 902.31 examples/s]22895 examples [00:25, 909.06 examples/s]22988 examples [00:25, 913.39 examples/s]23080 examples [00:25, 908.31 examples/s]23171 examples [00:26, 899.84 examples/s]23262 examples [00:26, 900.29 examples/s]23355 examples [00:26, 908.63 examples/s]23448 examples [00:26, 913.19 examples/s]23540 examples [00:26, 898.69 examples/s]23631 examples [00:26, 899.98 examples/s]23722 examples [00:26, 894.14 examples/s]23812 examples [00:26, 890.90 examples/s]23902 examples [00:26, 886.75 examples/s]23996 examples [00:27, 900.08 examples/s]24087 examples [00:27, 895.45 examples/s]24177 examples [00:27, 884.45 examples/s]24266 examples [00:27, 870.87 examples/s]24354 examples [00:27, 863.59 examples/s]24443 examples [00:27, 871.16 examples/s]24531 examples [00:27, 860.92 examples/s]24618 examples [00:27, 854.78 examples/s]24708 examples [00:27, 865.44 examples/s]24796 examples [00:27, 869.49 examples/s]24885 examples [00:28, 874.53 examples/s]24976 examples [00:28, 883.05 examples/s]25067 examples [00:28, 888.29 examples/s]25156 examples [00:28, 867.99 examples/s]25243 examples [00:28, 859.12 examples/s]25330 examples [00:28, 854.68 examples/s]25424 examples [00:28, 876.62 examples/s]25518 examples [00:28, 891.98 examples/s]25608 examples [00:28, 888.10 examples/s]25701 examples [00:28, 900.22 examples/s]25792 examples [00:29, 864.51 examples/s]25884 examples [00:29, 878.94 examples/s]25978 examples [00:29, 894.95 examples/s]26072 examples [00:29, 905.04 examples/s]26163 examples [00:29, 860.02 examples/s]26252 examples [00:29, 867.13 examples/s]26340 examples [00:29, 868.55 examples/s]26430 examples [00:29, 869.14 examples/s]26519 examples [00:29, 874.37 examples/s]26609 examples [00:29, 880.91 examples/s]26700 examples [00:30, 888.96 examples/s]26790 examples [00:30, 863.80 examples/s]26881 examples [00:30, 876.28 examples/s]26970 examples [00:30, 879.72 examples/s]27059 examples [00:30, 873.93 examples/s]27148 examples [00:30, 878.67 examples/s]27238 examples [00:30, 884.50 examples/s]27327 examples [00:30, 882.87 examples/s]27416 examples [00:30, 880.59 examples/s]27509 examples [00:31, 893.17 examples/s]27599 examples [00:31, 890.25 examples/s]27689 examples [00:31, 871.03 examples/s]27780 examples [00:31, 879.77 examples/s]27870 examples [00:31, 885.08 examples/s]27959 examples [00:31, 883.78 examples/s]28048 examples [00:31, 879.21 examples/s]28139 examples [00:31, 887.71 examples/s]28234 examples [00:31, 903.50 examples/s]28326 examples [00:31, 906.59 examples/s]28417 examples [00:32, 901.70 examples/s]28511 examples [00:32, 912.61 examples/s]28603 examples [00:32, 913.22 examples/s]28695 examples [00:32, 910.78 examples/s]28787 examples [00:32, 900.59 examples/s]28879 examples [00:32, 904.70 examples/s]28970 examples [00:32, 898.64 examples/s]29060 examples [00:32, 885.10 examples/s]29152 examples [00:32, 894.74 examples/s]29243 examples [00:32, 896.52 examples/s]29333 examples [00:33, 886.70 examples/s]29423 examples [00:33, 888.09 examples/s]29512 examples [00:33, 874.54 examples/s]29602 examples [00:33, 879.29 examples/s]29690 examples [00:33, 877.13 examples/s]29783 examples [00:33, 891.14 examples/s]29873 examples [00:33, 870.22 examples/s]29965 examples [00:33, 883.07 examples/s]30054 examples [00:33, 838.98 examples/s]30147 examples [00:33, 862.46 examples/s]30239 examples [00:34, 878.71 examples/s]30332 examples [00:34, 891.43 examples/s]30424 examples [00:34, 898.45 examples/s]30518 examples [00:34, 910.43 examples/s]30610 examples [00:34, 905.88 examples/s]30701 examples [00:34, 901.86 examples/s]30793 examples [00:34, 906.55 examples/s]30884 examples [00:34, 905.27 examples/s]30975 examples [00:34, 903.95 examples/s]31068 examples [00:34, 909.26 examples/s]31162 examples [00:35, 915.59 examples/s]31255 examples [00:35, 918.79 examples/s]31347 examples [00:35, 906.12 examples/s]31441 examples [00:35, 913.54 examples/s]31536 examples [00:35, 924.07 examples/s]31629 examples [00:35, 917.31 examples/s]31721 examples [00:35, 917.07 examples/s]31814 examples [00:35, 919.96 examples/s]31907 examples [00:35, 919.70 examples/s]32000 examples [00:36, 921.54 examples/s]32093 examples [00:36, 918.40 examples/s]32187 examples [00:36, 921.97 examples/s]32281 examples [00:36, 924.69 examples/s]32374 examples [00:36, 921.20 examples/s]32467 examples [00:36, 914.25 examples/s]32560 examples [00:36, 917.67 examples/s]32652 examples [00:36, 907.39 examples/s]32745 examples [00:36, 914.01 examples/s]32839 examples [00:36, 921.58 examples/s]32932 examples [00:37, 919.30 examples/s]33025 examples [00:37, 920.73 examples/s]33118 examples [00:37, 890.82 examples/s]33208 examples [00:37, 857.15 examples/s]33301 examples [00:37, 877.52 examples/s]33396 examples [00:37, 898.01 examples/s]33491 examples [00:37, 912.92 examples/s]33583 examples [00:37, 912.95 examples/s]33677 examples [00:37, 918.42 examples/s]33772 examples [00:37, 926.91 examples/s]33865 examples [00:38, 921.52 examples/s]33958 examples [00:38, 916.37 examples/s]34051 examples [00:38, 919.72 examples/s]34144 examples [00:38, 911.67 examples/s]34236 examples [00:38, 906.82 examples/s]34330 examples [00:38, 915.38 examples/s]34422 examples [00:38, 893.83 examples/s]34513 examples [00:38, 898.57 examples/s]34605 examples [00:38, 904.41 examples/s]34698 examples [00:38, 909.57 examples/s]34792 examples [00:39, 917.19 examples/s]34884 examples [00:39, 901.58 examples/s]34977 examples [00:39, 908.81 examples/s]35071 examples [00:39, 915.40 examples/s]35165 examples [00:39, 920.83 examples/s]35259 examples [00:39, 925.24 examples/s]35352 examples [00:39, 907.25 examples/s]35443 examples [00:39, 869.16 examples/s]35534 examples [00:39, 880.19 examples/s]35623 examples [00:40, 879.14 examples/s]35713 examples [00:40, 885.24 examples/s]35802 examples [00:40, 886.12 examples/s]35893 examples [00:40, 892.94 examples/s]35983 examples [00:40, 887.28 examples/s]36075 examples [00:40, 895.26 examples/s]36167 examples [00:40, 899.70 examples/s]36259 examples [00:40, 904.35 examples/s]36350 examples [00:40, 896.74 examples/s]36440 examples [00:40, 878.98 examples/s]36529 examples [00:41, 877.84 examples/s]36620 examples [00:41, 886.98 examples/s]36709 examples [00:41, 885.14 examples/s]36798 examples [00:41, 881.04 examples/s]36887 examples [00:41, 873.48 examples/s]36978 examples [00:41, 881.63 examples/s]37067 examples [00:41, 868.63 examples/s]37154 examples [00:41, 857.77 examples/s]37240 examples [00:41, 855.38 examples/s]37329 examples [00:41, 862.97 examples/s]37419 examples [00:42, 873.14 examples/s]37510 examples [00:42, 881.65 examples/s]37605 examples [00:42, 898.78 examples/s]37698 examples [00:42, 907.66 examples/s]37789 examples [00:42, 903.96 examples/s]37880 examples [00:42, 905.49 examples/s]37973 examples [00:42, 910.26 examples/s]38065 examples [00:42, 912.25 examples/s]38157 examples [00:42, 907.68 examples/s]38248 examples [00:42, 898.32 examples/s]38338 examples [00:43, 887.55 examples/s]38427 examples [00:43, 882.28 examples/s]38516 examples [00:43, 875.37 examples/s]38605 examples [00:43, 879.48 examples/s]38696 examples [00:43, 888.18 examples/s]38790 examples [00:43, 901.61 examples/s]38881 examples [00:43, 900.32 examples/s]38973 examples [00:43, 903.37 examples/s]39064 examples [00:43, 901.68 examples/s]39155 examples [00:43, 886.61 examples/s]39247 examples [00:44, 895.33 examples/s]39339 examples [00:44, 901.99 examples/s]39430 examples [00:44, 902.80 examples/s]39521 examples [00:44, 859.28 examples/s]39608 examples [00:44, 820.84 examples/s]39699 examples [00:44, 844.76 examples/s]39792 examples [00:44, 867.75 examples/s]39883 examples [00:44, 878.08 examples/s]39975 examples [00:44, 889.50 examples/s]40065 examples [00:45, 839.56 examples/s]40159 examples [00:45, 867.21 examples/s]40252 examples [00:45, 884.43 examples/s]40343 examples [00:45, 891.66 examples/s]40436 examples [00:45, 902.51 examples/s]40527 examples [00:45, 895.74 examples/s]40620 examples [00:45, 904.92 examples/s]40714 examples [00:45, 912.76 examples/s]40809 examples [00:45, 920.96 examples/s]40903 examples [00:45, 924.56 examples/s]40996 examples [00:46, 892.58 examples/s]41086 examples [00:46, 880.46 examples/s]41178 examples [00:46, 890.07 examples/s]41272 examples [00:46, 903.68 examples/s]41366 examples [00:46, 914.25 examples/s]41458 examples [00:46, 907.56 examples/s]41553 examples [00:46, 919.63 examples/s]41646 examples [00:46, 911.19 examples/s]41738 examples [00:46, 906.81 examples/s]41831 examples [00:46, 911.35 examples/s]41923 examples [00:47, 902.85 examples/s]42014 examples [00:47, 895.66 examples/s]42107 examples [00:47, 902.89 examples/s]42199 examples [00:47, 906.27 examples/s]42292 examples [00:47, 911.71 examples/s]42384 examples [00:47, 910.98 examples/s]42476 examples [00:47, 873.45 examples/s]42567 examples [00:47, 881.21 examples/s]42659 examples [00:47, 890.27 examples/s]42749 examples [00:48, 892.09 examples/s]42839 examples [00:48, 884.73 examples/s]42930 examples [00:48, 890.47 examples/s]43021 examples [00:48, 894.62 examples/s]43111 examples [00:48, 895.98 examples/s]43203 examples [00:48, 900.27 examples/s]43296 examples [00:48, 906.46 examples/s]43391 examples [00:48, 916.69 examples/s]43484 examples [00:48, 918.46 examples/s]43580 examples [00:48, 928.19 examples/s]43673 examples [00:49, 914.56 examples/s]43765 examples [00:49, 909.15 examples/s]43856 examples [00:49, 906.60 examples/s]43947 examples [00:49, 905.61 examples/s]44039 examples [00:49, 908.23 examples/s]44132 examples [00:49, 912.04 examples/s]44224 examples [00:49, 883.22 examples/s]44313 examples [00:49, 880.50 examples/s]44406 examples [00:49, 894.37 examples/s]44500 examples [00:49, 906.47 examples/s]44593 examples [00:50, 911.45 examples/s]44687 examples [00:50, 917.20 examples/s]44779 examples [00:50, 917.04 examples/s]44874 examples [00:50, 924.77 examples/s]44968 examples [00:50, 926.82 examples/s]45064 examples [00:50, 935.86 examples/s]45158 examples [00:50, 925.58 examples/s]45252 examples [00:50, 928.11 examples/s]45345 examples [00:50, 913.22 examples/s]45437 examples [00:50, 910.82 examples/s]45529 examples [00:51, 910.08 examples/s]45621 examples [00:51, 912.84 examples/s]45713 examples [00:51, 896.65 examples/s]45807 examples [00:51, 909.14 examples/s]45902 examples [00:51, 919.71 examples/s]45995 examples [00:51, 915.97 examples/s]46087 examples [00:51, 912.84 examples/s]46180 examples [00:51, 916.90 examples/s]46274 examples [00:51, 921.23 examples/s]46367 examples [00:51, 922.64 examples/s]46460 examples [00:52, 920.69 examples/s]46553 examples [00:52, 900.63 examples/s]46644 examples [00:52, 878.73 examples/s]46733 examples [00:52, 856.25 examples/s]46819 examples [00:52, 847.26 examples/s]46908 examples [00:52, 858.28 examples/s]46995 examples [00:52, 832.35 examples/s]47083 examples [00:52, 843.77 examples/s]47171 examples [00:52, 854.17 examples/s]47260 examples [00:53, 861.87 examples/s]47349 examples [00:53, 869.52 examples/s]47437 examples [00:53, 872.44 examples/s]47527 examples [00:53, 880.15 examples/s]47618 examples [00:53, 888.08 examples/s]47713 examples [00:53, 903.43 examples/s]47806 examples [00:53, 910.33 examples/s]47898 examples [00:53, 897.04 examples/s]47996 examples [00:53, 918.39 examples/s]48089 examples [00:53, 909.97 examples/s]48181 examples [00:54, 887.27 examples/s]48270 examples [00:54, 868.54 examples/s]48358 examples [00:54, 868.67 examples/s]48448 examples [00:54, 875.68 examples/s]48536 examples [00:54, 874.16 examples/s]48624 examples [00:54, 857.44 examples/s]48712 examples [00:54, 861.55 examples/s]48799 examples [00:54, 851.97 examples/s]48891 examples [00:54, 870.64 examples/s]48986 examples [00:54, 891.65 examples/s]49082 examples [00:55, 908.79 examples/s]49174 examples [00:55, 899.26 examples/s]49265 examples [00:55, 899.29 examples/s]49356 examples [00:55, 899.35 examples/s]49447 examples [00:55, 897.56 examples/s]49537 examples [00:55, 892.64 examples/s]49627 examples [00:55, 885.09 examples/s]49718 examples [00:55, 891.10 examples/s]49809 examples [00:55, 894.39 examples/s]49899 examples [00:55, 894.01 examples/s]49989 examples [00:56, 890.18 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5910/50000 [00:00<00:00, 59097.80 examples/s] 34%|███▍      | 17039/50000 [00:00<00:00, 68773.06 examples/s] 57%|█████▋    | 28336/50000 [00:00<00:00, 77916.27 examples/s] 79%|███████▉  | 39498/50000 [00:00<00:00, 85676.35 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.04 examples/s]161 examples [00:00, 749.87 examples/s]256 examples [00:00, 798.36 examples/s]351 examples [00:00, 836.65 examples/s]442 examples [00:00, 856.09 examples/s]530 examples [00:00, 861.49 examples/s]624 examples [00:00, 882.27 examples/s]718 examples [00:00, 898.17 examples/s]811 examples [00:00, 906.64 examples/s]906 examples [00:01, 916.96 examples/s]998 examples [00:01, 917.07 examples/s]1090 examples [00:01, 915.88 examples/s]1181 examples [00:01, 908.58 examples/s]1272 examples [00:01, 905.81 examples/s]1364 examples [00:01, 908.29 examples/s]1455 examples [00:01, 899.84 examples/s]1547 examples [00:01, 903.24 examples/s]1639 examples [00:01, 907.82 examples/s]1732 examples [00:01, 912.56 examples/s]1824 examples [00:02, 909.00 examples/s]1915 examples [00:02, 894.05 examples/s]2005 examples [00:02, 890.50 examples/s]2096 examples [00:02, 896.21 examples/s]2186 examples [00:02, 893.07 examples/s]2277 examples [00:02, 897.90 examples/s]2367 examples [00:02, 898.47 examples/s]2457 examples [00:02, 889.64 examples/s]2549 examples [00:02, 896.35 examples/s]2642 examples [00:02, 904.92 examples/s]2733 examples [00:03, 904.33 examples/s]2824 examples [00:03, 889.86 examples/s]2919 examples [00:03, 906.90 examples/s]3011 examples [00:03, 909.75 examples/s]3103 examples [00:03, 909.13 examples/s]3196 examples [00:03, 913.54 examples/s]3288 examples [00:03, 911.06 examples/s]3382 examples [00:03, 917.55 examples/s]3474 examples [00:03, 903.57 examples/s]3565 examples [00:03, 896.91 examples/s]3655 examples [00:04, 895.07 examples/s]3745 examples [00:04, 893.25 examples/s]3836 examples [00:04, 896.13 examples/s]3926 examples [00:04, 890.76 examples/s]4016 examples [00:04, 884.28 examples/s]4105 examples [00:04, 884.05 examples/s]4194 examples [00:04, 879.52 examples/s]4284 examples [00:04, 883.31 examples/s]4373 examples [00:04, 881.80 examples/s]4462 examples [00:04, 883.85 examples/s]4555 examples [00:05, 896.72 examples/s]4648 examples [00:05, 903.94 examples/s]4739 examples [00:05, 893.44 examples/s]4829 examples [00:05, 888.98 examples/s]4918 examples [00:05, 879.19 examples/s]5010 examples [00:05, 890.03 examples/s]5100 examples [00:05, 879.70 examples/s]5190 examples [00:05, 885.61 examples/s]5280 examples [00:05, 889.84 examples/s]5371 examples [00:05, 895.49 examples/s]5464 examples [00:06, 905.31 examples/s]5555 examples [00:06, 899.36 examples/s]5645 examples [00:06, 894.17 examples/s]5737 examples [00:06, 899.57 examples/s]5830 examples [00:06, 907.61 examples/s]5924 examples [00:06, 914.04 examples/s]6016 examples [00:06, 915.69 examples/s]6113 examples [00:06, 930.41 examples/s]6207 examples [00:06, 931.70 examples/s]6301 examples [00:07, 873.33 examples/s]6394 examples [00:07, 889.31 examples/s]6488 examples [00:07, 903.34 examples/s]6579 examples [00:07, 885.13 examples/s]6670 examples [00:07, 890.48 examples/s]6761 examples [00:07, 895.12 examples/s]6853 examples [00:07, 900.74 examples/s]6944 examples [00:07, 895.63 examples/s]7034 examples [00:07, 889.07 examples/s]7124 examples [00:07, 889.26 examples/s]7214 examples [00:08, 891.50 examples/s]7306 examples [00:08, 899.82 examples/s]7397 examples [00:08, 898.78 examples/s]7487 examples [00:08, 898.47 examples/s]7579 examples [00:08, 902.71 examples/s]7670 examples [00:08, 891.46 examples/s]7760 examples [00:08, 876.62 examples/s]7850 examples [00:08, 882.89 examples/s]7939 examples [00:08, 883.86 examples/s]8029 examples [00:08, 885.54 examples/s]8121 examples [00:09, 895.52 examples/s]8213 examples [00:09, 902.69 examples/s]8304 examples [00:09, 902.46 examples/s]8399 examples [00:09, 914.33 examples/s]8494 examples [00:09, 924.20 examples/s]8587 examples [00:09, 888.13 examples/s]8677 examples [00:09, 889.89 examples/s]8767 examples [00:09, 891.24 examples/s]8861 examples [00:09, 902.93 examples/s]8954 examples [00:09, 909.54 examples/s]9046 examples [00:10, 911.38 examples/s]9138 examples [00:10, 906.67 examples/s]9229 examples [00:10, 905.96 examples/s]9327 examples [00:10, 925.14 examples/s]9423 examples [00:10, 933.98 examples/s]9517 examples [00:10, 929.12 examples/s]9610 examples [00:10, 921.04 examples/s]9703 examples [00:10, 914.95 examples/s]9795 examples [00:10, 916.11 examples/s]9889 examples [00:10, 921.42 examples/s]9982 examples [00:11, 917.05 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO8FBGV/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO8FBGV/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9e90dcf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9e90dcf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9e90dcf0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9edc7ad7b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e2c09a940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9e90dcf0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9e90dcf0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9e90dcf0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e1af5e0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e77fb94a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:01,  2.98 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:01,  2.98 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  2.98 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  3.57 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  3.57 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.97 file/s]2020-08-15 00:09:09.306215: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-15 00:09:09.310647: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-15 00:09:09.311444: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b6f8798aa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-15 00:09:09.311461: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 458974.36it/s] 51%|█████     | 5062656/9912422 [00:00<00:07, 653097.58it/s]9920512it [00:00, 38990166.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 564178.47it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485023.88it/s]1654784it [00:00, 12312063.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 233739.72it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
